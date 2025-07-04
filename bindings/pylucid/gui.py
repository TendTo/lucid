import logging
import secrets
import threading
import warnings
import webbrowser
from queue import Queue

import matplotlib
import mpld3
import numpy as np
from flask import Blueprint, Flask, Response, request, send_from_directory, session
from flask_cors import CORS
from jsonschema import ValidationError
from matplotlib import pyplot as plt
from pyparsing import ParseException

from pylucid import *
from pylucid.cli import ConfigAction
from pylucid.dreal import verify_barrier_certificate
from pylucid.plot import plot_function, plot_solution

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

DEBUG = True

QUEUES: "dict[int, Queue[str]]" = {}


def run_lucid(args: Configuration):
    # Define the system dynamics function
    if args.seed >= 0:
        np.random.seed(args.seed)
        random.seed(args.seed)
    log.set_verbosity(args.verbose)
    f_det = args.system_dynamics
    f = lambda x: f_det(x) + np.random.normal(scale=args.noise_scale)  # Add noise to the dynamics

    # Sample points from the bounds
    x_samples = args.X_bounds.sample(args.num_samples)
    xp_samples = f(x_samples)

    # Create the estimator
    estimator = args.estimator(
        kernel=args.kernel(sigma_f=args.sigma_f, sigma_l=args.sigma_l),
        regularization_constant=args.lambda_,
    )

    feature_map: TruncatedFourierFeatureMap = args.feature_map(
        num_frequencies=args.num_frequencies,
        sigma_l=args.sigma_l,
        sigma_f=args.sigma_f,
        x_limits=args.X_bounds,
    )

    num_freq_per_dim = feature_map.num_frequencies if args.num_frequencies < 0 else args.num_frequencies
    n_per_dim = (
        np.ceil((2 * num_freq_per_dim + 1) * args.oversample_factor) if args.num_oversample < 0 else args.num_oversample
    )
    n_per_dim = int(n_per_dim)
    log.debug(f"Number of samples per dimension: {n_per_dim}")
    assert n_per_dim > 2 * num_freq_per_dim, "n_per_dim must be greater than nyquist (2 * num_freq_per_dim + 1)"

    f_xp_samples = feature_map(xp_samples)

    log.debug(f"Estimator pre-fit: {estimator}")
    estimator.fit(x=x_samples, y=f_xp_samples)  # Actual fitting of the regressor
    log.info(f"Estimator post-fit: {estimator}")

    if callable(feature_map) and not isinstance(feature_map, FeatureMap):
        feature_map = feature_map(estimator)  # Compute the feature map if it is a callable
    assert isinstance(feature_map, FeatureMap), "feature_map must return a FeatureMap instance"

    log.debug(f"RMSE on f_xp_samples {rmse_score(estimator(x_samples), f_xp_samples)}")
    log.debug(f"Score on f_xp_samples {estimator.score(x_samples, f_xp_samples)}")
    if f_det is not None:
        # Sample some other points (half of the x_samples) to evaluate the regressor against overfitting
        x_evaluation = args.X_bounds.sample(x_samples.shape[0] // 2)
        f_xp_evaluation = feature_map(f_det(x_evaluation))
        log.debug(f"RMSE on f_det_evaluated {rmse_score(estimator(x_evaluation), f_xp_evaluation)}")
        log.debug(f"Score on f_det_evaluated {estimator.score(x_evaluation, f_xp_evaluation)}")

    log.debug(f"Feature map: {feature_map}")
    x_lattice = args.X_bounds.lattice(n_per_dim, True)
    u_f_x_lattice = feature_map(x_lattice)
    u_f_xp_lattice_via_regressor = estimator(x_lattice)
    # We are fixing the zero frequency to the constant value we computed in the feature map
    # If we don't, the regressor has a hard time learning it on the extreme left and right points, because it tends to 0
    u_f_xp_lattice_via_regressor[:, 0] = feature_map.weights[0] * args.sigma_f

    x0_lattice = args.X_init.lattice(n_per_dim, True)
    f_x0_lattice = feature_map(x0_lattice)

    xu_lattice = args.X_unsafe.lattice(n_per_dim, True)
    f_xu_lattice = feature_map(xu_lattice)

    def check_cb(success: bool, obj_val: float, sol: "NVector", eta: float, c: float, norm: float):
        response = {
            "success": False,
            "obj_val": None,
            "sol": None,
            "eta": None,
            "c": None,
            "norm": None,
            "fig": None,
            "error": None,
        }
        if not success:
            log.error("Optimization failed")
            response["error"] = "Optimization failed"
            response["success"] = False
        else:
            log.info("Optimization succeeded")
            log.debug(f"{obj_val = }, {eta = }, {c = }, {norm = }")
            log.debug(f"{sol = }")
            response["success"] = True
            response["obj_val"] = obj_val
            response["sol"] = sol.tolist()
            response["eta"] = eta
            response["c"] = c
            response["norm"] = norm
        if args.plot:
            plt.figure()
            fig = plot_solution(
                X_bounds=args.X_bounds,
                X_init=args.X_init,
                X_unsafe=args.X_unsafe,
                feature_map=feature_map,
                eta=eta if success is None else None,
                gamma=args.gamma,
                sol=sol if success else None,
                f=f_det,
                estimator=estimator,
                c=c if success else None,
                show=False,
                num_samples=n_per_dim,
            )
            response["fig"] = mpld3.fig_to_html(fig)

        if args.verify and f_det is not None and success:
            response["verified"] = verify_barrier_certificate(
                X_bounds=args.X_bounds,
                X_init=args.X_init,
                X_unsafe=args.X_unsafe,
                sigma_f=args.sigma_f,
                eta=eta,
                c=c,
                f_det=f_det,
                gamma=args.gamma,
                estimator=estimator,
                tffm=feature_map,
                sol=sol,
            )
        QUEUES[threading.get_ident()].put(response)

    files = {
        "problem_log_file": args.problem_log_file,
        "iis_log_file": args.iis_log_file,
    }

    log.info(f"Running optimiser")
    try:
        v = args.optimiser(
            args.time_horizon,
            args.gamma,
            0.0,
            1.0,
            b_kappa=1.0,
            C_coeff=args.c_coefficient,
            sigma_f=args.sigma_f,
            **(files if isinstance(args.optimiser, GurobiOptimiser) else {}),
        ).solve(
            f0_lattice=f_x0_lattice,
            fu_lattice=f_xu_lattice,
            phi_mat=u_f_x_lattice,
            w_mat=u_f_xp_lattice_via_regressor,
            rkhs_dim=feature_map.dimension,
            num_frequencies_per_dim=num_freq_per_dim - 1,
            num_frequency_samples_per_dim=n_per_dim,
            original_dim=args.X_bounds.dimension,
            callback=check_cb,
        )
    except Exception as e:
        log.error(f"Error during optimisation: {e}")
        raise e
    finally:
        QUEUES[threading.get_ident()].put(None)


def handle_log(log_entry: str):
    """Handle log entries by putting them into the queue for the current thread."""
    QUEUES.setdefault(threading.get_ident(), Queue(maxsize=1)).put({"log": log_entry})
    print(log_entry, end="")  # Print to console for debugging


def get_args(config_dict: "dict | None" = None) -> Configuration:
    config_dict = config_dict or request.json
    try:
        args: Configuration = Configuration()
        config_action = ConfigAction(option_strings=None, dest="")
        config_action.validate(config_dict)
        config_action.dict_to_cliargs(config_dict, args)
        # Just to ensure the system dynamics function is compatible with the initial state
        args.system_dynamics(args.X_init.lattice(1))
        logger.debug("Parsed CLI arguments: %s", args)
    except ValidationError as val_err:
        logger.error("Validation error: %s", val_err.message)
        return {"message": val_err.message}, 400
    except ParseException as parse_err:
        logger.error("Parse error: %s", parse_err)
        return {"message": f"Error parsing system dynamics. {parse_err}", "cause": "system_dynamics"}, 400
    except TypeError as type_err:
        if "_lambdifygenerated()" in str(type_err):
            return {"message": str(type_err).split("_lambdifygenerated()")[1], "cause": "system_dynamics"}, 400
        return {"message": f"Error processing configuration. {type_err}"}, 500
    except Exception as e:
        logger.error(f"Error processing configuration: {e}")
        return {"message": f"Error processing configuration. {e}"}, 500
    return args


def event_streamer(worker_id: int):
    q = QUEUES.setdefault(worker_id, Queue(maxsize=1))
    while True:
        log_entry: str = q.get()
        if log_entry is None:
            break
        q.task_done()
        yield f"data: {json.dumps(log_entry)}\n\n"
    del QUEUES[worker_id]  # Clean up the queue after the worker is done


blueprint = Blueprint("pylucid", __name__, static_folder="frontend", static_url_path="")


@blueprint.route("/preview-graph", methods=["POST"])
def preview_graph():
    logger.info("Received request to preview graph.")
    args = get_args()
    if not isinstance(args, Namespace):
        return args
    plt.figure()
    fig = plot_function(
        X_bounds=args.X_bounds,
        X_init=args.X_init,
        X_unsafe=args.X_unsafe,
        f=args.system_dynamics,
        show=False,
    )
    logger.info("Graph preview generated successfully.")
    return {"fig": mpld3.fig_to_html(fig)}, 200


@blueprint.route("/run", methods=["POST"])
def post_run():
    logger.info("Received request to run lucid. Storing config_dict in session.")
    session["config_dict"] = request.json
    return Response(status=202)


@blueprint.route("/run", methods=["GET"])
def get_run():
    logger.info("Received request to get run status.")
    config_dict = session.get("config_dict", None)
    print(session)
    if config_dict is None:
        return {"message": "You must submit config_dict before starting a run"}, 404
    args = get_args(config_dict)
    if not isinstance(args, Namespace):
        return args
    # Implement logic to retrieve the status of the run using the UUID
    worker = threading.Thread(target=run_lucid, args=[args], daemon=True)
    worker.start()

    session.pop("args", None)  # Clear the args after starting the run
    return Response(event_streamer(worker.ident), mimetype="text/event-stream")


def main():
    app = Flask(__name__, static_folder="frontend", static_url_path="")
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex())
    app.register_blueprint(blueprint, url_prefix="/api")
    CORS(app)
    log.set_sink(handle_log)

    @app.route("/", methods=["GET"])
    def index():
        logger.info("Received request for index page.")
        return send_from_directory("frontend", "index.html")

    if not DEBUG:
        log.info("Opening the app in the default web browser.")
        # Open the app in the default web browser
        webbrowser.open("http://localhost:5000", new=2)  # Open the app in the default web browser

    app.run(debug=DEBUG, host="0.0.0.0", port=5000)
