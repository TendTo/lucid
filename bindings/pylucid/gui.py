import json
import logging
import os
import secrets
import threading
import webbrowser
from queue import Queue

import numpy as np
from flask import Blueprint, Flask, Response, request, send_from_directory, session
from flask_cors import CORS
from jsonschema import ValidationError
from pyparsing import ParseException

from .__main__ import scenario_config
from ._pylucid import *
from .cli import ConfigAction, Configuration
from .pipeline import OptimiserResult, pipeline
from .plot import plot_function

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEBUG = True

QUEUES: "dict[int, Queue[str]]" = {}


def run_lucid(args: Configuration):
    # Define the system dynamics function
    if args.seed >= 0:
        np.random.seed(args.seed)
        random.seed(args.seed)
    log.set_verbosity(args.verbose)

    def check_cb(result: "OptimiserResult"):
        if not result["success"]:
            result["error"] = "Optimization failed"
        else:
            result["sol"] = result["sol"].tolist()
        if result["fig"] is not None:
            result["fig"] = result["fig"].to_html(include_plotlyjs=False, full_html=False)
        QUEUES[threading.get_ident()].put(result)

    try:
        pipeline(scenario_config(args), show=False, optimiser_cb=check_cb)
    except Exception as e:
        log.error(f"Error during optimisation: {e}")
        raise e
    finally:
        QUEUES[threading.get_ident()].put(None)


def handle_log(log_entry: str):
    """Handle log entries by putting them into the queue for the current thread."""
    q = QUEUES.setdefault(threading.get_ident(), Queue(maxsize=1))
    q.put({"log": log_entry})
    print(log_entry, end="")  # Print to console for debugging
    q.join()  # Ensure the log entry is processed before continuing


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
        yield f"data: {json.dumps(log_entry)}\n\n"
        q.task_done()
    del QUEUES[worker_id]  # Clean up the queue after the worker is done


blueprint = Blueprint("pylucid", __name__, static_folder="frontend", static_url_path="")


@blueprint.route("/preview-graph", methods=["POST"])
def preview_graph():
    logger.info("Received request to preview graph.")
    args = get_args()
    if not isinstance(args, Configuration):
        return args
    fig = plot_function(
        X_bounds=args.X_bounds,
        X_init=args.X_init,
        X_unsafe=args.X_unsafe,
        f=args.system_dynamics,
        show=False,
    )
    logger.info("Graph preview generated successfully.")
    return {"fig": fig.to_html(include_plotlyjs=False, full_html=False)}, 200


@blueprint.route("/run", methods=["POST"])
def post_run():
    logger.info("Received request to run lucid. Storing config_dict in session.")
    session["config_dict"] = request.json
    return Response(status=202)


@blueprint.route("/run", methods=["GET"])
def get_run():
    logger.info("Received request to get run status.")
    config_dict = session.get("config_dict", None)
    if config_dict is None:
        return {"message": "You must submit config_dict before starting a run"}, 404
    args = get_args(config_dict)
    if not isinstance(args, Configuration):
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
    log.set_pattern("[%Y-%m-%d %H:%M:%S] [%l] %v")  # Set the log pattern

    @app.route("/", methods=["GET"])
    def index():
        logger.info("Received request for index page.")
        return send_from_directory("frontend", "index.html")

    if not DEBUG:
        log.info("Opening the app in the default web browser.")
        # Open the app in the default web browser
        webbrowser.open("http://localhost:5000", new=2)  # Open the app in the default web browser

    app.run(debug=DEBUG, host="0.0.0.0", port=5000)
