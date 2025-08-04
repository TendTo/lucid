import argparse
import json
import logging
import os
import secrets
import threading
import time
import webbrowser
from queue import Queue

import numpy as np
from cachelib import FileSystemCache
from flask import Blueprint, Flask, Response, request, send_from_directory, session
from flask_cors import CORS
from jsonschema import ValidationError
from pyparsing import ParseException

from flask_session import Session

from . import CAPABILITIES
from .__main__ import scenario_config
from ._pylucid import *
from .cli import ConfigAction, Configuration
from .pipeline import OptimiserResult, pipeline
from .plot import plot_data, plot_function

logger = logging.getLogger(__name__)

QUEUES: "dict[int, Queue[str]]" = {}


def run_lucid(config: Configuration):
    # Define the system dynamics function
    if config.seed >= 0:
        np.random.seed(config.seed)
        random.seed(config.seed)
    log.set_verbosity(config.verbose)

    def optimiser_cb(result: "OptimiserResult"):
        if not result["success"]:
            result["error"] = "Optimization failed"
        if isinstance(result["sol"], np.ndarray):
            result["sol"] = result["sol"].tolist()
        QUEUES[threading.get_ident()].put(result)

    def plot_cb(fig: "Figure"):
        if fig is not None:
            fig_json = fig.to_json(validate=False)
            QUEUES[threading.get_ident()].put({"fig": fig_json})
            time.sleep(0.01)  # Give time for the figure to be processed

    def check_cb(verified: bool):
        QUEUES[threading.get_ident()].put({"verified": verified})

    try:
        pipeline(scenario_config(config), show=False, optimiser_cb=optimiser_cb, plot_cb=plot_cb, verify_cb=check_cb)
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


def get_config(config_dict: "dict | None" = None) -> Configuration:
    config_dict = config_dict or request.json
    try:
        config: Configuration = Configuration()
        config_action = ConfigAction(option_strings=None, dest="")
        config_action.validate(config_dict)
        config_action.dict_to_configuration(config_dict, config)
        # Just to ensure the system dynamics function is compatible with the initial state
        if config.system_dynamics is None and len(config.xp_samples) == 0:
            return {
                "error": "System dynamics must be provided if xp_samples is not given.",
                "cause": "system_dynamics",
            }, 400
        if config.system_dynamics is not None:
            config.system_dynamics(config.X_init.lattice(1))
        logger.debug("Parsed CLI arguments: %s", config)
    except ValidationError as val_err:
        logger.error("Validation error: %s", val_err.message)
        return {"error": val_err.message}, 400
    except ParseException as parse_err:
        logger.error("Parse error: %s", parse_err)
        return {"error": f"Error parsing system dynamics. {parse_err}", "cause": "system_dynamics"}, 400
    except TypeError as type_err:
        if "_lambdifygenerated()" in str(type_err):
            return {"error": str(type_err).split("_lambdifygenerated()")[1], "cause": "system_dynamics"}, 400
        return {"error": f"Error processing configuration. {type_err}"}, 500
    except Exception as e:
        logger.error(f"Error processing configuration: {e}")
        return {"error": f"Error processing configuration. {e}"}, 500
    return config


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


@blueprint.route("/capabilities", methods=["GET"])
def capabilities():
    return CAPABILITIES, 200


@blueprint.route("/preview-graph", methods=["POST"])
def preview_graph():
    logger.info("Received request to preview graph.")
    config = get_config()
    if not isinstance(config, Configuration):
        return config
    if config.system_dynamics is not None:
        try:
            fig = plot_function(
                X_bounds=config.X_bounds,
                X_init=config.X_init,
                X_unsafe=config.X_unsafe,
                f=config.system_dynamics,
                show=False,
            )
        except ValueError as ve:
            logger.error(f"Could not plot function: {ve}")
            return {"error": f"Could not plot function: {ve}"}, 400
    elif len(config.x_samples) > 0 and (len(config.xp_samples) > 0 or config.system_dynamics is not None):
        if len(config.xp_samples) == 0 and config.system_dynamics is not None:
            # If xp_samples is not provided, compute it using the system dynamics function
            config.xp_samples = config.system_dynamics(config.x_samples)
        fig = plot_data(
            X_bounds=config.X_bounds,
            X_init=config.X_init,
            X_unsafe=config.X_unsafe,
            x_samples=config.x_samples,
            xp_samples=config.xp_samples,
            show=False,
        )
    else:
        return {"error": "No system dynamics or samples provided for graph preview."}, 400
    logger.info("Graph preview generated successfully.")
    return {"fig": fig.to_json(validate=False)}, 200


@blueprint.route("/run", methods=["POST"])
def post_run():
    logger.info("Received request to run lucid. Storing config_dict in session.")
    config = get_config()
    if not isinstance(config, Configuration):
        return config
    session["config_dict"] = request.json
    return Response(status=202)


@blueprint.route("/run", methods=["GET"])
def get_run():
    logger.info("Received request to get run status.")
    config_dict = session.get("config_dict", None)
    if config_dict is None:
        return {"error": "You must submit config_dict before starting a run"}, 404
    config = get_config(config_dict)
    if not isinstance(config, Configuration):
        return config
    # Implement logic to retrieve the status of the run using the UUID
    worker = threading.Thread(target=run_lucid, args=[config], daemon=True)
    worker.start()

    session.pop("config_dict", None)  # Clear the config after starting the run
    return Response(event_streamer(worker.ident), mimetype="text/event-stream")


class CliArgs(argparse.Namespace):
    debug: bool
    host: str
    port: int
    cache_dir: str
    threshold: int


def parse_args(args: "list[str] | None" = None) -> CliArgs:
    parser = argparse.ArgumentParser(description="Run the PyLucid GUI.")
    parser.add_argument(
        "--host",
        type=str,
        help="Host to run the application on.",
        default="0.0.0.0",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        help="Port to run the application on.",
        default=3661,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run the app in debug mode.",
    )
    parser.add_argument(
        "-c",
        "--cache-dir",
        type=str,
        help="Directory to store cache files.",
        default="flask_session",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=int,
        help="Threshold for cache size.",
        default=500,
    )
    return parser.parse_args(args=args)


def main():
    args = parse_args()
    app = Flask(__name__, static_folder="frontend", static_url_path="")
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex())
    app.config["SESSION_TYPE"] = "cachelib"
    app.config["SESSION_CACHELIB"] = FileSystemCache(cache_dir=args.cache_dir, threshold=args.threshold)
    Session(app)
    app.register_blueprint(blueprint, url_prefix="/api")

    @app.route("/", methods=["GET"])
    def index():
        logger.info("Received request for index page.")
        return send_from_directory("frontend", "index.html")

    CORS(app)
    # Setup the logging configuration so that logs can be captured and sent to the frontend
    if not args.debug:
        log.info("Opening the app in the default web browser.")
        # Open the app in the default web browser
        webbrowser.open(f"http://localhost:{args.port}", new=2)  # Open the app in the default web browser

    log.set_pattern("[%Y-%m-%d %H:%M:%S] [%l] %v")  # Set the log pattern
    log.set_sink(handle_log)
    app.run(debug=args.debug, host=args.host, port=args.port)
