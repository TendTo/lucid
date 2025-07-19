import argparse
import json
import logging
import os
import secrets
import threading
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


def run_lucid(args: Configuration):
    # Define the system dynamics function
    if args.seed >= 0:
        np.random.seed(args.seed)
        random.seed(args.seed)
    log.set_verbosity(args.verbose)

    def check_cb(result: "OptimiserResult"):
        if not result["success"]:
            result["error"] = "Optimization failed"
        if isinstance(result["sol"], np.ndarray):
            result["sol"] = result["sol"].tolist()
        if result["fig"] is not None:
            result["fig"] = result["fig"].to_json(validate=False)
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
        config_action.dict_to_configuration(config_dict, args)
        # Just to ensure the system dynamics function is compatible with the initial state
        if args.system_dynamics is None and len(args.xp_samples) == 0:
            return {
                "error": "System dynamics must be provided if xp_samples is not given.",
                "cause": "system_dynamics",
            }, 400
        if args.system_dynamics is not None:
            args.system_dynamics(args.X_init.lattice(1))
        logger.debug("Parsed CLI arguments: %s", args)
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


@blueprint.route("/capabilities", methods=["GET"])
def capabilities():
    return CAPABILITIES, 200


@blueprint.route("/preview-graph", methods=["POST"])
def preview_graph():
    logger.info("Received request to preview graph.")
    args = get_args()
    if not isinstance(args, Configuration):
        return args
    if args.system_dynamics is not None:
        try:
            fig = plot_function(
                X_bounds=args.X_bounds,
                X_init=args.X_init,
                X_unsafe=args.X_unsafe,
                f=args.system_dynamics,
                show=False,
            )
        except ValueError as ve:
            logger.error(f"Could not plot function: {ve}")
            return {"error": f"Could not plot function: {ve}"}, 400
    elif len(args.x_samples) > 0 and (len(args.xp_samples) > 0 or args.system_dynamics is not None):
        if len(args.xp_samples) == 0 and args.system_dynamics is not None:
            # If xp_samples is not provided, compute it using the system dynamics function
            args.xp_samples = args.system_dynamics(args.x_samples)
        fig = plot_data(
            X_bounds=args.X_bounds,
            X_init=args.X_init,
            X_unsafe=args.X_unsafe,
            x_samples=args.x_samples,
            xp_samples=args.xp_samples,
            show=False,
        )
    else:
        return {"error": "No system dynamics or samples provided for graph preview."}, 400
    logger.info("Graph preview generated successfully.")
    return {"fig": fig.to_json(validate=False)}, 200


@blueprint.route("/run", methods=["POST"])
def post_run():
    logger.info("Received request to run lucid. Storing config_dict in session.")
    args = get_args()
    if not isinstance(args, Configuration):
        return args
    session["config_dict"] = request.json
    return Response(status=202)


@blueprint.route("/run", methods=["GET"])
def get_run():
    logger.info("Received request to get run status.")
    config_dict = session.get("config_dict", None)
    if config_dict is None:
        return {"error": "You must submit config_dict before starting a run"}, 404
    args = get_args(config_dict)
    if not isinstance(args, Configuration):
        return args
    # Implement logic to retrieve the status of the run using the UUID
    worker = threading.Thread(target=run_lucid, args=[args], daemon=True)
    worker.start()

    session.pop("args", None)  # Clear the args after starting the run
    return Response(event_streamer(worker.ident), mimetype="text/event-stream")


class CliArgs(argparse.Namespace):
    release: bool
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
        default=5000,
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help="Run the app in release mode.",
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
    return parser.parse_args()


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
    log.set_sink(handle_log)
    log.set_pattern("[%Y-%m-%d %H:%M:%S] [%l] %v")  # Set the log pattern

    if args.release:
        log.info("Opening the app in the default web browser.")
        # Open the app in the default web browser
        webbrowser.open("http://localhost:5000", new=2)  # Open the app in the default web browser

    app.run(debug=not args.release, host=args.host, port=args.port)
