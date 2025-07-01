import warnings

import mpld3
from flask import Flask, request, send_from_directory
from flask_cors import CORS
from jsonschema import ValidationError
from matplotlib import pyplot as plt
from pyparsing import ParseException

from pylucid.cli import CLIArgs, ConfigAction, arg_parser
from pylucid.plot import plot_function
from pylucid import log
import webbrowser
import warnings
import matplotlib


warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

DEBUG = True


def main():
    app = Flask(__name__, static_folder="frontend", static_url_path="")
    CORS(app, origins=["*"])

    @app.route("/", methods=["GET"])
    def index():
        log.info("Received request for index page.")
        return send_from_directory("frontend", "index.html")

    @app.route("/preview-graph", methods=["GET", "POST"])
    def preview_graph():
        log.info("Received request to preview graph.")
        try:
            args: CLIArgs = arg_parser().parse_args([])  # Create an empty CLIArgs object
            config_action = ConfigAction(option_strings=None, dest="")
            config_action.validate(request.json)
            config_action.dict_to_cliargs(request.json, args)
            log.debug(f"Parsed CLI arguments: {args}")
        except ValidationError as e:
            log.error(f"Validation error: {e.message}")
            return {"message": e.message, "cause": ""}, 400
        except ParseException as e:
            log.error(f"Parse error: {e}")
            return {"message": f"Error parsing system dynamics. {e}", "cause": "system_dynamics"}, 400
        except Exception as e:
            log.error(f"Error processing configuration: {e}")
            return {"message": f"Error processing configuration. {e}", "cause": ""}, 500
        # Generate the figure **without using pyplot**.
        plt.figure()
        try:
            fig = plot_function(
                X_bounds=args.X_bounds,
                X_init=args.X_init,
                X_unsafe=args.X_unsafe,
                f=args.system_dynamics,
                show=False,
            )
        except TypeError as e:
            log.error(f"Type error in system dynamics: {e}")
            if "_lambdifygenerated()" in str(e):
                return {"message": str(e).split("_lambdifygenerated()")[1], "cause": "system_dynamics"}, 400
            return {"message": f"Error processing configuration. {e}", "cause": ""}, 500
        log.info("Graph preview generated successfully.")
        return mpld3.fig_to_html(fig)

    @app.route("/run", methods=["GET", "POST"])
    def run():
        log.info("Received request to run lucid.")
        try:
            args: CLIArgs = arg_parser().parse_args([])  # Create an empty CLIArgs object
            config_action = ConfigAction(option_strings=None, dest="")
            config_action.validate(request.json)
            config_action.dict_to_cliargs(request.json, args)
            log.debug(f"Parsed CLI arguments: {args}")
        except ValidationError as e:
            log.error(f"Validation error: {e.message}")
            return {"message": e.message, "cause": ""}, 400
        except ParseException as e:
            log.error(f"Parse error: {e}")
            return {"message": f"Error parsing system dynamics. {e}", "cause": "system_dynamics"}, 400
        except Exception as e:
            log.error(f"Error processing configuration: {e}")
            return {"message": f"Error processing configuration. {e}", "cause": ""}, 500
        # Generate the figure **without using pyplot**.
        plt.figure()
        try:
            fig = plot_function(
                X_bounds=args.X_bounds,
                X_init=args.X_init,
                X_unsafe=args.X_unsafe,
                f=args.system_dynamics,
                show=False,
            )
        except TypeError as e:
            if "_lambdifygenerated()" in str(e):
                return {"message": str(e).split("_lambdifygenerated()")[1], "cause": "system_dynamics"}, 400
            return {"message": f"Error processing configuration. {e}", "cause": ""}, 500

        log.info("Lucid run completed successfully.")
        return {
            "graph": mpld3.fig_to_html(fig),
            "message": "Lucid run completed successfully.",
            "safety_probability": 0.95,
        }

    if not DEBUG:
        log.info("Opening the app in the default web browser.")
        # Open the app in the default web browser
        webbrowser.open("http://localhost:5000", new=2)  # Open the app in the default web browser
    app.run(debug=DEBUG, host="0.0.0.0", port=5000)
