import warnings

import mpld3
from flask import Flask, request
from flask_cors import CORS
from jsonschema import ValidationError
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pyparsing import ParseException

from pylucid.cli import CLIArgs, ConfigAction, arg_parser
from pylucid.plot import plot_function

warnings.filterwarnings("ignore", module="matplotlib")


def main():
    app = Flask(__name__)
    CORS(app, origins=["*"])

    @app.route("/preview-graph", methods=["GET", "POST"])
    def preview_graph():
        print(request.json)

        try:
            args: CLIArgs = arg_parser().parse_args([])  # Create an empty CLIArgs object
            config_action = ConfigAction(option_strings=None, dest="")
            config_action.validate(request.json)
            config_action.dict_to_cliargs(request.json, args)
            print(args)
        except ValidationError as e:
            return {"message": e.message, "cause": ""}, 400
        except ParseException as e:
            return {"message": f"Error parsing system dynamics. {e}", "cause": "system_dynamics"}, 400
        except Exception as e:
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
        return mpld3.fig_to_html(fig)

    app.run(debug=True, host="0.0.0.0", port=5000)
