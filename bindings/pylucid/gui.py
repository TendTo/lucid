from flask import Flask, request
from flask_cors import CORS

from matplotlib.figure import Figure
import mpld3

from pylucid.cli import ConfigAction, arg_parser
from jsonschema import ValidationError


def main():
    app = Flask(__name__)
    CORS(app, origins=["*"])

    @app.route("/preview-graph", methods=["GET", "POST"])
    def preview_graph():
        print(request.json)

        try:
            args = arg_parser().parse_args([])  # Create an empty CLIArgs object
            ConfigAction(option_strings=None, dest="").dict_to_cliargs(request.json, args)
            print(args)
        except ValidationError:
            return "Invalid configuration", 400
        except Exception as e:
            return f"Error processing configuration: {str(e)}", 500
        # Generate the figure **without using pyplot**.
        fig = Figure()
        ax = fig.subplots()
        ax.plot([1, 2])
        return mpld3.fig_to_html(fig)

    app.run(debug=True, host="0.0.0.0", port=5000)
