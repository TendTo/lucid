/**
 * @typedef {object} HLine
 * @property {number} y - The y-coordinate of the line.
 * @property {string|undefined} c - The color of the line.
 * @property {"dashed" | "solid" | undefined} s - The style of the line (e.g., 'dash', 'solid').
 *
 * @typedef {Object} VLine
 * @property {number} x - The x-coordinate of the horizontal line.
 * @property {string|undefined} c - The color of the line.
 * @property {"dashed" | "solid" | undefined} s - The style of the line (e.g., 'dash', 'solid').
 * 
 * @typedef {Object} PlotConfig
 * @property {(number) => number} f - Function to evaluate for plotting.
 * @property {number[]} xBounds - Samples for the x-axis.
 * @property {number} n - Number of samples to generate.
 * @property {number[]} xSamples - Samples for the x-axis.
 * @property {number[]} ySamples - Samples for the y-axis.
 * @property {"scatter" | "bar"} type - Type of plot (e.g., 'scatter').
 * @property {string} title - Title of the plot.
 * @property {boolean} showlegend - Whether to show the legend.
 * @property {VLine[]|undefined} vLines - Vertical lines to draw on the plot.
 * @property {VLine[]|undefined} vBoundedLines - Horizontal lines to draw on the plot.
 * @property {HLine[]|undefined} hLines - Horizontal lines to draw on the plot.
 * @property {HLine[]|undefined} hBoundedLines - Horizontal lines to draw on the plot.
 */

function plotlyJS() {
    document.querySelectorAll("div[data-plot]").forEach(function (div) {
        /** @type {PlotConfig} */
        const conf = JSON.parse(div.dataset.plot);
        if (conf.f) conf.f = eval(conf.f); // Evaluate the function string

        if (!conf.xSamples && conf.xBounds && conf.n) {
            // Generate xSamples if not provided
            conf.xSamples = [];
            const step = (conf.xBounds[1] - conf.xBounds[0]) / (conf.n - 1);
            for (let i = 0; i < conf.n; i++) {
                conf.xSamples.push(conf.xBounds[0] + i * step);
            }
        }
        if (!conf.xSamples) {
            return console.error("xSamples must be provided or generated from xBounds and n");
        }

        if (!conf.ySamples && conf.f && conf.xSamples) {
            // Generate ySamples if not provided
            conf.ySamples = conf.xSamples.map(x => conf.f(x));
        }

        const shapes = [];
        if (conf.vLines) {
            conf.vLines.forEach(l => {
                shapes.push({
                    type: 'line',
                    x0: l.x,
                    x1: l.x,
                    y0: Math.min(...conf.ySamples),
                    y1: Math.max(...conf.ySamples),
                    line: { color: l.c || "blue", width: 2, dash: l.s || 'solid' },
                });
            });
        }
        if (conf.vBoundedLines) {
            conf.vBoundedLines.forEach(l => {
                shapes.push({
                    type: 'line',
                    x0: l.x,
                    x1: l.x,
                    y0: Math.min(...conf.ySamples),
                    y1: conf.f(l.x),
                    line: { color: l.c || "blue", width: 2, dash: l.s || 'dash' },
                });
            });
        }

        const data = [{
            x: conf.xSamples,
            y: conf.ySamples,
            type: conf.type || 'scatter',
        }];

        const layout = {
            title: {
                text: conf.title || "Plot",
            },
            showlegend: conf.showlegend ?? false,
            template: "plotly_dark",
            shapes
        };

        Plotly.newPlot(div, data, layout, { scrollZoom: true, displayModeBar: false });

        div.getElementsByClassName("main-svg")[0].style.cssText = "background: rgba(0, 0, 0, 0);";
    });
}

document.addEventListener("DOMContentLoaded", plotlyJS);