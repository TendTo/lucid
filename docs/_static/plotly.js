/**
 * @typedef {object} HLine
 * @property {number} y - The y-coordinate of the line.
 * @property {string|undefined} c - The color of the line.
 * @property {"dashed" | "solid" | undefined} s - The style of the line (e.g., 'dash', 'solid').
 * @property {string|undefined} label - Label for the line.
 *
 * @typedef {Object} VLine
 * @property {number} x - The x-coordinate of the horizontal line.
 * @property {number|"full"|undefined} yMax - The maximum y-coordinate for bounded lines. Use "full" to extend fully, otherwise defaults to function value.
 * @property {string|undefined} c - The color of the line.
 * @property {"dashed" | "solid" | undefined} s - The style of the line (e.g., 'dash', 'solid').
 * @property {string|undefined} label - Label for the line.
 * 
 * @typedef {Rectangle} Rectangle
 * @property {number} x0 - The starting x-coordinate of the rectangle.
 * @property {number} x1 - The ending x-coordinate of the rectangle.
 * @property {number} y0 - The starting y-coordinate of the rectangle.
 * @property {number|"full"|undefined} y1 - The ending y-coordinate of the rectangle for bounded rectangles. Use "full" to extend fully, otherwise defaults to function value.
 * @property {string|undefined} c - The color of the rectangle.
 * @property {number|undefined} opacity - The opacity of the rectangle (0 to 1).
 * @property {string|undefined} label - Label for the rectangle.
 * 
 * @typedef {Annotation} Annotation
 * @property {number} x - The x-coordinate of the annotation.
 * @property {number} y - The y-coordinate of the annotation.
 * @property {string} text - The text of the annotation.
 * 
 * @typedef {Gradient} Gradient
 * @property {[number, string][]} colorscale - Array of [position, color] pairs defining the gradient.
 *
 * @typedef {Object} PlotConfig
 * @property {(number) => number} f - Function to evaluate for plotting.
 * @property {number[]} xBounds - Samples for the x-axis.
 * @property {number} n - Number of samples to generate.
 * @property {string|[number, number]} fill - Fill under the curve (true for all, or array of [x0, x1] ranges).
 * @property {string|string[]|undefined} fillColor - Fill color(s) for the area under the curve.
 * @property {number[]} xSamples - Samples for the x-axis.
 * @property {number[]} ySamples - Samples for the y-axis.
 * @property {"scatter" | "bar"} type - Type of plot (e.g., 'scatter').
 * @property {string} title - Title of the plot.
 * @property {boolean} showlegend - Whether to show the legend.
 * @property {VLine[]|undefined} vLines - Vertical lines to draw on the plot.
 * @property {HLine[]|undefined} hLines - Horizontal lines to draw on the plot.
 * @property {Rectangle[]|undefined} rectangles - Rectangles to draw on the plot.
 * @property {Annotation[]|undefined} annotations - Annotations to add to the plot.
 * @property {Gradient|undefined} fillGradientH - Gradient fill under the curve horizontally.
 * @property {Gradient|undefined} fillGradientV - Gradient fill under the curve vertically.
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
        const annotations = [];
        (conf.vLines ?? []).forEach(l => {
            shapes.push({
                type: 'line',
                x0: l.x,
                x1: l.x,
                y0: Math.min(...conf.ySamples),
                y1: l.yMax === "full" ? Math.max(...conf.ySamples) : l.yMax ?? conf.f(l.x),
                line: { color: l.c || "blue", width: 2, dash: l.s || 'solid' },
                name: l.label || undefined
            });
            if (l.label) {
                annotations.push({
                    x: l.x,
                    y: -0.15,
                    yref: 'paper',
                    text: l.label,
                    showarrow: false,
                    font: {
                        size: 16,
                        weight: 'bold',
                    },
                    align: "center",
                });
            }
        });

        (conf.rectangles ?? []).forEach(r => {
            shapes.push({
                type: 'rect',
                x0: r.x0,
                x1: r.x1,
                y0: r.y0,
                y1: r.y1 === "full" ? Math.max(...conf.ySamples) : r.y1 ?? conf.f((r.x0 + r.x1) / 2),
                fillcolor: r.c || "blue",
                opacity: r.opacity || 0.2,
                line: { width: 0 },
                name: r.label || undefined
            });
            if (r.label) {
                annotations.push({
                    x: (r.x0 + r.x1) / 2,
                    y: (r.y0 + (r.y1 === "full" ? Math.max(...conf.ySamples) : r.y1 ?? conf.f((r.x0 + r.x1) / 2))) / 2,
                    text: r.label,
                    showarrow: true,
                    font: {
                        size: 16,
                        weight: 'bold',
                    },
                    align: "center",
                });
            }
        });

        (conf.annotations ?? []).forEach(a => {
            annotations.push({
                x: a.x,
                y: a.y,
                text: a.text,
                showarrow: true,
                font: {
                    size: 16,
                    weight: 'bold',
                },
                align: "center",
            });
        });

        const data = [{
            x: conf.xSamples,
            y: conf.ySamples,
            type: conf.type ?? 'scatter',
            fill: conf.fill ?? 'none',
            mode: 'lines',
            ...(conf.fillColor ? { fillcolor: conf.fillColor } : {}),
            ...((conf.fillGradientH) ? {
                fillgradient: {
                    type: "horizontal",
                    colorscale: conf.fillGradientH,
                }
            } : {}),
            ...((conf.fillGradientV) ? {
                fillgradient: {
                    type: "vertical",
                    colorscale: conf.fillGradientV,
                }
            } : {}),
        }];

        console.log(data);

        const layout = {
            title: {
                text: conf.title,
            },
            showlegend: conf.showlegend ?? false,
            template: "plotly_dark",
            shapes,
            annotations,
        };

        Plotly.newPlot(div, data, layout, { scrollZoom: true, displayModeBar: false });

        div.getElementsByClassName("main-svg")[0].style.cssText = "background: rgba(0, 0, 0, 0);";
    });
}

document.addEventListener("DOMContentLoaded", plotlyJS);