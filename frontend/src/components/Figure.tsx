import Plot, { type PlotParams } from "react-plotly.js";

type FigureProps = PlotParams;

export default function Figure(params: FigureProps) {
  return params.data.length > 0 ? <Plot {...params} /> : null;
}
