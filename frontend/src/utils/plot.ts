import type { Annotations, Data, Layout, Shape } from "plotly.js";
import type { PlotParams } from "react-plotly.js";

interface PlotPreviewData {
  x_lattice: number[][];
  xp_lattice: number[][];
}

interface PlotSolutionData {
  x_lattice: number[][];
  B_lattice: number[];
  Bp_lattice: number[];
  Bp_lattice_est: number[];
}

interface Set {
  dimension: number;
}

export interface RectSet extends Set {
  lower_bound: number[];
  upper_bound: number[];
}

export interface SphereSet extends Set {
  center: number[];
  radius: number;
}

export interface EllipseSet extends Set {
  center: number[];
  radii: number[];
}

export interface MultiSet extends Set {
  sets: Set[];
}

// Utility functions
function validateInputs(
  x_samples: number[][] | null,
  xp_samples: number[][] | null,
  X_bounds?: RectSet | null,
  X_init?: Set | null,
  X_unsafe?: Set | null
): void {
  const dimensions = new Set<number>();
  if (X_bounds) dimensions.add(X_bounds.dimension);
  if (X_init) dimensions.add(X_init.dimension);
  if (X_unsafe) dimensions.add(X_unsafe.dimension);

  if (dimensions.size > 1) {
    throw new Error(
      "X_bounds, X_init, and X_unsafe must have the same dimension if provided."
    );
  }

  if (x_samples !== null && xp_samples !== null) {
    if (
      x_samples.length !== xp_samples.length ||
      x_samples[0]?.length !== xp_samples[0]?.length
    ) {
      throw new Error("x_samples and xp_samples must have the same shape.");
    }

    if (dimensions.size === 1) {
      const dim = Array.from(dimensions)[0];
      if (x_samples[0]?.length !== dim) {
        throw new Error(`x_samples must have ${dim} dimensions.`);
      }
    }
  }
}

function isMultiSet(set: Set): set is MultiSet {
  return "sets" in set;
}

function isRectSet(set: Set): set is RectSet {
  return "lower_bound" in set && "upper_bound" in set;
}

function isSphereSet(set: Set): set is SphereSet {
  return "center" in set && "radius" in set;
}

function isEllipseSet(set: Set): set is EllipseSet {
  return "center" in set && "radii" in set;
}

// Core plotting functions
function plotSet(
  pltFun: (set: Set, color: string, label: string, data: Data[]) => void,
  xSet: Set,
  color: string,
  label: string = "",
  data: Data[] = []
): Data[] {
  if (isMultiSet(xSet)) {
    xSet.sets.forEach((subset, i) => {
      plotSet(pltFun, subset, color, i === 0 ? label : "", data);
    });
  } else {
    pltFun(xSet, color, label, data);
  }
  return data;
}

function plotSet1d(
  XSet: Set,
  color: string,
  label: string = "",
  data: Data[] = []
): Data[] {
  function plotRect1d(
    s: Set,
    color: string,
    label: string,
    data: Data[]
  ): void {
    let x0 = 0.0,
      x1 = 0.0;
    if (isRectSet(s)) {
      x0 = s.lower_bound[0];
      x1 = s.upper_bound[0];
    } else if (isSphereSet(s)) {
      x0 = s.center[0] - s.radius;
      x1 = s.center[0] + s.radius;
    } else if (isEllipseSet(s)) {
      x0 = s.center[0] - s.radii[0];
      x1 = s.center[0] + s.radii[0];
    }

    data.push({
      type: "scatter",
      x: [x0, x1],
      y: [0, 0],
      mode: "lines",
      line: { color, width: 3 },
      name: label,
      showlegend: !!label,
    });
  }

  return plotSet(plotRect1d, XSet, color, label, data);
}

function plotSet2d(
  XSet: Set,
  color: string,
  label: string = "",
  data: Data[] = []
): Data[] {
  function plotRect2d(
    s: Set,
    color: string,
    label: string,
    data: Data[]
  ): void {
    let x: number[] = [],
      y: number[] = [],
      z: number[] = [];

    if (isRectSet(s)) {
      x = [
        s.lower_bound[0],
        s.upper_bound[0],
        s.upper_bound[0],
        s.lower_bound[0],
        s.lower_bound[0],
      ];
      y = [
        s.lower_bound[1],
        s.lower_bound[1],
        s.upper_bound[1],
        s.upper_bound[1],
        s.lower_bound[1],
      ];
      z = [0, 0, 0, 0, 0];
    } else if (isSphereSet(s)) {
      const theta = Array.from(
        { length: 100 },
        (_, i) => (i * 2 * Math.PI) / 100
      );
      x = theta.map((t) => s.center[0] + s.radius * Math.cos(t));
      y = theta.map((t) => s.center[1] + s.radius * Math.sin(t));
      z = new Array(100).fill(0);
    } else if (isEllipseSet(s)) {
      const theta = Array.from(
        { length: 100 },
        (_, i) => (i * 2 * Math.PI) / 100
      );
      x = theta.map((t) => s.center[0] + s.radii[0] * Math.cos(t));
      y = theta.map((t) => s.center[1] + s.radii[1] * Math.sin(t));
      z = new Array(100).fill(0);
    }

    data.push({
      type: "scatter3d",
      x,
      y,
      z,
      mode: "lines",
      line: { color, width: 5 },
      name: label,
      showlegend: !!label,
    });
  }

  return plotSet(plotRect2d, XSet, color, label, data);
}

function plotSet2dPlane(
  XSet: Set,
  color: string,
  label: string = "",
  shapes: Partial<Layout["shapes"]> = []
): Partial<Layout["shapes"]> {
  function plotRect2d(
    s: Set,
    color: string,
    label: string,
    shapes: Partial<Layout["shapes"]>
  ): void {
    if (isRectSet(s)) {
      shapes.push({
        type: "rect",
        x0: s.lower_bound[0],
        y0: s.lower_bound[1],
        x1: s.upper_bound[0],
        y1: s.upper_bound[1],
        line: { color },
        name: label,
      });
    } else if (isSphereSet(s)) {
      shapes.push({
        type: "circle",
        xref: "x",
        yref: "y",
        x0: s.center[0] - s.radius,
        y0: s.center[1] - s.radius,
        x1: s.center[0] + s.radius,
        y1: s.center[1] + s.radius,
        line: { color },
        name: label,
      });
    } else if (isEllipseSet(s)) {
      shapes.push({
        type: "circle",
        xref: "x",
        yref: "y",
        x0: s.center[0] - s.radii[0],
        y0: s.center[1] - s.radii[1],
        x1: s.center[0] + s.radii[0],
        y1: s.center[1] + s.radii[1],
        line: { color },
        name: label,
      });
    }
  }

  if (isMultiSet(XSet)) {
    XSet.sets.forEach((subset, i) => {
      plotSet2dPlane(subset, color, i === 0 ? label : "", shapes);
    });
  } else {
    plotRect2d(XSet, color, label, shapes);
  }

  return shapes;
}

function plotSet3d(
  XSet: Set,
  color: string,
  label: string = "",
  data: Data[] = []
): Data[] {
  function plotRect3d(
    s: Set,
    color: string,
    label: string,
    data: Data[]
  ): void {
    if (isRectSet(s)) {
      const [x_l, y_l, z_l] = s.lower_bound;
      const [x_u, y_u, z_u] = s.upper_bound;

      const x = [
        x_l,
        x_u,
        x_u,
        x_l,
        x_l,
        null,
        x_l,
        x_u,
        x_u,
        x_l,
        x_l,
        null,
        x_l,
        x_l,
        null,
        x_u,
        x_u,
        null,
        x_l,
        x_l,
        null,
        x_u,
        x_u,
        null,
      ];
      const y = [
        y_l,
        y_l,
        y_u,
        y_u,
        y_l,
        null,
        y_l,
        y_l,
        y_u,
        y_u,
        y_l,
        null,
        y_l,
        y_l,
        null,
        y_l,
        y_l,
        null,
        y_u,
        y_u,
        null,
        y_u,
        y_u,
        null,
      ];
      const z = [
        z_l,
        z_l,
        z_l,
        z_l,
        z_l,
        null,
        z_u,
        z_u,
        z_u,
        z_u,
        z_u,
        null,
        z_l,
        z_u,
        null,
        z_l,
        z_u,
        null,
        z_l,
        z_u,
        null,
        z_l,
        z_u,
        null,
      ];

      data.push({
        type: "scatter3d",
        x: x as number[],
        y: y as number[],
        z: z as number[],
        mode: "lines",
        line: { color },
        name: label,
        showlegend: !!label,
      });
    } else if (isSphereSet(s) || isEllipseSet(s)) {
      const radius = isSphereSet(s) ? [s.radius, s.radius, s.radius] : s.radii;
      const theta = Array.from(
        { length: 120 },
        (_, i) => (i * 2 * Math.PI) / 120
      );
      const phi = Array.from({ length: 60 }, (_, i) => (i * Math.PI) / 60);

      const x: (number | null)[] = [];
      const y: (number | null)[] = [];
      const z: (number | null)[] = [];

      // Meridians
      for (let k = 0; k < 12; k++) {
        const t = theta[10 * k];
        phi.forEach((p) => {
          x.push(s.center[0] + radius[0] * Math.cos(t) * Math.sin(p));
          y.push(s.center[1] + radius[1] * Math.sin(t) * Math.sin(p));
          z.push(s.center[2] + radius[2] * Math.cos(p));
        });
        x.push(null);
        y.push(null);
        z.push(null);
      }

      // Parallels
      for (let k = 0; k < 10; k++) {
        const p = phi[6 * k];
        theta.forEach((t) => {
          x.push(s.center[0] + radius[0] * Math.cos(t) * Math.sin(p));
          y.push(s.center[1] + radius[1] * Math.sin(t) * Math.sin(p));
          z.push(s.center[2] + radius[2] * Math.cos(p));
        });
        x.push(null);
        y.push(null);
        z.push(null);
      }

      data.push({
        type: "scatter3d",
        x: x as number[],
        y: y as number[],
        z: z as number[],
        mode: "lines",
        line: { color },
        name: label,
        showlegend: !!label,
      });
    }
  }

  return plotSet(plotRect3d, XSet, color, label, data);
}

// Main plotting functions
function plotSolution1d(
  solutionData: PlotSolutionData,
  XBounds: RectSet,
  options: {
    XInit?: Set | null;
    XUnsafe?: Set | null;
    eta?: number | null;
    gamma?: number | null;
    c?: number;
  } = {}
): PlotParams {
  const {
    XInit = null,
    XUnsafe = null,
    eta = null,
    gamma = null,
    c = 0.0,
  } = options;

  const data: Data[] = [];
  const shapes: Partial<Shape>[] = [];
  const X_lattice = solutionData.x_lattice.map((row: number[]) => row[0]);

  // Draw sets
  if (XUnsafe) plotSet1d(XUnsafe, "red", "unsafe set", data);
  if (XInit) plotSet1d(XInit, "blue", "initial set", data);

  // Add eta and gamma lines
  if (eta !== null) {
    shapes.push({
      type: "line",
      x0: XBounds.lower_bound[0],
      y0: eta,
      x1: XBounds.upper_bound[0],
      y1: eta,
      line: { color: "green", dash: "dot" },
    });
  }
  if (gamma !== null) {
    shapes.push({
      type: "line",
      x0: XBounds.lower_bound[0],
      y0: gamma,
      x1: XBounds.upper_bound[0],
      y1: gamma,
      line: { color: "red", dash: "dot" },
    });
  }

  if (solutionData.B_lattice) {
    data.push({
      type: "scatter",
      x: X_lattice,
      y: solutionData.B_lattice,
      mode: "lines",
      line: { color: "green" },
      name: "B(x)",
    });

    data.push({
      type: "scatter",
      x: X_lattice,
      y: solutionData.B_lattice,
      mode: "markers",
      marker: { color: "green" },
      showlegend: false,
    });

    // Fill area
    data.push({
      type: "scatter",
      x: X_lattice,
      y: solutionData.B_lattice.map((v) => v + c + 1e-8),
      fill: "tonexty",
      fillcolor: "rgba(144, 238, 144, 0.3)",
      line: { color: "rgba(255,255,255,0)" },
      showlegend: false,
      name: "Barrier region",
    });
  }

  if (solutionData.Bp_lattice) {
    data.push({
      type: "scatter",
      x: solutionData.x_lattice.map((row) => row[0]),
      y: solutionData.Bp_lattice,
      mode: "lines",
      line: { color: "black" },
      name: "B(xp)",
    });
    data.push({
      type: "scatter",
      x: solutionData.x_lattice.map((row) => row[0]),
      y: solutionData.Bp_lattice,
      mode: "markers",
      marker: { color: "black" },
      showlegend: false,
    });
  }

  if (solutionData.Bp_lattice_est) {
    data.push({
      type: "scatter",
      x: solutionData.x_lattice.map((row) => row[0]),
      y: solutionData.Bp_lattice_est,
      mode: "lines",
      line: { color: "purple" },
      name: "B(xp) est.",
    });
    data.push({
      type: "scatter",
      x: solutionData.x_lattice.map((row) => row[0]),
      y: solutionData.Bp_lattice_est,
      mode: "markers",
      marker: { color: "purple" },
      showlegend: false,
    });
  }

  return {
    data,
    layout: {
      title: { text: "Barrier certificate" },
      xaxis: {
        title: { text: "State space" },
        range: [XBounds.lower_bound[0], XBounds.upper_bound[0]],
      },
      shapes,
      showlegend: true,
    },
  };
}

function plotSolution2d(
  solutionData: PlotSolutionData,
  XBounds: RectSet,
  options: {
    XInit?: Set | null;
    XUnsafe?: Set | null;
    eta?: number | null;
    gamma?: number | null;
  } = {}
): PlotParams {
  const { XInit = null, XUnsafe = null, eta = null, gamma = null } = options;

  const data: Data[] = [];

  // Draw sets
  if (XInit) plotSet2d(XInit, "blue", "initial set", data);
  if (XUnsafe) plotSet2d(XUnsafe, "red", "unsafe set", data);

  const X_lattice: number[][] = [];
  const Y_lattice: number[][] = [];
  const B_lattice: number[][] = [];
  const Bp_lattice: number[][] = [];
  const Bp_lattice_est: number[][] = [];
  const n = Math.round(Math.sqrt(solutionData.x_lattice.length));
  for (let i = 0; i < n - 1; i++) {
    X_lattice.push(
      solutionData.x_lattice.slice(i * n, i * n + n).map((row) => row[0])
    );
    Y_lattice.push(
      solutionData.x_lattice.slice(i * n, i * n + n).map((row) => row[1])
    );
    B_lattice.push(solutionData.B_lattice!.slice(i * n, i * n + n));
    if (solutionData.Bp_lattice) {
      Bp_lattice.push(solutionData.Bp_lattice!.slice(i * n, i * n + n));
    }
    if (solutionData.Bp_lattice_est) {
      Bp_lattice_est.push(solutionData.Bp_lattice_est!.slice(i * n, i * n + n));
    }
  }

  if (solutionData.B_lattice) {
    data.push({
      type: "surface",
      x: X_lattice,
      y: Y_lattice,
      z: B_lattice,
      colorscale: "Viridis",
      opacity: 0.7,
      name: "B(x)",
      showscale: false,
      showlegend: true,
      contours: {
        /* @ts-ignore */
        z: {
          show: true,
          start: eta || 0,
          end: (gamma || 0) + 0.1,
          size: (gamma || 0) - (eta || 0),
          project: { z: true },
          highlight: false,
          usecolormap: true,
        },
      },
    });
  }

  if (eta !== null) {
    data.push({
      type: "surface",
      x: [
        [XBounds.lower_bound[0], XBounds.upper_bound[0]],
        [XBounds.lower_bound[0], XBounds.upper_bound[0]],
      ],
      y: [
        [XBounds.lower_bound[1], XBounds.lower_bound[1]],
        [XBounds.upper_bound[1], XBounds.upper_bound[1]],
      ],
      z: [Array(2).fill(eta), Array(2).fill(eta)],
      colorscale: [
        [0, "green"],
        [1, "green"],
      ],
      opacity: 0.2,
      name: "eta",
      showscale: false,
      showlegend: true,
    });
  }

  if (gamma !== null) {
    data.push({
      type: "surface",
      x: [
        [XBounds.lower_bound[0], XBounds.upper_bound[0]],
        [XBounds.lower_bound[0], XBounds.upper_bound[0]],
      ],
      y: [
        [XBounds.lower_bound[1], XBounds.lower_bound[1]],
        [XBounds.upper_bound[1], XBounds.upper_bound[1]],
      ],
      z: [Array(2).fill(gamma), Array(2).fill(gamma)],
      colorscale: [
        [0, "red"],
        [1, "red"],
      ],
      opacity: 0.2,
      name: "gamma",
      showscale: false,
      showlegend: true,
    });
  }

  if (Bp_lattice) {
    data.push({
      type: "surface",
      x: X_lattice,
      y: Y_lattice,
      z: Bp_lattice,
      colorscale: [
        [0, "black"],
        [1, "black"],
      ],
      opacity: 0.3,
      name: "B(xp)",
      showscale: false,
      showlegend: true,
    });
  }

  if (Bp_lattice_est) {
    data.push({
      type: "surface",
      x: X_lattice,
      y: Y_lattice,
      z: Bp_lattice_est,
      colorscale: [
        [0, "purple"],
        [1, "purple"],
      ],
      opacity: 0.3,
      name: "B(xp) est.",
      showscale: false,
      showlegend: true,
    });
  }

  return {
    data,
    layout: {
      title: { text: "Barrier certificate" },
      scene: {
        xaxis: {
          title: { text: "State space x[0]" },
          range: [XBounds.lower_bound[0], XBounds.upper_bound[0]],
        },
        yaxis: {
          title: { text: "State space x[1]" },
          range: [XBounds.lower_bound[1], XBounds.upper_bound[1]],
        },
        zaxis: { title: { text: "Barrier value" }, range: [0, undefined] },
      },
    },
  };
}

function plotData1d(
  previewData: PlotPreviewData,
  _: RectSet,
  options: {
    XInit?: Set | null;
    XUnsafe?: Set | null;
  } = {}
): PlotParams {
  const { XInit = null, XUnsafe = null } = options;
  const X_lattice = previewData.x_lattice.map((row: number[]) => row[0]);
  const Xp_lattice = previewData.xp_lattice.map((row: number[]) => row[0]);
  const ySub = previewData.x_lattice.map(
    (_, i) => i / (previewData.x_lattice.length - 1)
  );

  const data: Data[] = [];
  const annotations: Partial<Annotations>[] = [];

  if (XInit) plotSet1d(XInit, "blue", "Initial Set", data);
  if (XUnsafe) plotSet1d(XUnsafe, "red", "Unsafe Set", data);

  // Add current state points
  data.push({
    type: "scatter",
    x: X_lattice,
    y: ySub,
    mode: "markers",
    marker: { color: "blue", size: 0.1 },
    name: "Current state points",
    showlegend: false,
  });

  // Add next state points
  data.push({
    type: "scatter",
    x: Xp_lattice,
    y: ySub,
    mode: "markers",
    marker: { color: "orange", size: 0.1 },
    name: "Next state points",
    showlegend: false,
  });

  // Add arrows to show vector field
  for (let i = 0; i < X_lattice.length; i++) {
    annotations.push({
      x: Xp_lattice[i],
      y: ySub[i],
      ax: X_lattice[i],
      ay: ySub[i],
      xref: "x",
      yref: "y",
      axref: "x",
      ayref: "y",
      showarrow: true,
      arrowhead: 2,
      arrowsize: 1,
      arrowwidth: 1,
      arrowcolor: "blue",
    });
  }

  return {
    data,
    layout: {
      title: { text: "Data Plot" },
      xaxis: { title: { text: "Input" } },
      yaxis: { title: { text: "Output" } },
      annotations,
    },
  };
}

function plotData2d(
  previewData: PlotPreviewData,
  XBounds: RectSet,
  options: {
    XInit?: Set | null;
    XUnsafe?: Set | null;
  } = {}
): PlotParams {
  const { XInit = null, XUnsafe = null } = options;
  const data: Data[] = [];
  const shapes: Partial<Shape>[] = [];
  const xSamples = previewData.x_lattice;
  const xpSamples = previewData.xp_lattice;

  if (XInit) plotSet2dPlane(XInit, "blue", "Initial Set", shapes);
  if (XUnsafe) plotSet2dPlane(XUnsafe, "red", "Unsafe Set", shapes);

  // Create arrow traces
  const separator = new Array(xSamples.length).fill(NaN);
  const x = xSamples
    .map((row, i) => [row[0], xpSamples[i][0], separator[i]])
    .flat();
  const y = xSamples
    .map((row, i) => [row[1], xpSamples[i][1], separator[i]])
    .flat();

  data.push({
    type: "scatter",
    x,
    y,
    mode: "lines+markers",
    line: { color: "blue", width: 0.5 },
    marker: {
      symbol: "arrow",
      color: "blue",
      size: 10,
      /* @ts-ignore */
      angleref: "previous",
    },
    name: "Samples",
    showlegend: false,
  });

  return {
    data,
    layout: {
      title: { text: "Data Plot" },
      xaxis: {
        title: { text: "Input Dimension 1" },
        range: XBounds
          ? [XBounds.lower_bound[0], XBounds.upper_bound[0]]
          : undefined,
      },
      yaxis: {
        title: { text: "Input Dimension 2" },
        range: XBounds
          ? [XBounds.lower_bound[1], XBounds.upper_bound[1]]
          : undefined,
      },
      shapes,
    },
  };
}

function plotData3d(
  previewData: PlotPreviewData,
  _: RectSet,
  options: {
    XInit?: Set | null;
    XUnsafe?: Set | null;
  } = {}
): PlotParams {
  const { XInit = null, XUnsafe = null } = options;
  const xSamples = previewData.x_lattice;
  const xpSamples = previewData.xp_lattice;

  const data: Data[] = [];

  if (XInit) plotSet3d(XInit, "blue", "Initial Set", data);
  if (XUnsafe) plotSet3d(XUnsafe, "red", "Unsafe Set", data);

  // Create line segments
  const separator = new Array(xSamples.length).fill(NaN);
  const x = xSamples
    .map((row, i) => [row[0], xpSamples[i][0], separator[i]])
    .flat();
  const y = xSamples
    .map((row, i) => [row[1], xpSamples[i][1], separator[i]])
    .flat();
  const z = xSamples
    .map((row, i) => [row[2], xpSamples[i][2], separator[i]])
    .flat();

  data.push({
    type: "scatter3d",
    x,
    y,
    z,
    mode: "lines",
    line: { color: "blue", width: 2 },
    name: "Current state points",
    showlegend: false,
  });

  // Add cones for arrows
  const arrowTipRatio = 0.2;
  const arrowStartingRatio = 0.98;

  for (let i = 0; i < xSamples.length; i++) {
    const dx = xpSamples[i][0] - xSamples[i][0];
    const dy = xpSamples[i][1] - xSamples[i][1];
    const dz = xpSamples[i][2] - xSamples[i][2];

    data.push({
      type: "cone",
      x: [xSamples[i][0] + arrowStartingRatio * dx],
      y: [xSamples[i][1] + arrowStartingRatio * dy],
      z: [xSamples[i][2] + arrowStartingRatio * dz],
      /* @ts-ignore */
      u: [arrowTipRatio * dx],
      v: [arrowTipRatio * dy],
      w: [arrowTipRatio * dz],
      showlegend: false,
      showscale: false,
      colorscale: [
        [0, "blue"],
        [1, "blue"],
      ],
    });
  }

  return {
    data,
    layout: {
      title: { text: "Data Plot" },
      scene: {
        xaxis: { title: { text: "Input Dimension 1" } },
        yaxis: { title: { text: "Input Dimension 2" } },
        zaxis: { title: { text: "Input Dimension 3" } },
      },
    },
  };
}

export function plotSolution(
  solutionData: PlotSolutionData,
  XBounds: RectSet | null,
  options: {
    XInit?: Set | null;
    XUnsafe?: Set | null;
    eta?: number | null;
    gamma?: number | null;
    c?: number;
  } = {}
): PlotParams {
  const plotSolutionFunctions = [plotSolution1d, plotSolution2d];

  if (XBounds && XBounds.dimension <= plotSolutionFunctions.length) {
    return plotSolutionFunctions[XBounds.dimension - 1](
      solutionData,
      XBounds,
      options
    );
  }

  throw new Error(
    `Plotting is not supported for ${XBounds?.dimension ?? "?"
    }-dimensional sets. Only 1D and 2D are supported.`
  );
}

function plotFunction1d(
  previewData: PlotPreviewData,
  XBounds: RectSet,
  options: {
    XInit?: Set | null;
    XUnsafe?: Set | null;
  } = {}
): PlotParams {
  return plotData1d(previewData, XBounds, options);
}

function plotFunction2d(
  previewData: PlotPreviewData,
  XBounds: RectSet,
  options: {
    XInit?: Set | null;
    XUnsafe?: Set | null;
  } = {}
): PlotParams {
  return plotData2d(previewData, XBounds, options);
}

export function plotFunction(
  previewData: PlotPreviewData,
  XBounds: RectSet,
  options: {
    XInit?: Set | null;
    XUnsafe?: Set | null;
  } = {}
): PlotParams {
  const plotFunctionFunctions = [plotFunction1d, plotFunction2d];

  if (XBounds.dimension <= plotFunctionFunctions.length) {
    return plotFunctionFunctions[XBounds.dimension - 1](
      previewData,
      XBounds,
      options
    );
  }

  throw new Error(
    `Plotting is not supported for ${XBounds.dimension}-dimensional sets. Only 1D and 2D are supported.`
  );
}

export function plotData(
  previewData: PlotPreviewData,
  XBounds: RectSet,
  options: {
    XInit?: Set | null;
    XUnsafe?: Set | null;
  } = {}
): PlotParams {
  const { XInit = null, XUnsafe = null } = options;
  validateInputs(
    previewData.x_lattice,
    previewData.xp_lattice,
    XBounds,
    XInit,
    XUnsafe
  );

  const dimension = XBounds.dimension;
  if (dimension === 1) {
    return plotData1d(previewData, XBounds!, { XInit, XUnsafe });
  } else if (dimension === 2) {
    return plotData2d(previewData, XBounds, { XInit, XUnsafe });
  } else if (dimension === 3) {
    return plotData3d(previewData, XBounds, { XInit, XUnsafe });
  }

  throw new Error(
    `Plotting is not supported for ${dimension}-dimensional sets. Only 1D, 2D and 3D are supported.`
  );
}
