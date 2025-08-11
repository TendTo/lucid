import type { ServerCapabilities } from "@/types/types";
import type { Configuration } from "./schema";
import type { availableSets } from "./constants";

export function onChangeNumber(
  onChange: (value: number | "") => void
): (e: React.ChangeEvent<HTMLInputElement>) => void {
  return (e) => onChange(e.target.value !== "" ? Number(e.target.value) : "");
}

export function handleDragOver(e: React.DragEvent<HTMLElement>) {
  e.stopPropagation();
  e.preventDefault();
  e.currentTarget.classList.add("drag-over");
}

export function handleDragLeave(e: React.DragEvent<HTMLElement>) {
  e.stopPropagation();
  e.preventDefault();
  e.currentTarget.classList.remove("drag-over");
}

export function parseNumberListOrString(value: string) {
  if (value.length < 1) return "";
  if (/[^\d.]/.test(value.at(-1)!)) return value;
  return Array.from(value.matchAll(/-?\d+(\.\d+)?/g)).map((match) =>
    Number(match[0])
  );
}

export function parseNumberList(value: string) {
  return Array.from(value.matchAll(/-?\d+(\.\d+)?/g)).map((match) =>
    Number(match[0])
  );
}

export function formatNumber(value: number | undefined) {
  if (value === undefined) return "N/A";
  return typeof value === "number" ? value.toFixed(6) : value;
}

export function capableConfiguration<T extends Partial<Configuration>>(
  config: T,
  capabilities: ServerCapabilities
): T {
  return {
    ...config,
    optimiser: capabilities.GUROBI
      ? "GurobiOptimiser"
      : capabilities.ALGLIB
      ? "AlglibOptimiser"
      : "HighsOptimiser",
    plot: capabilities.PLOT && config.plot,
    verify: capabilities.VERIFICATION && config.verify,
  };
}

export function defaultSetWithDimension(
  set: keyof typeof availableSets,
  dimension: number
) {
  if (set === "RectSet") {
    return Array.from({ length: dimension }, () => [0, 1]);
  } else if (set === "SphereSet") {
    return {
      center: Array.from({ length: dimension }).fill(1.0),
      radius: 1,
    };
  } else {
    throw new Error(`Unknown set type: ${set}`);
  }
}
