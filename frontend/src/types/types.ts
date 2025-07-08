import type { FieldErrors, FieldValues } from "react-hook-form";

export type FormStep = {
  name: string;
  current: boolean;
  error: (errors: FieldErrors<FieldValues>) => boolean;
  href: string;
};
export type FormSteps = {
  system: FormStep;
  algorithm: FormStep;
  execution: FormStep;
};
export type FormStepName = "system" | "algorithm" | "execution";

export type EstimatorType = "KernelRidgeRegressor";
export type KernelType = "GaussianKernel";
export type FeatureMapType =
  | "LinearTruncatedFourierFeatureMap"
  | "ConstantTruncatedFourierFeatureMap"
  | "LogTruncatedFourierFeatureMap";
export type OptimiserType = "GurobiOptimiser" | "AlglibOptimiser" | "HighsOptimiser";
export type ServerResponse = {
  success: boolean;
  obj_val?: number;
  sol?: number[];
  eta?: number;
  c?: number;
  norm?: number;
  fig?: string;
  error?: string;
  log?: string;
  cause?: string;
  uuid?: string;
  verified?: boolean;
};

export type LogEntry = {
  text: string;
  type: "trace" | "debug" | "info" | "warning" | "error" | "critical";
  timestamp: string;
};

export type RectSet = {
  RectSet: [number, number][];
};
