import SamplesInput from "@/components/SamplesInput";
import SetsInput from "@/components/SetsInput";
import SystemDynamicsInput from "@/components/SystemDynamicsInput";
import type { ServerResponse } from "@/types/types";
import { emptyFigure } from "@/utils/constants";
import { useCallback, useState } from "react";
import {
  useFormContext,
  type FieldErrors,
  type FieldValues,
} from "react-hook-form";
import { FaEye, FaSpinner } from "react-icons/fa6";
import type { PlotParams } from "react-plotly.js";
import TabGroup from "./TabGroup";
import { Button } from "./ui/button";

export function systemFormErrors(errors: FieldErrors<FieldValues>): boolean {
  return Boolean(
    errors.system_dynamics ||
      errors.X_bounds ||
      errors.X_init ||
      errors.X_unsafe ||
      errors.x_samples ||
      errors.xp_samples
  );
}

type ConfigSystemProps = {
  onSubmit: (data: FieldValues) => Promise<void>;
  loading: boolean;
  error: string | null;
};

export default function ConfigSystem({
  onSubmit,
  loading,
  error,
}: ConfigSystemProps) {
  const { setValue, clearErrors, handleSubmit } = useFormContext();

  return (
    <>
      <div className="flex flex-row justify-between">
        <h2 className="text-lg font-semibold">System dynamics</h2>
        <div className="flex flex-col items-end">
          <Button
            type="button"
            disabled={loading}
            onClick={() => handleSubmit(onSubmit)()}
          >
            {loading ? (
              <>
                <FaSpinner className="mr-2 animate-spin" />
                Computing
              </>
            ) : (
              <>
                <FaEye className="inline-block mr-1 size-4" />
                Preview
              </>
            )}
          </Button>
          <small className="text-red-500 mt-1">{error}</small>
        </div>
      </div>
      <TabGroup
        className="my-4"
        tabs={{
          Data: {
            content: (
              <div className="flex flex-col gap-4">
                <h3 className="font-semibold">Dataset</h3>
                <SamplesInput name="x_samples" label="Initial states" />
                <SamplesInput name="xp_samples" label="Consecutive states" />
              </div>
            ),
            onClick: () => {
              setValue("x_samples", []);
              setValue("xp_samples", []);
              clearErrors(["x_samples", "xp_samples"]);
            },
          },
          "Closed-form expression": {
            content: <SystemDynamicsInput />,
            onClick: () => {
              setValue("system_dynamics", []);
              clearErrors("system_dynamics");
            },
          },
        }}
      />
    </>
  );
}
