import {
  useFormContext,
  type FieldErrors,
  type FieldValues,
} from "react-hook-form";
import SetsInput from "@app/components/SetsInput";
import { FaEye } from "react-icons/fa6";
import { useCallback, useState } from "react";
import { DangerousElement } from "@components/DangerousElement";
import SystemDynamicsInput from "./SystemDynamicsInput";
import type { ServerResponse } from "@app/types/types";
import SamplesInput from "./SamplesInput";

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

export default function ConfigSystem() {
  const { getValues, trigger, setError } = useFormContext();
  const [fig, setFig] = useState<string>("");

  const handlePreview = useCallback(async () => {
    if (
      !(await trigger([
        "system_dynamics",
        "X_bounds",
        "X_init",
        "X_unsafe",
        "x_samples",
        "xp_samples",
      ]))
    ) {
      console.error("Form validation failed");
      return;
    }
    const response = await fetch("/api/preview-graph", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(getValues()),
    });
    if (!response.ok) {
      setFig("");
      const error: ServerResponse = await response.json();
      if (error.cause) {
        setError(error.cause, {
          type: "value",
          message: error.error,
        });
      }
      throw new Error(
        `Error fetching graph preview: ${error.error ?? "Unknown error"}`
      );
    }
    const json = await response.json();
    setFig(json.fig || "");
  }, [getValues, setFig, trigger, setError]);

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="font-bold text-lg mb-2">System Configuration</h2>
        <button
          type="button"
          className="btn btn-primary"
          onClick={handlePreview}
        >
          <FaEye className="inline-block mr-1 size-4" />
          Preview
        </button>
      </div>
      <SamplesInput name="x_samples" label="Samples" />
      <SamplesInput name="xp_samples" label="Transition samples" />

      <SystemDynamicsInput />

      <SetsInput name="X_bounds" label="X bounds" />

      <SetsInput name="X_init" label="X init" />

      <SetsInput name="X_unsafe" label="X unsafe" />

      <DangerousElement markup={fig} />
    </div>
  );
}
