import {
  useFormContext,
  type FieldErrors,
  type FieldValues,
} from "react-hook-form";
import SetsInput from "@app/components/SetsInput";
import { FaEye, FaSpinner } from "react-icons/fa6";
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
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

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
    setLoading(true);
    const response = await fetch("/api/preview-graph", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(getValues()),
    });
    setLoading(false);
    if (!response.ok) {
      setFig("");
      const error: ServerResponse = await response.json();
      if (error.cause) {
        setError(error.cause, {
          type: "value",
          message: error.error,
        });
      }
      setSubmitError(error.error ?? "Unknown error");
    }
    const json = await response.json();
    setFig(json.fig || "");
  }, [getValues, setFig, trigger, setError, setSubmitError, setLoading]);

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="font-bold text-lg mb-2">System Configuration</h2>
        <div className="flex flex-row-reverse">
          <button
            type="button"
            className="btn btn-primary flex items-center justify-center w-36"
            disabled={loading}
            onClick={handlePreview}
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
          </button>
          <div className="flex items-center mr-2">
            {submitError && (
              <small className="text-red-500">{submitError}</small>
            )}
          </div>
        </div>
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
