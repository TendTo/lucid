import {
  useFormContext,
  useFieldArray,
  type FieldErrors,
  type FieldValues,
} from "react-hook-form";
import SetsInput from "@app/components/SetsInput";
import { FaEye, FaPlus, FaTrash } from "react-icons/fa6";
import { ErrorMessage } from "@hookform/error-message";
import { useCallback, useState } from "react";
import { DangerousElement } from "@components/DangerousElement";

export function systemFormErrors(errors: FieldErrors<FieldValues>): boolean {
  return Boolean(
    errors.system_dynamics ||
      errors.X_bounds ||
      errors.X_init ||
      errors.X_unsafe
  );
}

type ErrorMessage = {
  message: string;
  cause: string;
};

export default function ConfigSystem() {
  const { register, control, formState, getValues, trigger, setError } =
    useFormContext();
  const [fig, setFig] = useState<string>("");
  const { fields, append, remove } = useFieldArray({
    control,
    name: "system_dynamics",
  });

  const handlePreview = useCallback(async () => {
    if (
      !(await trigger(["system_dynamics", "X_bounds", "X_init", "X_unsafe"]))
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
      const error: ErrorMessage = await response.json();
      if (error.cause) {
        setError(error.cause, {
          type: "value",
          message: error.message,
        });
      }
      throw new Error(`Error fetching graph preview: ${error.message}`);
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
          disabled={systemFormErrors(formState.errors)}
          onClick={handlePreview}
        >
          <FaEye className="inline-block mr-1 size-4" />
          Preview
        </button>
      </div>
      <div className="form-group">
        <label className="block font-bold">System Dynamics</label>
        {fields.map((field, index) => (
          <div className="my-2" key={field.id}>
            <div className="flex items-center gap-2">
              <input
                className="flex-grow-1 border rounded px-3 py-2 border-solid border-[#ddd]"
                {...register(`system_dynamics.${index}`)}
                placeholder="x1 + sin(x2)"
              />
              <button
                type="button"
                onClick={() => remove(index)}
                className="btn btn-danger size-8"
              >
                <FaTrash className="flex-grow-1" />
              </button>
            </div>
            <ErrorMessage
              errors={formState.errors}
              name={`system_dynamics.${index}`}
              render={({ message }) => (
                <small className="text-red-500">{message}</small>
              )}
            />
          </div>
        ))}
        <ErrorMessage
          errors={formState.errors}
          name="system_dynamics"
          render={({ message }) => (
            <small className="text-red-500 block mb-1">{message}</small>
          )}
        />
        <button
          type="button"
          onClick={() => append(`x${fields.length + 1}`)}
          className="btn btn-success"
        >
          <FaPlus className="inline-block mr-1 size-4" />
          Add output dimension
        </button>
      </div>

      <SetsInput name="X_bounds" label="X bounds" />

      <SetsInput name="X_init" label="X init" />

      <SetsInput name="X_unsafe" label="X unsafe" />

      <DangerousElement markup={fig} />
    </div>
  );
}
