import {
  useFormContext,
  useFieldArray,
  type FieldErrors,
  type FieldValues,
} from "react-hook-form";
import SetInput from "@components/SetInput";
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
  const [graph, setGraph] = useState<string>("");
  const { fields, append, remove } = useFieldArray({
    control,
    name: "system_dynamics",
  });

  console.log("ConfigSystem: useFormContext", formState.errors);

  const handlePreview = useCallback(async () => {
    if (
      !(await trigger(["system_dynamics", "X_bounds", "X_init", "X_unsafe"]))
    ) {
      console.error("Form validation failed");
      return;
    }
    const response = await fetch("http://127.0.0.1:5000/preview-graph", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(getValues()),
    });
    if (!response.ok) {
      setGraph("");
      const error: ErrorMessage = await response.json();
      if (error.cause) {
        setError(error.cause, {
          type: "value",
          message: error.message,
        });
      }
      throw new Error(`Error fetching graph preview: ${error.message}`);
    }
    setGraph(await response.text());
  }, [getValues, setGraph, trigger, setError]);

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="font-bold text-lg mb-2">System Configuration</h2>
        <button
          type="button"
          className="bg-blue-500 px-4 py-2 rounded flex items-center text-white"
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
                className="bg-red-500 size-8 p-2 rounded"
              >
                <FaTrash />
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
          className="bg-green-500 px-4 py-2 rounded flex items-center"
        >
          <FaPlus className="inline-block mr-1 size-4" />
          Add output dimension
        </button>
      </div>

      <SetInput name="X_bounds" label="X bounds" />

      <SetInput name="X_init" label="X init" />

      <SetInput name="X_unsafe" label="X unsafe" />

      <DangerousElement markup={graph} />
    </div>
  );
}
