import { ErrorMessage } from "@hookform/error-message";
import { useFieldArray, useFormContext } from "react-hook-form";
import { FaPlus, FaTrash } from "react-icons/fa6";

export default function SystemDynamicsInput() {
  const { register, control, formState } = useFormContext();
  const { fields, append, remove } = useFieldArray({
    control,
    name: "system_dynamics",
  });

  return (
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
  );
}
