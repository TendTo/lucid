import { ErrorMessage } from "@hookform/error-message";
import { useFieldArray, useFormContext } from "react-hook-form";
import { FaPlus, FaTrash } from "react-icons/fa6";

type SetInputProps = {
  name: string;
  label: string;
};

export default function SetInput({ name, label }: SetInputProps) {
  const { register, control, formState } = useFormContext();
  const { fields, append, remove } = useFieldArray({
    control,
    name: `${name}.RectSet`,
  });

  // Very basic structure display - you'd want a proper editor here
  return (
    <div className="form-group">
      <label className="block font-bold">{label}</label>
      {fields.map((field, index) => (
        <div className="my-2" key={field.id}>
          <div className="flex items-center gap-2">
            <input
              className="flex-grow-1 border rounded px-3 py-2 border-solid border-[#ddd] min-w-16"
              {...register(`${name}.RectSet.${index}.0`, {
                valueAsNumber: true,
                required: true,
              })}
              step="any"
              placeholder="0.0"
              type="number"
            />
            <input
              className="flex-grow-1 border rounded px-3 py-2 border-solid border-[#ddd] min-w-16"
              {...register(`${name}.RectSet.${index}.1`, {
                valueAsNumber: true,
                required: true,
              })}
              placeholder="0.0"
              step="any"
              type="number"
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
            name={`${name}.RectSet.${index}`}
            render={({ message }) => (
              <small className="text-red-500">{message}</small>
            )}
          />
          <ErrorMessage
            errors={formState.errors}
            name={`${name}.RectSet.${index}.root`}
            render={({ message }) => (
              <small className="text-red-500">{message}</small>
            )}
          />
        </div>
      ))}
      <ErrorMessage
        errors={formState.errors}
        name={`${name}.RectSet`}
        render={({ message }) => (
          <small className="text-red-500 block mb-1">{message}</small>
        )}
      />
      <button
        type="button"
        onClick={() => append([[0, 1]])}
        className="bg-green-500 px-4 py-2 rounded flex items-center"
      >
        <FaPlus className="inline-block mr-1 size-4" />
        Add dimension
      </button>
    </div>
  );
}
