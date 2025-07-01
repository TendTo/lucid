import { ErrorMessage } from "@hookform/error-message";
import { useFieldArray, useFormContext } from "react-hook-form";
import { FaPlus, FaTrash } from "react-icons/fa6";

type SetInputProps = {
  name: string;
  idx: number;
  label: string;
  type: "RectSet";
  removeItself: (index: number) => void;
};

export default function SetInput({
  name,
  idx,
  type,
  label,
  removeItself,
}: SetInputProps) {
  const { register, control, formState, getValues } = useFormContext();
  const { fields, append, remove } = useFieldArray({
    control,
    name: `${name}.${idx}.${type}`,
  });

  return (
    <div className="form-group">
      <label className="block font-bold">{label}</label>
      {fields.map((field, index) => (
        <div className="my-2" key={field.id}>
          <div className="flex items-center gap-2">
            <input
              className="border rounded px-3 py-2 border-solid border-[#ddd] w-24"
              {...register(`${name}.${idx}.${type}.${index}.0`, {
                valueAsNumber: true,
                required: true,
              })}
              max={getValues(`${name}.${idx}.${type}.${index}.1`)}
              step="any"
              placeholder="0.0"
              type="number"
            />
            <input
              className="border rounded px-3 py-2 border-solid border-[#ddd] w-24"
              {...register(`${name}.${idx}.${type}.${index}.1`, {
                valueAsNumber: true,
                required: true,
              })}
              min={getValues(`${name}.${idx}.${type}.${index}.0`)}
              placeholder="0.0"
              step="any"
              type="number"
            />
            <button
              type="button"
              onClick={() => remove(index)}
              className="btn btn-danger"
            >
              <FaTrash />
            </button>
          </div>
          <ErrorMessage
            errors={formState.errors}
            name={`${name}.${idx}.${type}.${index}`}
            render={({ message }) => (
              <small className="text-red-500">{message}</small>
            )}
          />
          <ErrorMessage
            errors={formState.errors}
            name={`${name}.${idx}.${type}.${index}.root`}
            render={({ message }) => (
              <small className="text-red-500">{message}</small>
            )}
          />
        </div>
      ))}
      <ErrorMessage
        errors={formState.errors}
        name={`${name}.${idx}.${type}`}
        render={({ message }) => (
          <small className="text-red-500 block mb-1">{message}</small>
        )}
      />
      <div className="flex items-center gap-2 mt-2">
        <button
          type="button"
          onClick={() => append([[0, 1]])}
          className="btn btn-success flex items-center"
        >
          <FaPlus className="inline-block mr-1 size-4" />
          Add dimension
        </button>
        <button
          type="button"
          onClick={() => removeItself(idx)}
          className="btn btn-danger"
        >
          <FaTrash />
        </button>
      </div>
    </div>
  );
}
