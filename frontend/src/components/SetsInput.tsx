import { ErrorMessage } from "@hookform/error-message";
import { useFieldArray, useFormContext } from "react-hook-form";
import { FaPlus } from "react-icons/fa6";
import SetInput from "./SetInput";

type SetsInputProps = {
  name: string;
  label: string;
};

export default function SetsInput({ name, label }: SetsInputProps) {
  const { control, formState } = useFormContext();
  const { fields, append, remove } = useFieldArray({
    control,
    name: `${name}`,
  });

  // Very basic structure display - you'd want a proper editor here
  return (
    <div className="form-group">
      <label className="block font-bold">{label}</label>
      <div className="flex flex-row gap-2">
        {fields.map((field, index) => (
          <div className="my-2" key={field.id}>
            <SetInput
              name={name}
              idx={index}
              type="RectSet"
              label={`RectSet ${index + 1}`}
              removeItself={remove}
            />
            <ErrorMessage
              errors={formState.errors}
              name={`${name}.${index}`}
              render={({ message }) => (
                <small className="text-red-500">{message}</small>
              )}
            />
          </div>
        ))}
      </div>
      <ErrorMessage
        errors={formState.errors}
        name={name}
        render={({ message }) => (
          <small className="text-red-500 block mb-1">{message}</small>
        )}
      />
      <button
        type="button"
        onClick={() => append({ RectSet: [[0, 1]] })}
        className="btn btn-success flex items-center"
      >
        <FaPlus className="inline-block mr-1 size-4" />
        Add set
      </button>
    </div>
  );
}
