import { ErrorMessage } from "@hookform/error-message";
import { useFieldArray, useFormContext } from "react-hook-form";
import { FaPlus } from "react-icons/fa6";
import SetInput from "./SetInput";
import { Button } from "./ui/button";
import { defaultSetWithDimension } from "@/utils/utils";

type SetsInputProps = {
  name: string;
  label: string;
};

export default function SetsInput({ name, label }: SetsInputProps) {
  const { control, formState, getValues } = useFormContext();
  const { fields, append, remove } = useFieldArray({
    control,
    name: `${name}`,
  });

  // Very basic structure display - you'd want a proper editor here
  return (
    <div>
      <label className="block font-bold">{label}</label>
      <div className="flex flex-col gap-1">
        {fields.map((field, index) => (
          <div key={field.id}>
            <SetInput
              name={name}
              idx={index}
              type="RectSet"
              label={`RectSet ${index + 1}`}
              removeItself={fields.length > 1 ? () => remove(index) : undefined}
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
      {name !== "X_bounds" && (
        <Button
          onClick={() =>
            append({
              RectSet: defaultSetWithDimension(
                "RectSet",
                getValues("dimension")
              ),
            })
          }
          variant={"default"}
        >
          <FaPlus className="inline-block size-4" />
        </Button>
      )}
      <ErrorMessage
        errors={formState.errors}
        name={`${name}.root`}
        render={({ message }) => (
          <small className="text-red-500 block mb-1">{message}</small>
        )}
      />
    </div>
  );
}
