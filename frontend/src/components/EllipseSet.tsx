import { parseNumberList } from "@/utils/utils";
import { Input } from "./ui/input";
import { ErrorMessage } from "@hookform/error-message";
import { useFieldArray, useFormContext } from "react-hook-form";

type EllipseSetProps = {
  name: string;
  idx: number;
};

export default function EllipseSet({ name, idx }: EllipseSetProps) {
  const { setValue, register, formState, control } = useFormContext();
  const { fields } = useFieldArray({
    control,
    name: `${name}.${idx}.EllipseSet.center`,
  });
  const { fields: radiiFields } = useFieldArray({
    control,
    name: `${name}.${idx}.EllipseSet.radii`,
  });
  return (
    <>
      <span className="font-semibold">Center</span>
      <div className="flex items-center gap-2 p-1 border border-blue-300 rounded border-dashed">
        {fields.map((field, index) => (
          <div className="my-2" key={field.id}>
            <Input
              onPaste={(e: React.ClipboardEvent<HTMLInputElement>) => {
                if (e.clipboardData == null) return;
                e.preventDefault();
                const values = parseNumberList(e.clipboardData.getData("text"));
                let valuesPointer = 0;
                for (
                  let i = index;
                  i < fields.length && valuesPointer < values.length;
                  i++
                ) {
                  setValue(
                    `${name}.${idx}.EllipseSet.center.${i}`,
                    values[valuesPointer]
                  );
                  valuesPointer++;
                }
              }}
              {...register(`${name}.${idx}.EllipseSet.center.${index}`, {
                valueAsNumber: true,
                required: true,
              })}
              className="max-w-24"
              step="any"
              placeholder="0.0"
              type="number"
            />
            <ErrorMessage
              errors={formState.errors}
              name={`${name}.${idx}.EllipseSet.center.${index}`}
              render={({ message }) => (
                <small className="text-red-500">{message}</small>
              )}
            />
            <ErrorMessage
              errors={formState.errors}
              name={`${name}.${idx}.EllipseSet.center.${index}.root`}
              render={({ message }) => (
                <small className="text-red-500">{message}</small>
              )}
            />
          </div>
        ))}
      </div>
      <span className="font-semibold">Radii</span>
      <div className="flex items-center gap-2 p-1 border border-blue-300 rounded border-dashed">
        {radiiFields.map((field, index) => (
          <div className="my-2" key={field.id}>
            <Input
              onPaste={(e: React.ClipboardEvent<HTMLInputElement>) => {
                if (e.clipboardData == null) return;
                e.preventDefault();
                const values = parseNumberList(e.clipboardData.getData("text"));
                let valuesPointer = 0;
                for (
                  let i = index;
                  i < fields.length && valuesPointer < values.length;
                  i++
                ) {
                  setValue(
                    `${name}.${idx}.EllipseSet.center.${i}`,
                    values[valuesPointer]
                  );
                  valuesPointer++;
                }
              }}
              {...register(`${name}.${idx}.EllipseSet.center.${index}`, {
                valueAsNumber: true,
                required: true,
              })}
              className="max-w-24"
              step="any"
              placeholder="0.0"
              type="number"
            />
            <ErrorMessage
              errors={formState.errors}
              name={`${name}.${idx}.EllipseSet.center.${index}`}
              render={({ message }) => (
                <small className="text-red-500">{message}</small>
              )}
            />
            <ErrorMessage
              errors={formState.errors}
              name={`${name}.${idx}.EllipseSet.center.${index}.root`}
              render={({ message }) => (
                <small className="text-red-500">{message}</small>
              )}
            />
          </div>
        ))}
      </div>
    </>
  );
}
