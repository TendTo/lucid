import { parseNumberList } from "@/utils/utils";
import { Input } from "./ui/input";
import { ErrorMessage } from "@hookform/error-message";
import { useFieldArray, useFormContext } from "react-hook-form";

type RectSetProps = {
  name: string;
  idx: number;
};

export default function RectSet({ name, idx }: RectSetProps) {
  const { setValue, register, formState, control } = useFormContext();
  const { fields } = useFieldArray({
    control,
    name: `${name}.${idx}.RectSet`,
  });

  return (
    <>
      <span className="font-semibold">Bounds</span>
      {fields.map((field, index) => (
        <div className="my-2" key={field.id}>
          <div className="flex items-center gap-2">
            <div className="flex flex-col gap-1 p-1 border border-blue-300 rounded border-dashed">
              <Input
                onPaste={(e: React.ClipboardEvent<HTMLInputElement>) => {
                  if (e.clipboardData == null) return;
                  e.preventDefault();
                  const values = parseNumberList(
                    e.clipboardData.getData("text")
                  );
                  let valuesPointer = 0;
                  for (
                    let i = index;
                    i < fields.length && valuesPointer < values.length;
                    i++
                  ) {
                    setValue(
                      `${name}.${idx}.RectSet.${i}.0`,
                      values[valuesPointer]
                    );
                    valuesPointer++;
                    if (valuesPointer >= values.length) continue;
                    setValue(
                      `${name}.${idx}.RectSet.${i}.1`,
                      values[valuesPointer]
                    );
                    valuesPointer++;
                  }
                }}
                {...register(`${name}.${idx}.RectSet.${index}.0`, {
                  valueAsNumber: true,
                  required: true,
                })}
                className="max-w-24"
                step="any"
                placeholder="0.0"
                type="number"
              />
              <Input
                {...register(`${name}.${idx}.RectSet.${index}.1`, {
                  valueAsNumber: true,
                  required: true,
                })}
                onPaste={(e: React.ClipboardEvent<HTMLInputElement>) => {
                  if (e.clipboardData == null) return;
                  e.preventDefault();
                  const values = parseNumberList(
                    e.clipboardData.getData("text")
                  );
                  setValue(
                    `${name}.${idx}.RectSet.${index}.1`,
                    values[0] || ""
                  );
                  let valuesPointer = 1;
                  for (
                    let i = index + 1;
                    i < fields.length && valuesPointer < values.length;
                    i++
                  ) {
                    setValue(
                      `${name}.${idx}.RectSet.${i}.0`,
                      values[valuesPointer]
                    );
                    valuesPointer++;
                    if (valuesPointer >= values.length) continue;
                    setValue(
                      `${name}.${idx}.RectSet.${i}.1`,
                      values[valuesPointer]
                    );
                    valuesPointer++;
                  }
                }}
                className="max-w-24"
                placeholder="0.0"
                step="any"
                type="number"
              />
            </div>
          </div>
          <ErrorMessage
            errors={formState.errors}
            name={`${name}.${idx}.RectSet.${index}`}
            render={({ message }) => (
              <small className="text-red-500">{message}</small>
            )}
          />
          <ErrorMessage
            errors={formState.errors}
            name={`${name}.${idx}.RectSet.${index}.root`}
            render={({ message }) => (
              <small className="text-red-500">{message}</small>
            )}
          />
        </div>
      ))}
    </>
  );
}
