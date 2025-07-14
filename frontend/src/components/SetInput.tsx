import { ErrorMessage } from "@hookform/error-message";
import { useFieldArray, useFormContext } from "react-hook-form";
import { FaMinus } from "react-icons/fa6";
import { Button } from "./ui/button";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { availableSets } from "@/utils/constants";
import { useState } from "react";
import { Input } from "./ui/input";
import { parseNumberList } from "@/utils/utils";

type SetInputProps = {
  name: string;
  idx: number;
  label: string;
  type: "RectSet";
  removeItself?: () => void;
};

export default function SetInput({ name, idx, removeItself }: SetInputProps) {
  const [type, setType] = useState<string>("RectSet");
  const { register, formState, getValues, setValue, control } =
    useFormContext();

  const { fields } = useFieldArray({
    control,
    name: `${name}.${idx}.${type}`,
  });

  return (
    <div className="flex flex-row gap-2 items-center">
      <Button
        onClick={removeItself}
        variant="destructive"
        disabled={removeItself == undefined}
      >
        <FaMinus />
      </Button>
      <Select
        onValueChange={(e) => {
          setValue(`${name}.${idx}`, { [e]: [] });
          setType(e);
        }}
        defaultValue={Object.keys(availableSets).at(0)}
      >
        <SelectTrigger>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectGroup>
            {Object.entries(availableSets).map(([value, label]) => (
              <SelectItem key={value} value={value}>
                {label}
              </SelectItem>
            ))}
          </SelectGroup>
        </SelectContent>
      </Select>
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
                      `${name}.${idx}.${type}.${i}.0`,
                      values[valuesPointer]
                    );
                    valuesPointer++;
                    if (valuesPointer >= values.length) continue;
                    setValue(
                      `${name}.${idx}.${type}.${i}.1`,
                      values[valuesPointer]
                    );
                    valuesPointer++;
                  }
                }}
                {...register(`${name}.${idx}.${type}.${index}.0`, {
                  valueAsNumber: true,
                  required: true,
                })}
                className="max-w-24"
                step="any"
                placeholder="0.0"
                type="number"
              />
              <Input
                {...register(`${name}.${idx}.${type}.${index}.1`, {
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
                    `${name}.${idx}.${type}.${index}.1`,
                    values[0] || ""
                  );
                  let valuesPointer = 1;
                  for (
                    let i = index + 1;
                    i < fields.length && valuesPointer < values.length;
                    i++
                  ) {
                    setValue(
                      `${name}.${idx}.${type}.${i}.0`,
                      values[valuesPointer]
                    );
                    valuesPointer++;
                    if (valuesPointer >= values.length) continue;
                    setValue(
                      `${name}.${idx}.${type}.${i}.1`,
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
    </div>
  );
}
