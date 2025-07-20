import { ErrorMessage } from "@hookform/error-message";
import { useFormContext } from "react-hook-form";
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
import RectSet from "./RectSet";
import type { SetType } from "@/types/types";
import SphereSet from "./SphereSet";
import { defaultSetWithDimension } from "@/utils/utils";

type SetInputProps = {
  name: string;
  idx: number;
  label: string;
  type: "RectSet";
  removeItself?: () => void;
};

export default function SetInput({ name, idx, removeItself }: SetInputProps) {
  const { formState, setValue, watch, getValues } = useFormContext();
  const type = Object.keys(watch(`${name}.${idx}`))[0] ?? "RectSet";

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
        onValueChange={(e: SetType) =>
          setValue(`${name}.${idx}`, {
            [e]: defaultSetWithDimension(e, getValues("dimension")),
          })
        }
        defaultValue={type}
      >
        <SelectTrigger disabled={name === "X_bounds"}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectGroup>
            {Object.entries(availableSets).map(([value, label]) => (
              <SelectItem key={value} value={value} disabled={name === "X_bounds"}>
                {label}
              </SelectItem>
            ))}
          </SelectGroup>
        </SelectContent>
      </Select>
      {type === "RectSet" ? (
        <RectSet name={name} idx={idx} />
      ) : type === "SphereSet" ? (
        <SphereSet name={name} idx={idx} />
      ) : null}
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
