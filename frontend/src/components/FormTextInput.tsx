import { useFormContext } from "react-hook-form";
import {
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "./ui/form";
import { Input } from "./ui/input";

type FormTextInputProps = {
  type?: "text" | "number";
  label: string;
  name: string;
  placeholder?: string;
  description?: string;
  min?: number;
  max?: number;
  step?: number | "any";
  required?: boolean;
  form?: string;
  onChange?: (value: number | string) => number | string;
};

export default function FormTextInput({
  type = "text",
  label,
  name,
  description,
  placeholder,
  min = undefined,
  max = undefined,
  step = "any",
  required = false,
  form = undefined,
  onChange = undefined,
}: FormTextInputProps) {
  const { control } = useFormContext();

  return (
    <FormField
      control={control}
      name={name}
      render={({ field }) => {
        const customOnChange =
          onChange == undefined
            ? field.onChange
            : (e: string | number) => field.onChange(onChange(e));
        const valueOnChange =
          type === "number"
            ? (e: string) => customOnChange(e === "" ? "" : Number(e))
            : customOnChange;
        return (
          <FormItem>
            <div className="flex items-center justify-between w-full">
              <FormLabel>{label}</FormLabel>
              <FormMessage />
            </div>
            <FormControl>
              <Input
                type={type}
                required={required}
                placeholder={placeholder}
                min={min}
                max={max}
                step={step}
                form={form}
                {...field}
                onChange={(e) => valueOnChange(e.target.value)}
              />
            </FormControl>
            <FormDescription>{description}</FormDescription>
          </FormItem>
        );
      }}
    />
  );
}
