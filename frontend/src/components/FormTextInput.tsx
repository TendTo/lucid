import { useFormContext } from "react-hook-form";
import { FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage } from "./ui/form";
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
}: FormTextInputProps) {
  const { control } = useFormContext();

  return (
    <FormField
      control={control}
      name={name}
      render={({ field }) => (
        <FormItem>
          <div className="flex items-center justify-between w-full">
            <FormLabel>{label}</FormLabel>
            <FormMessage />
          </div>
          <FormControl>
            <Input type={type} placeholder={placeholder} min={min} max={max} step={step} {...field} />
          </FormControl>
          <FormDescription>{description}</FormDescription>
        </FormItem>
      )}
    />
  );
}
