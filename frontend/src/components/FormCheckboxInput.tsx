import { useFormContext } from "react-hook-form";
import { FormControl, FormField, FormItem, FormLabel } from "./ui/form";
import { Checkbox } from "./ui/checkbox";

type FormCheckboxInputProps = {
  label: string;
  name: string;
  description?: string;
};

export default function FormCheckboxInput({
  label,
  name,
  description,
}: FormCheckboxInputProps) {
  const { control } = useFormContext();

  return (
    <FormField
      control={control}
      name={name}
      render={({ field }) => (
        <FormItem className="flex flex-row items-center gap-2">
          <FormControl>
            <Checkbox checked={field.value} onCheckedChange={field.onChange} />
          </FormControl>
          <FormLabel className="text-sm font-normal">{label}</FormLabel>
        </FormItem>
      )}
    />
  );
}
