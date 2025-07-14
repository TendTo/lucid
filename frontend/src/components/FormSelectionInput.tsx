import { useFormContext } from "react-hook-form";
import {
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "./ui/form";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "./ui/select";

type FormTextInputProps = {
  label: string;
  name: string;
  placeholder?: string;
  description?: string;
  options: Record<string | number, string>;
  valueAsNumber?: boolean;
};

export default function FormSelectionInput({
  label,
  name,
  description,
  options,
  valueAsNumber = false,
}: FormTextInputProps) {
  const { control } = useFormContext();
  return (
    <>
      <FormField
        control={control}
        name={name}
        render={({ field }) => (
          <FormItem>
            <div className="flex items-center justify-between">
              <FormLabel htmlFor={name}>{label}</FormLabel>
              <FormMessage />
            </div>
            <FormControl className="">
              <Select
                onValueChange={(v) =>
                  field.onChange(valueAsNumber ? Number(v) : v)
                }
                defaultValue={field.value.toString()}
                name={field.name}
                key={field.name}
              >
                <SelectTrigger id={field.name} className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectGroup>
                    <SelectLabel>{label}</SelectLabel>
                    {Object.entries(options).map(([value, label]) => (
                      <SelectItem key={value} value={value}>
                        {label}
                      </SelectItem>
                    ))}
                  </SelectGroup>
                </SelectContent>
              </Select>
            </FormControl>
            <FormDescription>{description}</FormDescription>
          </FormItem>
        )}
      />
    </>
  );
}
