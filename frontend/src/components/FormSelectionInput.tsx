import { ErrorMessage } from "@hookform/error-message";
import { useFormContext } from "react-hook-form";

type FormTextInputProps = {
  label: string;
  name: string;
  placeholder?: string;
  description?: string;
  options: Record<string | number, string>;
  defaultValue?: string | number;
};

export default function FormTextInput({
  label,
  name,
  description,
  options,
}: FormTextInputProps) {
  const { register, formState } = useFormContext();

  return (
    <div>
      <label className="block font-bold mb-1" htmlFor={name}>
        {label}
      </label>
      <select
        id={name}
        className="w-full border rounded mb-1 px-3 py-2 border-solid border-[#ddd]"
        {...register(name)}
      >
        {Object.entries(options).map(([value, label]) => (
          <option key={value} value={value}>
            {label}
          </option>
        ))}
      </select>
      {description && <small>{description}</small>}
      <ErrorMessage
        errors={formState.errors}
        name={name}
        render={({ message }) => (
          <small className="text-red-500">{message}</small>
        )}
      />
    </div>
  );
}
