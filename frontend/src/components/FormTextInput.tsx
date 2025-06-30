import { ErrorMessage } from "@hookform/error-message";
import { useFormContext } from "react-hook-form";

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
  const { register, formState } = useFormContext();

  return (
    <div>
      <label className="block font-bold mb-1" htmlFor={name}>
        {label}
      </label>
      <input
        type={type}
        id={name}
        placeholder={placeholder}
        min={min}
        max={max}
        step={step}
        className="w-full border rounded px-3 py-2 border-solid border-[#ddd]"
        {...register(name, {
          valueAsNumber: type === "number",
        })}
      />
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
