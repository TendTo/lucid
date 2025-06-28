import { ErrorMessage } from "@hookform/error-message";
import { useFormContext } from "react-hook-form";

type FormCheckboxInputProps = {
  label: string;
  name: string;
  placeholder?: string;
  description?: string;
};

export default function FormCheckboxInput({
  label,
  name,
  description,
  placeholder,
}: FormCheckboxInputProps) {
  const { register, formState } = useFormContext();

  return (
    <div className="flex items-center mr-2 mb-0">
      <input
        type="checkbox"
        id={name}
        placeholder={placeholder}
        className="w-auto mr-2"
        {...register(name)}
      />
      <label className="block font-bold mb-1" htmlFor={name}>
        {label}
      </label>
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
