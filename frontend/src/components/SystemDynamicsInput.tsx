import { ErrorMessage } from "@hookform/error-message";
import { useFieldArray, useFormContext } from "react-hook-form";
import { FaMinus, FaPlus } from "react-icons/fa6";
import FormCheckboxInput from "./FormCheckboxInput";
import { Button } from "./ui/button";
import { Input } from "./ui/input";

export default function SystemDynamicsInput() {
  const { register, control, formState } = useFormContext();
  const { fields, append, remove } = useFieldArray({
    control,
    name: "system_dynamics",
  });

  return (
    <div className="form-group">
      <label className="block font-bold">System Dynamics</label>
      {fields.map((field, index) => (
        <div className="my-2" key={field.id}>
          <div className="flex items-center gap-2">
            <Button
              variant="destructive"
              onClick={() => remove(index)}
              disabled={fields.length <= 1}
            >
              <FaMinus />
            </Button>
            <Input
              {...register(`system_dynamics.${index}`)}
              placeholder="x1 + sin(x2)"
            />
          </div>
          <ErrorMessage
            errors={formState.errors}
            name={`system_dynamics.${index}`}
            render={({ message }) => (
              <small className="text-red-500">{message}</small>
            )}
          />
        </div>
      ))}
      <ErrorMessage
        errors={formState.errors}
        name="system_dynamics"
        render={({ message }) => (
          <small className="text-red-500 block mb-1">{message}</small>
        )}
      />
      <ErrorMessage
        errors={formState.errors}
        name="system_dynamics.root"
        render={({ message }) => (
          <small className="text-red-500 block mb-1">{message}</small>
        )}
      />
      <Button onClick={() => append(`x${fields.length + 1}`)}>
        <FaPlus />
      </Button>
      <div className="mt-4">
        <FormCheckboxInput
          name="verify"
          label="Verify Results"
          description="Enable result verification"
        />
      </div>
    </div>
  );
}
