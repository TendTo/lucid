import FormTextInput from "@components/FormTextInput";
import FormCheckboxInput from "@components/FormCheckboxInput";
import FormSelectionInput from "@components/FormSelectionInput";
import { type FieldErrors, type FieldValues } from "react-hook-form";

export function executionFormErrors(errors: FieldErrors<FieldValues>): boolean {
  return Boolean(
    errors.verbose ||
      errors.seed ||
      errors.plot ||
      errors.verify ||
      errors.problem_log_file ||
      errors.iis_log_file
  );
}

export default function ConfigExecution() {
  return (
    <div>
      <h2 className="font-bold text-lg mb-2">Execution and Output Options</h2>

      <div className="grid grid-cols-[repeat(auto-fill,minmax(300px,1fr))] gap-5">
        <FormSelectionInput
          name="verbose"
          label="Verbosity Level"
          valueAsNumber
          options={{
            [-1]: "Silent",
            0: "Critical",
            1: "Error",
            2: "Warning",
            3: "Info",
            4: "Debug",
            5: "Trace",
          }}
        />

        <FormTextInput
          name="seed"
          label="Random Seed"
          placeholder="Enter random seed"
          type="number"
          min={-1}
          description="Use -1 for random seed"
        />

        <FormCheckboxInput
          name="plot"
          label="Enable Plot"
          placeholder="Enable plotting of results"
        />

        <FormCheckboxInput
          name="verify"
          label="Verify Results"
          placeholder="Enable result verification"
        />

        <FormTextInput
          name="problem_log_file"
          label="Problem Log File"
          placeholder="Enter problem log file path"
        />

        <FormTextInput
          name="iis_log_file"
          label="IIS Log File"
          placeholder="Enter IIS log file path"
        />
      </div>
    </div>
  );
}
