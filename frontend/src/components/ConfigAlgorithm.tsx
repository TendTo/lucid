import FormTextInput from "@/components/FormTextInput";
import FormSelectionInput from "@/components/FormSelectionInput";
import { type FieldErrors, type FieldValues } from "react-hook-form";

export function algorithmFormErrors(errors: FieldErrors<FieldValues>): boolean {
  return Boolean(
    errors.verbose ||
      errors.gamma ||
      errors.c_coefficient ||
      errors.lambda ||
      errors.num_samples ||
      errors.time_horizon ||
      errors.sigma_f ||
      errors.sigma_l ||
      errors.num_frequencies ||
      errors.oversample_factor ||
      errors.num_oversample ||
      errors.noise_scale ||
      errors.estimator ||
      errors.kernel ||
      errors.feature_map ||
      errors.optimiser
  );
}

export default function ConfigAlgorithm() {
  return (
    <div>
      <h2 className="font-bold text-lg mb-2">Algorithm Parameters</h2>

      <div className="grid grid-cols-[repeat(auto-fill,minmax(300px,1fr))] gap-5">
        <FormTextInput
          name="gamma"
          label="Gamma"
          placeholder="Gamma value"
          type="number"
          step={0.01}
          min={0}
          description="Gamma value for the algorithm"
        />

        <FormTextInput
          name="c_coefficient"
          label="C Coefficient"
          placeholder="C Coefficient value"
          type="number"
          step={0.01}
          min={0}
          description="Strictness parameter for the algorithm. A higher value means more strictness."
        />

        <FormTextInput
          name="lambda"
          label="Lambda"
          placeholder="Lambda value"
          type="number"
          min={0}
          description="Regularization parameter for the algorithm"
        />

        <FormTextInput
          name="num_samples"
          label="Number of Samples"
          placeholder="Enter number of samples"
          type="number"
          min={1}
          description="Total number of samples to use in the algorithm"
        />

        <FormTextInput
          name="time_horizon"
          label="Time Horizon"
          placeholder="Enter time horizon"
          type="number"
          min={1}
          description="Time horizon for the algorithm in seconds"
        />

        <FormTextInput
          name="sigma_f"
          label="Sigma F"
          placeholder="Sigma F value"
          type="number"
          step={0.01}
          min={0}
          description="Sigma F parameter for the algorithm"
        />

        <FormTextInput
          name="sigma_l"
          label="Sigma L"
          placeholder="Sigma L value(s)"
          type="text"
          description="Single value or comma-separated list of Sigma L values"
        />

        <FormTextInput
          name="num_frequencies"
          label="Number of Frequencies"
          placeholder="Enter number of frequencies"
          type="number"
          min={1}
          description="Total number of frequencies to use in the algorithm"
        />

        <FormTextInput
          name="oversample_factor"
          label="Oversample Factor"
          placeholder="Enter oversample factor"
          type="number"
          step={0.01}
          min={0}
          description="Oversample factor for the algorithm"
        />

        <FormTextInput
          name="num_oversample"
          label="Number of Oversamples"
          placeholder="Enter number of oversamples"
          type="number"
          min={-1}
          description="Number of oversamples to use, -1 for no oversampling"
        />

        <FormTextInput
          name="noise_scale"
          label="Noise Scale"
          placeholder="Enter noise scale"
          type="number"
          step={0.001}
          min={0}
          description="Scale of the noise to add to the system dynamics"
        />
      </div>

      <hr className="my-5" />

      <h3 className="font-bold text-lg mb-2">Component Selection</h3>

      <div className="grid grid-cols-[repeat(auto-fill,minmax(300px,1fr))] gap-5">
        <FormSelectionInput
          name="estimator"
          label="Estimator"
          options={{ KernelRidgeRegressor: "Kernel Ridge Regressor" }}
        />

        <FormSelectionInput
          name="kernel"
          label="Kernel"
          options={{ GaussianKernel: "Gaussian Kernel" }}
        />

        <FormSelectionInput
          name="feature_map"
          label="Feature Map"
          options={{
            LinearTruncatedFourierFeatureMap:
              "Linear Truncated Fourier Feature Map",
            ConstantTruncatedFourierFeatureMap:
              "Constant Truncated Fourier Feature Map",
            LogTruncatedFourierFeatureMap: "Log Truncated Fourier Feature Map",
          }}
        />

        <FormSelectionInput
          name="optimiser"
          label="Optimiser"
          options={{
            GurobiOptimiser: "Gurobi Optimiser",
            AlglibOptimiser: "Alglib Optimiser",
            HighsOptimiser: "Highs Optimiser",
          }}
        />
      </div>
    </div>
  );
}
