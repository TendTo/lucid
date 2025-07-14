import FormSelectionInput from "@/components/FormSelectionInput";
import FormTextInput from "@/components/FormTextInput";
import FormCheckboxInput from "./FormCheckboxInput";

export default function ConfigAdvanced() {
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
          required
        />

        <FormTextInput
          name="c_coefficient"
          label="C Coefficient"
          placeholder="C Coefficient value"
          type="number"
          step={0.01}
          min={0}
          description="Strictness parameter for the algorithm. A higher value means more strictness."
          required
        />

        <FormTextInput
          name="lambda"
          label="Lambda"
          placeholder="Lambda value"
          type="number"
          min={0}
          description="Regularization parameter for the algorithm"
          required
        />

        <FormTextInput
          name="num_samples"
          label="Number of Samples"
          placeholder="Enter number of samples"
          type="number"
          min={1}
          description="Total number of samples to use in the algorithm"
          required
        />

        <FormTextInput
          name="time_horizon"
          label="Time Horizon"
          placeholder="Enter time horizon"
          type="number"
          min={1}
          description="Time horizon for the algorithm in seconds"
          required
        />

        <FormTextInput
          name="sigma_f"
          label="Sigma F"
          placeholder="Sigma F value"
          type="number"
          step={0.01}
          min={0}
          description="Sigma F parameter for the algorithm"
          required
        />

        <FormTextInput
          name="sigma_l"
          label="Sigma L"
          placeholder="Sigma L value(s)"
          type="text"
          description="Single value or comma-separated list of Sigma L values"
          required
        />

        <FormTextInput
          name="num_frequencies"
          label="Number of Frequencies"
          placeholder="Enter number of frequencies"
          type="number"
          min={1}
          description="Total number of frequencies to use in the algorithm"
          required
        />

        <FormTextInput
          name="oversample_factor"
          label="Oversample Factor"
          placeholder="Enter oversample factor"
          type="number"
          step={0.01}
          min={0}
          description="Oversample factor for the algorithm"
          required
        />

        <FormTextInput
          name="num_oversample"
          label="Number of Oversamples"
          placeholder="Enter number of oversamples"
          type="number"
          min={-1}
          description="Number of oversamples to use, -1 for no oversampling"
          required
        />

        <FormTextInput
          name="noise_scale"
          label="Noise Scale"
          placeholder="Enter noise scale"
          type="number"
          step={0.001}
          min={0}
          description="Scale of the noise to add to the system dynamics"
          required
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

      <hr className="my-5" />

      <h2 className="font-bold text-lg mb-2">Execution and Output</h2>

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
          description="Enable plotting of results"
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
