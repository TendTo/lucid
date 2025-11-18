import FormSelectionInput from "@/components/FormSelectionInput";
import FormTextInput from "@/components/FormTextInput";
import FormCheckboxInput from "./FormCheckboxInput";
import { useCapabilities } from "@/hooks/useCapabilities";
import { parseNumberListOrString } from "@/utils/utils";

export default function ConfigAdvanced() {
  const { ALGLIB, GUROBI, HIGHS, PLOT } = useCapabilities();
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
          description="Minimum value of the barrier over the unsafe set"
          required
        />
        <FormTextInput
          name="epsilon"
          label="Epsilon"
          placeholder="Epsilon value"
          type="number"
          min={0}
          description="Robustifying radius"
          required
        />
        <FormTextInput
          name="b_kappa"
          label="Barrier kappa"
          placeholder="Barrier kappa value"
          type="number"
          step={0.01}
          min={0}
          description="Barrier kappa value for the optimisation"
          required
        />
        <FormTextInput
          name="b_norm"
          label="Barrier norm"
          placeholder="Barrier norm value"
          type="number"
          step={0.01}
          min={0}
          description="Norm of the barrier function"
          required
        />

        <FormTextInput
          name="C_coeff"
          label="C coefficient"
          placeholder="C coefficient value"
          type="number"
          step={0.01}
          min={0}
          description="Conservativeness parameter for the algorithm"
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
          description="Number of transitions to sample. Only used if the data is not provided"
          required
        />

        <FormTextInput
          name="time_horizon"
          label="Time Horizon"
          placeholder="Enter time horizon"
          type="number"
          min={1}
          description="Number of time steps to consider in the specification"
          required
        />

        <FormTextInput
          name="sigma_f"
          label="Sigma F"
          placeholder="Sigma F value"
          type="number"
          step={0.01}
          min={0}
          description="Amplitude parameter of the kernel"
          required
        />

        <FormTextInput
          name="sigma_l"
          label="Sigma L"
          placeholder="Sigma L value(s)"
          type="text"
          description="Single or comma-separated list of length scales of the kernel"
          required
          onChange={(value: string | number) => {
            if (typeof value === "number") return value;
            return parseNumberListOrString(value);
          }}
        />

        <FormTextInput
          name="num_frequencies"
          label="Number of Frequencies"
          placeholder="Enter number of frequencies"
          type="number"
          min={1}
          description="Number of frequencies to use in the algorithm. Includes the constant term (0 frequency)"
          required
        />

        <FormTextInput
          name="oversample_factor"
          label="Oversample factor"
          placeholder="Oversample factor"
          type="number"
          step={0.01}
          min={0}
          description="Lattice points for each dimension in terms of the nyquist frequency. Only used if the number of lattice points is not provided"
          required
        />

        <FormTextInput
          name="lattice_resolution"
          label="Lattice resolution"
          placeholder="Lattice resolution"
          type="number"
          min={-1}
          description="Lattice points for each dimension"
          required
        />

        <FormTextInput
          name="noise_scale"
          label="Noise Scale"
          placeholder="Enter noise scale"
          type="number"
          step={0.001}
          min={0}
          description="Scale of the noise to add to the system dynamics. Only used if the data is not provided"
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
            ...(GUROBI ? { GurobiOptimiser: "Gurobi Optimiser" } : {}),
            ...(ALGLIB ? { AlglibOptimiser: "Alglib Optimiser" } : {}),
            ...(HIGHS ? { HighsOptimiser: "Highs Optimiser" } : {}),
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

        {PLOT && (
          <FormCheckboxInput
            name="plot"
            label="Enable Plot"
            description="Enable plotting of results"
          />
        )}

        <FormTextInput
          name="problem_log_file"
          label="Problem Log File"
          placeholder="Log file path"
        />

        <FormTextInput
          name="iis_log_file"
          label="IIS Log File"
          placeholder="IIS log file path"
        />
      </div>
    </div>
  );
}
