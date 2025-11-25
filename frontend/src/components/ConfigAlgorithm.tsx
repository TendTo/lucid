import { type FieldErrors, type FieldValues } from "react-hook-form";

export function algorithmFormErrors(errors: FieldErrors<FieldValues>): boolean {
  return Boolean(
    errors.verbose ||
      errors.gamma ||
      errors.C_coeff ||
      errors.lambda ||
      errors.num_samples ||
      errors.time_horizon ||
      errors.sigma_f ||
      errors.sigma_l ||
      errors.feature_sigma_l ||
      errors.num_frequencies ||
      errors.oversample_factor ||
      errors.lattice_resolution ||
      errors.set_scaling ||
      errors.noise_scale ||
      errors.estimator ||
      errors.kernel ||
      errors.feature_map ||
      errors.optimiser
  );
}
