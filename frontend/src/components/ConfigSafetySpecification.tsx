import SetsInput from "@/components/SetsInput";

export default function ConfigSafetySpecification() {
  return (
    <div>
      <h2 className="text-lg font-semibold">Safety specification</h2>
      <SetsInput name="X_bounds" label="X bounds" />
      <SetsInput name="X_init" label="X init" />
      <SetsInput name="X_unsafe" label="X unsafe" />
    </div>
  );
}
