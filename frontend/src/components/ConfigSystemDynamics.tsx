import SamplesInput from "@/components/SamplesInput";
import SystemDynamicsInput from "@/components/SystemDynamicsInput";
import {
  useFormContext,
  type FieldErrors,
  type FieldValues,
  type UseFormSetValue,
} from "react-hook-form";
import { FaEye, FaSpinner } from "react-icons/fa6";
import TabGroup from "./TabGroup";
import { Button } from "./ui/button";
import type { Configuration, RectSet, SphereSet } from "@/utils/schema";
import FormTextInput from "./FormTextInput";
import { useEffect, useState } from "react";

function setDimension(
  getValue: (name: string) => Configuration["X_bounds"],
  setValue: UseFormSetValue<Record<string, any>>
) {
  return (value: number | string) => {
    if (typeof value === "string" || isNaN(value) || value < 1 || value > 9)
      return value;

    for (const setName of ["X_bounds", "X_init", "X_unsafe"]) {
      const prevSets = getValue(setName);
      for (let i = 0; i < prevSets.length; i++) {
        Object.entries(prevSets[i]).forEach(([key, prevSet]) => {
          if (key === "RectSet") {
            if (prevSet.length == value) return;
            const newSet = [...prevSet] as RectSet["RectSet"];
            newSet.length = value;
            newSet.fill([0, 1], prevSet.length);
            setValue(`${setName}.${i}.RectSet`, newSet);
          } else if (key === "SphereSet") {
            if (prevSet.center.length == value) return;
            const newCenter = [
              ...prevSet.center,
            ] as SphereSet["SphereSet"]["center"];
            newCenter.length = value;
            newCenter.fill(1.0, prevSet.center.length);
            setValue(`${setName}.${i}.SphereSet.center`, newCenter);
          } else {
            console.warn(`Unknown set type: ${key}`);
            return;
          }
        });
      }
    }
    return value;
  };
}

function onChangeTab(
  tabName: "Data" | "Model",
  setValue: (name: string, value: any) => void,
  clearErrors: (name?: string | string[]) => void
) {
  return () => {
    setValue("system_dynamics", tabName === "Model" ? ["x1 + 1"] : []);
    clearErrors("system_dynamics");
    setValue("x_samples", []);
    setValue("xp_samples", []);
    clearErrors(["x_samples", "xp_samples"]);
  };
}

export function systemFormErrors(errors: FieldErrors<FieldValues>): boolean {
  return Boolean(
    errors.system_dynamics ||
      errors.X_bounds ||
      errors.X_init ||
      errors.X_unsafe ||
      errors.x_samples ||
      errors.xp_samples
  );
}

type ConfigSystemDynamicsProps = {
  onSubmit: (data: FieldValues) => Promise<void>;
  loading: boolean;
  error: string | null;
  disabled: boolean;
};

export default function ConfigSystemDynamics({
  onSubmit,
  loading,
  error,
  disabled,
}: ConfigSystemDynamicsProps) {
  const { getValues, setValue, clearErrors, handleSubmit, watch } =
    useFormContext();
  const [tabIndex, setTabIndex] = useState(0);
  const x_samples = watch("x_samples");
  const system_dynamics = watch("system_dynamics");

  useEffect(() => {
    if (x_samples.length > 0) {
      setTabIndex(0);
    }
    if (system_dynamics.length > 0) {
      setTabIndex(1);
    }
  }, [x_samples, system_dynamics]);

  return (
    <div>
      <div className="flex flex-row justify-between">
        <h2 className="text-lg font-semibold">System dynamics</h2>
        <div className="flex flex-row items-center gap-2">
          <small className="text-red-500 mt-1">{error}</small>
          <Button
            type="button"
            disabled={disabled}
            onClick={() => handleSubmit(onSubmit)()}
          >
            {loading ? (
              <>
                <FaSpinner className="mr-2 animate-spin" />
                Computing
              </>
            ) : (
              <>
                <FaEye className="inline-block mr-1 size-4" />
                Preview
              </>
            )}
          </Button>
        </div>
      </div>
      <FormTextInput
        name="dimension"
        label="Dimension"
        description="State space dimension"
        type="number"
        min={1}
        step={1}
        max={9}
        onChange={setDimension(getValues, setValue)}
      />
      <TabGroup
        className="my-4"
        selectedIndex={tabIndex}
        setSelectedIndex={setTabIndex}
        tabs={{
          Data: {
            content: (
              <div className="flex flex-col gap-4">
                <h3 className="font-semibold">Dataset</h3>

                <SamplesInput name="x_samples" label="Initial states" />
                <SamplesInput name="xp_samples" label="Consecutive states" />
              </div>
            ),
            onClick: onChangeTab("Data", setValue, clearErrors),
          },
          Model: {
            content: <SystemDynamicsInput />,
            onClick: onChangeTab("Model", setValue, clearErrors),
          },
        }}
      />
    </div>
  );
}
