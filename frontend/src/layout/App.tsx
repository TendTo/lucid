import { algorithmFormErrors } from "@/components/ConfigAlgorithm";
import { executionFormErrors } from "@/components/ConfigExecution";
import { systemFormErrors } from "@/components/ConfigSystemDynamics";
import Header from "@/layout/Header";
import type {
  FormStepName,
  FormSteps,
  LogEntry,
  ServerResponse,
  SuccessResponseData,
} from "@/types/types";
import { defaultValues, emptyFigure } from "@/utils/constants";
import { parseLogEntry } from "@/utils/parseLog";
import { configurationSchema, type Configuration } from "@/utils/schema";
import { zodResolver } from "@hookform/resolvers/zod";
import { useCallback, useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import type { PlotParams } from "react-plotly.js";
import OutputSection from "./OutputSection";
import InputSection from "./InputSection";
import { capableConfiguration } from "@/utils/utils";
import { useCapabilities } from "@/hooks/useCapabilities";

const initialFormSteps = {
  system: {
    name: "System",
    href: "#",
    current: true,
    error: systemFormErrors,
  },
  algorithm: {
    name: "Algorithm",
    href: "#",
    current: false,
    error: algorithmFormErrors,
  },
  execution: {
    name: "Execution",
    href: "#",
    current: false,
    error: executionFormErrors,
  },
} as FormSteps;

export default function App() {
  const capabilities = useCapabilities();
  const [formSteps, setFormSteps] = useState<FormSteps>(initialFormSteps);
  const [fig, setFig] = useState<PlotParams>(emptyFigure);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [successData, setSuccessData] = useState<SuccessResponseData | null>(
    null
  );

  const [submitError, setSubmitError] = useState<string>("");
  const [previewError, setPreviewError] = useState<string>("");
  const [submitLoading, setSubmitLoading] = useState<boolean>(false);
  const [previewLoading, setPreviewLoading] = useState<boolean>(false);

  const methods = useForm({
    resolver: zodResolver(configurationSchema),
    defaultValues: defaultValues,
    mode: "onBlur",
  });

  useEffect(() => {
    // If the capabilities change, reset the form to the correct defaults
    methods.reset(capableConfiguration(defaultValues, capabilities));
  }, [methods, capabilities]);

  const resetOutput = useCallback(() => {
    setFig(emptyFigure);
    setLogs([]);
    setSuccessData(null);
    setSubmitError("");
    setPreviewError("");
    setSubmitLoading(false);
    setPreviewLoading(false);
  }, [
    setFig,
    setLogs,
    setSuccessData,
    setSubmitError,
    setPreviewError,
    setSubmitLoading,
    setPreviewLoading,
  ]);

  const onSubmit = useCallback(
    async (data: Configuration) => {
      if (previewLoading || submitLoading) {
        console.warn("Already connected, ignoring new submission");
        return;
      }
      // Validate form data
      const isValid =
        (await methods.trigger()) &&
        (await methods.trigger([
          "system_dynamics",
          "X_bounds",
          "X_init",
          "X_unsafe",
          "x_samples",
          "xp_samples",
        ]));
      if (!isValid) {
        console.error("Form validation failed");
        console.error(methods.formState.errors);
        return;
      }
      resetOutput();
      setSubmitLoading(true);
      const response = await fetch("/api/run", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      });
      if (response.status !== 202) {
        try {
          const error: ServerResponse = await response.json();
          if (error.cause) {
            methods.setError(error.cause as keyof Configuration, {
              type: "value",
              message: error.error,
            });
          } else {
            setSubmitError(error.error ?? "Unknown error");
          }
        } catch {
          setSubmitError("Unknown error");
        }
        setSubmitLoading(false);
        return;
      }

      // Create SSE connection
      const eventSource = new EventSource("/api/run", {});

      // Handle incoming messages
      eventSource.onmessage = (event) => {
        const data: ServerResponse = JSON.parse(event.data);
        if (data.log)
          setLogs((prevLogs) => [...prevLogs, parseLogEntry(data.log)]);
        try {
          setFig(data.fig ? JSON.parse(data.fig) : emptyFigure);
        } catch (e) {
          setFig(emptyFigure);
          console.error("Failed to parse figure data:", e);
        }

        if (data.success !== undefined) {
          setSuccessData({
            success: data.success,
            obj_val: data.obj_val,
            eta: data.eta,
            c: data.c,
            norm: data.norm,
            verified: data.verified,
            time: data.time ?? 0,
          });
        }
      };

      // Handle connection open
      eventSource.onopen = () => {
        setSubmitLoading(true);
      };

      // Handle errors and close events
      eventSource.onerror = (error) => {
        if (error.eventPhase !== EventSource.CLOSED) {
          console.error("SSE error:", error);
          setSubmitError("Error during submission");
        }
        setSubmitLoading(false);
        eventSource.close();
      };

      // Clean up on unmount
      return () => {
        eventSource.close();
        setSubmitLoading(false);
      };
    },
    [
      setLogs,
      setFig,
      methods,
      setSubmitError,
      setSubmitLoading,
      previewLoading,
      submitLoading,
      resetOutput,
    ]
  );

  const onPreview = useCallback(async () => {
    if (
      !(await methods.trigger([
        "system_dynamics",
        "X_bounds",
        "X_init",
        "X_unsafe",
        "x_samples",
        "xp_samples",
      ]))
    ) {
      console.error("Form validation failed");
      return;
    }
    const [x_samples, xp_samples] = methods.getValues([
      "x_samples",
      "xp_samples",
    ]);
    if (x_samples.length > 0 && xp_samples.length > 0) {
      if (x_samples[0].length !== xp_samples[0].length) {
        return setPreviewError(
          "Samples must have the same dimension to be previewed"
        );
      }
      if (x_samples[0].length > 3) {
        return setPreviewError("4+ dimensional samples cannot be previewed");
      }
    }
    resetOutput();
    setPreviewLoading(true);
    const response = await fetch("/api/preview-graph", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(methods.getValues()),
    });
    setPreviewLoading(false);
    if (!response.ok) {
      setFig({ data: [], layout: {} });
      const error: ServerResponse = await response.json();
      if (error.cause) {
        methods.setError(error.cause as keyof Configuration, {
          type: "value",
          message: error.error,
        });
      }
      setPreviewError(error.error ?? "Unknown error");
      return;
    }
    const json = await response.json();
    try {
      setFig(json.fig ? JSON.parse(json.fig) : emptyFigure);
    } catch (e) {
      setFig(emptyFigure);
      console.error("Failed to parse figure data:", e);
    }
  }, [methods, setFig, setPreviewError, setPreviewLoading, resetOutput]);

  const setCurrentStep = useCallback(
    (step: FormStepName) => {
      setFormSteps((prev) =>
        Object.entries(prev).reduce((acc, [key, value]) => {
          acc[key as FormStepName] = {
            ...value,
            current: key === step,
          };
          return acc;
        }, {} as FormSteps)
      );
    },
    [setFormSteps]
  );

  return (
    <div className="bg-background text-foreground min-h-svh">
      <Header
        errors={methods.formState.errors}
        steps={formSteps}
        setCurrentStep={setCurrentStep}
        // @ts-expect-error reset is not typed properly
        reset={methods.reset}
        // @ts-expect-error methods is not typed properly
        methods={methods}
      />
      <main className="h-[calc(100vh-4rem)] min-w-full mx-auto flex flex-row justify-evenly relative overflow-y-hidden">
        <InputSection
          // @ts-expect-error methods is not typed properly
          methods={methods}
          submitError={submitError}
          previewError={previewError}
          handlePreview={onPreview}
          handleSubmit={methods.handleSubmit(onSubmit)}
          submitLoading={submitLoading}
          previewLoading={previewLoading}
        />
        <OutputSection
          fig={fig}
          logs={logs}
          loading={submitLoading || previewLoading}
          successData={successData}
        />
      </main>
      {/* <Footer methods={methods} /> */}
    </div>
  );
}
