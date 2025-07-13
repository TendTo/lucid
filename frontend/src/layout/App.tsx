import ConfigAlgorithm, {
  algorithmFormErrors,
} from "@/components/ConfigAlgorithm";
import ConfigExecution, {
  executionFormErrors,
} from "@/components/ConfigExecution";
import ConfigSystem, { systemFormErrors } from "@/components/ConfigSystem";
import Header from "@/layout/Header";
import JsonPreview from "@/components/JsonPreview";
import type {
  FormStepName,
  FormSteps,
  LogEntry,
  ServerResponse,
} from "@/types/types";
import { defaultValues, emptyFigure } from "@/utils/constants";
import { parseLogEntry } from "@/utils/parseLog";
import { configurationSchema } from "@/utils/schema";
import { zodResolver } from "@hookform/resolvers/zod";
import "plotly.js-dist-min";
import { useCallback, useState } from "react";
import { FormProvider, useForm } from "react-hook-form";
import { FaPaperPlane, FaSpinner } from "react-icons/fa6";
import type { PlotParams } from "react-plotly.js";
import Result from "@/components/Result";
import Tabs from "./TabGroup";
import Footer from "./Footer";
import OutputSection from "./OutputSection";
import InputSection from "./InputSection";

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
  const [formSteps, setFormSteps] = useState<FormSteps>(initialFormSteps);
  const [fig, setFig] = useState<PlotParams>(emptyFigure);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [submitError, setSubmitError] = useState<string>("");

  const methods = useForm({
    resolver: zodResolver(configurationSchema),
    defaultValues: defaultValues as any,
    mode: "onChange",
  });

  const onSubmit = useCallback(
    async (data: typeof defaultValues) => {
      if (isConnected) {
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
      setLogs([]);
      setFig(emptyFigure);
      const response = await fetch("/api/run", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      });
      if (response.status !== 202) {
        const error: ServerResponse = await response.json();
        if (error.cause) {
          methods.setError(error.cause, {
            type: "value",
            message: error.error,
          });
        } else {
          setSubmitError(error.error ?? "Unknown error");
        }
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
      };

      // Handle connection open
      eventSource.onopen = () => {
        setIsConnected(true);
      };

      // Handle errors and close events
      eventSource.onerror = (error) => {
        if (error.eventPhase !== EventSource.CLOSED)
          console.error("SSE error:", error);
        setIsConnected(false);
        eventSource.close();
      };

      // Clean up on unmount
      return () => {
        eventSource.close();
        setIsConnected(false);
      };
    },
    [isConnected, setIsConnected, setLogs, setFig, methods]
  );

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
    <div className="bg-background text-foreground flex min-h-svh flex-col">
      <Header
        errors={methods.formState.errors}
        steps={formSteps}
        setCurrentStep={setCurrentStep}
        reset={methods.reset}
      />
      <main className="flex-grow min-w-full mx-auto p-4 flex flex-row justify-evenly relative">
        <InputSection />
        <OutputSection />
        <aside className="">
          <h3>JSON Preview</h3>
          <JsonPreview formData={methods.watch()} />
        </aside>
      </main>
      <Footer />
    </div>
  );
}
