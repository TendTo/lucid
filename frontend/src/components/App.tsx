import { useCallback, useState } from "react";
import { useForm, FormProvider } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { jsonSchema } from "@utils/schema";
import Header from "@components/Header";
import ConfigSystem, { systemFormErrors } from "@components/ConfigSystem";
import ConfigAlgorithm, {
  algorithmFormErrors,
} from "@components/ConfigAlgorithm";
import ConfigExecution, {
  executionFormErrors,
} from "@components/ConfigExecution";
import JsonPreview from "@components/JsonPreview";
import { FaPaperPlane } from "react-icons/fa6";
import type {
  EstimatorType,
  FeatureMapType,
  FormStepName,
  FormSteps,
  KernelType,
  LogEntry,
  OptimiserType,
  RectSet,
  ServerResponse,
} from "@app/types/types";
import { parseLogEntry } from "@app/utils/parseLog";
import Result from "./Result";
import "plotly.js-dist-min";

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

export const defaultValues = {
  verbose: 3,
  seed: -1,
  x_samples: [] as number[][],
  xp_samples: [] as number[][],
  system_dynamics: [] as string[],
  X_bounds: [] as RectSet[],
  X_init: [] as RectSet[],
  X_unsafe: [] as RectSet[],
  gamma: 1.0,
  c_coefficient: 1.0,
  lambda: 1.0,
  num_samples: 1000,
  time_horizon: 5,
  sigma_f: 15.0,
  sigma_l: 1.0,
  num_frequencies: 4,
  oversample_factor: 2.0,
  num_oversample: -1,
  noise_scale: 0.01,
  plot: false,
  verify: true,
  problem_log_file: "problem.lp",
  iis_log_file: "iis.ilp",
  estimator: "KernelRidgeRegressor" as EstimatorType,
  kernel: "GaussianKernel" as KernelType,
  feature_map: "LinearTruncatedFourierFeatureMap" as FeatureMapType,
  optimiser: "GurobiOptimiser" as OptimiserType,
};

export default function App() {
  const [formSteps, setFormSteps] = useState<FormSteps>(initialFormSteps);
  const [fig, setFig] = useState<string>("");
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [, setIsConnected] = useState(false);

  const methods = useForm({
    resolver: zodResolver(jsonSchema),
    defaultValues: defaultValues as any,
    mode: "onChange",
  });

  const onSubmit = useCallback(
    async (data: typeof defaultValues) => {
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
      setFig("");
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
        }
        throw new Error(
          `Error fetching graph preview: ${error.error ?? "Unknown error"}`
        );
      }

      // Create SSE connection
      const eventSource = new EventSource("/api/run", {});

      // Handle incoming messages
      eventSource.onmessage = (event) => {
        const data: ServerResponse = JSON.parse(event.data);
        if (data.log)
          setLogs((prevLogs) => [...prevLogs, parseLogEntry(data.log)]);
        if (data.fig) setFig(data.fig);
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
    [setIsConnected, setLogs, setFig, methods]
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
    <div className="py-lucid-dashboard">
      <Header
        errors={methods.formState.errors}
        steps={formSteps}
        setCurrentStep={setCurrentStep}
        reset={methods.reset}
      />

      <div className="dashboard-container">
        <main className="content">
          <FormProvider {...methods}>
            <form onSubmit={methods.handleSubmit(onSubmit as any)}>
              {formSteps.system.current && <ConfigSystem />}
              {formSteps.algorithm.current && <ConfigAlgorithm />}
              {formSteps.execution.current && <ConfigExecution />}
              <div className="flex flex-row-reverse">
                <button
                  type="submit"
                  className={
                    "btn btn-primary flex items-center justify-center w-50"
                  }
                >
                  <FaPaperPlane className="mr-2" />
                  Submit
                </button>
              </div>
            </form>
            <Result logs={logs} fig={fig} />
          </FormProvider>
        </main>

        <aside className="preview-panel">
          <h3>JSON Preview</h3>
          <JsonPreview formData={methods.watch()} />
        </aside>
      </div>

      <style>{`
        .py-lucid-dashboard {
          display: flex;
          flex-direction: column;
          height: 100vh;
        }
        .dashboard-container {
          display: flex;
          flex: 1;
        }
        .sidebar {
          width: 250px;
          background: #2c3e50;
          color: white;
          padding: 20px;
        }
        .sidebar ul {
          list-style: none;
          padding: 0;
        }
        .sidebar li {
          padding: 12px 15px;
          margin-bottom: 5px;
          border-radius: 4px;
          cursor: pointer;
        }
        .sidebar li.active {
          background: #3498db;
        }
        .content {
          flex: 1;
          padding: 20px;
          background: #f5f5f5;
          overflow-y: auto;
        }
        .preview-panel {
          width: 350px;
          padding: 20px;
          background: #f8f9fa;
          border-left: 1px solid #dee2e6;
        }
        .button-group {
          display: flex;
          justify-content: space-between;
          margin-top: 20px;
        }
      `}</style>
    </div>
  );
}
