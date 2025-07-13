import type { LogEntry, ServerResponse } from "@/types/types";
import { defaultValues, emptyFigure } from "@/utils/constants";
import { parseLogEntry } from "@/utils/parseLog";
import { configurationSchema, type Configuration } from "@/utils/schema";
import { zodResolver } from "@hookform/resolvers/zod";
import "plotly.js-dist-min";
import { useCallback, useState } from "react";
import { useForm } from "react-hook-form";
import { FaPaperPlane, FaSpinner } from "react-icons/fa6";
import type { PlotParams } from "react-plotly.js";
import { Form } from "@/components/ui/form";
import { Button } from "@/components/ui/button";
import SetsInput from "@/components/SetsInput";
import Figure from "@/components/Figure";
import ConfigSystem from "@/components/ConfigSystem";

export default function InputSection() {
  const [fig, setFig] = useState<PlotParams>(emptyFigure);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [submitError, setSubmitError] = useState<string>("");
  const [loadingPreview, setLoadingPreview] = useState<boolean>(false);
  const [errorPreview, setErrorPreview] = useState<string>("");

  const methods = useForm({
    resolver: zodResolver(configurationSchema),
    defaultValues: defaultValues,
    mode: "onChange",
  });

  const onSubmit = useCallback(
    async (data: Configuration) => {
      if (isConnected) {
        console.warn("Already connected, ignoring new submission");
        return;
      }
      console.log("Submitting data:", data);
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
          methods.setError(error.cause as keyof Configuration, {
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

  const handlePreview = useCallback(async () => {
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
    setLoadingPreview(true);
    const response = await fetch("/api/preview-graph", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(methods.getValues()),
    });
    setLoadingPreview(false);
    if (!response.ok) {
      setFig({ data: [], layout: {} });
      const error: ServerResponse = await response.json();
      if (error.cause) {
        methods.setError(error.cause as keyof Configuration, {
          type: "value",
          message: error.error,
        });
      }
      setErrorPreview(error.error ?? "Unknown error");
      return;
    }
    const json = await response.json();
    try {
      setFig(json.fig ? JSON.parse(json.fig) : emptyFigure);
    } catch (e) {
      setFig(emptyFigure);
      console.error("Failed to parse figure data:", e);
    }
  }, [methods, setFig, setErrorPreview, setLoadingPreview]);

  console.log("Error state:", methods.formState.errors);

  return (
    <section className="flex-grow basis-0 mx-auto p-4">
      <Form {...methods}>
        <form onSubmit={methods.handleSubmit(onSubmit)}>
          <ConfigSystem
            onSubmit={handlePreview}
            loading={loadingPreview}
            error={errorPreview}
          />
          <div>
            <SetsInput name="X_bounds" label="X bounds" />

            <SetsInput name="X_init" label="X init" />

            <SetsInput name="X_unsafe" label="X unsafe" />

            <Figure data={fig.data} layout={fig.layout} />
          </div>
          <Button type="submit" disabled={isConnected}>
            {isConnected ? (
              <>
                <FaSpinner className="mr-2 animate-spin" />
                Computing
              </>
            ) : (
              <>
                <FaPaperPlane className="mr-2" />
                Submit
              </>
            )}
          </Button>
          <Button type="submit">Submit</Button>
        </form>
      </Form>
    </section>
  );
}
