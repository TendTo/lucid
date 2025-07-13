import Figure from "@/components/Figure";
import type { LogEntry } from "@/types/types";
import type { PlotParams } from "react-plotly.js";

type ResultProps = {
  logs: LogEntry[];
  fig: PlotParams;
};

export default function Result({ logs, fig }: ResultProps) {
  return (
    <div>
      <pre>
        <code>
          {logs.map((log, index) => (
            <span
              key={index}
              className={"log-entry"}
              title={`${log.timestamp} - ${log.type}`}
            >
              {log.timestamp} -{" "}
              <span className={`log-${log.type}`}>{log.type}</span>:{" "}
              <span>{log.text}</span>
            </span>
          ))}
        </code>
      </pre>
      <Figure {...fig} />
    </div>
  );
}
