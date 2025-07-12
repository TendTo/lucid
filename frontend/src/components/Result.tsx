import { DangerousElement } from "@/components/DangerousElement";
import type { LogEntry } from "@app/types/types";

type ResultProps = {
  logs: LogEntry[];
  fig: string;
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
      <DangerousElement markup={fig} />
    </div>
  );
}
