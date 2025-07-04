import type { LogEntry } from "@app/types/types";

export function parseLogEntry(entry?: string): LogEntry {
  if (!entry) throw new Error("Invalid log entry");

  const [timestamp, , type_str, , ...text] = entry.split("]");
  console.log("Parsing log entry:", entry);
  console.log("Parsing log entry:", entry.split("]"));

  const type = type_str.slice(2) as LogEntry["type"];

  return {
    timestamp: timestamp.slice(1), // Remove the leading '['
    type,
    text: text.join(" "),
  };
}
