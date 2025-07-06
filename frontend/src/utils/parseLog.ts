import type { LogEntry } from "@app/types/types";

export function parseLogEntry(entry?: string): LogEntry {
  if (!entry) throw new Error("Invalid log entry");
  const [date, time, type, , ...text] = entry.split(" ");
  const timestamp = date + " " + time;

  // Remove the leading '[' and trailing ']' from strings
  return {
    timestamp: timestamp.substring(1, timestamp.length - 1),
    type: type.substring(1, type.length - 1) as LogEntry["type"],
    text: text.join(" "),
  };
}
