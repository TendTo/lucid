import Figure from "@/components/Figure";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { useDimensions } from "@/hooks/useDimension";
import type { LogEntry } from "@/types/types";
import { useRef } from "react";
import type { PlotParams } from "react-plotly.js";

type OutputSectionProps = {
  fig: PlotParams;
  logs: LogEntry[];
  loading: boolean;
};

export default function OutputSection({
  fig,
  logs,
  loading,
}: OutputSectionProps) {
  const figContainerRef = useRef<HTMLDivElement>(null);
  const dimensions = useDimensions(figContainerRef);
  fig.layout.width = dimensions.width;
  fig.layout.height = dimensions.height;
  return (
    <section className="flex-grow basis-0 mx-auto p-4 flex flex-col items-center">
      <div ref={figContainerRef} className="w-full flex-grow-2 basis-0">
        {fig.data.length > 0 && <Figure {...fig} />}
        {loading && <Skeleton className="h-full w-full rounded" />}
      </div>
      <Accordion
        type="single"
        collapsible
        defaultValue="logs"
        className="w-full flex-grow-1 basis-0"
      >
        <AccordionItem value="logs">
          <AccordionTrigger>
            <div className="w-full">
              <h3 className="text-lg font-semibold mb-2">Logs</h3>
            </div>
          </AccordionTrigger>
          <AccordionContent>
            <ScrollArea className="h-64">
              <code>
                <ul className="list-none px-2">
                  {logs.map((log, index) => (
                    <li key={index} className="mb-1">
                      <span
                        className="font-mono text-sm"
                        title={`${log.timestamp} - ${log.type}`}
                      >
                        {log.timestamp} -{" "}
                        <span className={`log-${log.type}`}>{log.type}</span>:{" "}
                        <span>{log.text}</span>
                      </span>
                    </li>
                  ))}
                </ul>
              </code>
            </ScrollArea>
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </section>
  );
}
