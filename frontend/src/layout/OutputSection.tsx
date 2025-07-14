import Figure from "@/components/Figure";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { useDimensions } from "@/hooks/useDimension";
import type { LogEntry, SuccessResponseData } from "@/types/types";
import { formatNumber } from "@/utils/utils";
import { useRef } from "react";
import { FaCheck, FaX } from "react-icons/fa6";
import type { PlotParams } from "react-plotly.js";

type OutputSectionProps = {
  fig: PlotParams;
  logs: LogEntry[];
  loading: boolean;
  successData?: SuccessResponseData | null;
};

export default function OutputSection({
  fig,
  logs,
  loading,
  successData,
}: OutputSectionProps) {
  const figContainerRef = useRef<HTMLDivElement>(null);
  const dimensions = useDimensions(figContainerRef);
  fig.layout.width = dimensions.width;
  fig.layout.height = dimensions.height;
  return (
    <section className="flex-grow basis-0 mx-auto p-4 flex flex-col items-center relative">
      <div ref={figContainerRef} className="w-full flex-grow-2 basis-0">
        {fig.data.length > 0 && <Figure {...fig} />}
        {loading && <Skeleton className="h-full w-full rounded" />}
      </div>
      <Accordion
        type="single"
        collapsible
        defaultValue="results"
        className="w-full flex-grow-1 basis-0"
      >
        <AccordionItem value="results">
          <AccordionTrigger>
            <div className="w-full flex items-baseline gap-2">
              <h3 className="text-lg font-semibold mb-2">Results</h3>
              {successData ? (
                successData.success ? (
                  <FaCheck className="text-green-500" />
                ) : (
                  <FaX className="text-red-500" />
                )
              ) : null}
            </div>
          </AccordionTrigger>
          <AccordionContent>
            {successData && successData.success && (
              <div className="grid grid-cols-2 gap-4">
                <Label htmlFor="resultObj_val" className="font-bold">
                  Safety probability
                </Label>
                <Input
                  id="resultObj_val"
                  value={`${formatNumber(
                    1 - (successData.obj_val ?? 1) * 100
                  )}%`}
                  readOnly
                />
                <Label htmlFor="resultEta" className="font-bold">
                  Eta (η)
                </Label>
                <Input
                  id="resultEta"
                  value={formatNumber(successData.eta)}
                  readOnly
                />
                <Label htmlFor="resultC" className="font-bold">
                  c
                </Label>
                <Input
                  id="resultC"
                  value={formatNumber(successData.c)}
                  readOnly
                />
                <Label htmlFor="resultNorm" className="font-bold">
                  Norm
                </Label>
                <Input
                  id="resultNorm"
                  value={formatNumber(successData.norm)}
                  readOnly
                />
                <Label className="font-bold">Verified</Label>
                {successData.verified ? (
                  <FaCheck className="text-green-500" />
                ) : (
                  <FaX className="text-red-500" />
                )}
              </div>
            )}
          </AccordionContent>
        </AccordionItem>
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

      {/* Success Card - floating at bottom right */}
      {/* {successData && <SuccessCard data={successData} />} */}
    </section>
  );
}
