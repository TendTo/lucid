import { defaultValues, examples } from "@/utils/constants";
import { configurationSchema, type Configuration } from "@/utils/schema";
import { useCallback, useEffect, useState } from "react";
import { type UseFormReset } from "react-hook-form";
import { FaFileImport } from "react-icons/fa6";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "./ui/dialog";
import { Button, buttonVariants } from "./ui/button";
import { Textarea } from "./ui/textarea";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { FileUpload } from "./ui/fileupload";
import { handleDragLeave, handleDragOver } from "@/utils/utils";

interface JsonImportModalProps {
  reset: UseFormReset<Configuration>;
}

export default function JsonImportModal({ reset }: JsonImportModalProps) {
  const [jsonText, setJsonText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    // Load the last saved JSON from local storage if available
    const savedJson = localStorage.getItem("lucid-import-json");
    if (savedJson) {
      setJsonText(savedJson);
    }
  }, []);

  const handleImport = () => {
    try {
      // Parse the JSON
      const parsedJson = { ...defaultValues, ...JSON.parse(jsonText) };

      // Validate against schema
      const result = configurationSchema.safeParse(parsedJson);

      if (!result.success) {
        setError("Invalid JSON format: " + result.error.message);
        return;
      }

      parsedJson.dimension =
        parsedJson.dimension || parsedJson.X_bounds[0].RectSet[0].length || 1;

      // Store the current JSON text in the browser's local storage
      localStorage.setItem("lucid-import-json", jsonText);
      // Reset form with new values
      reset(parsedJson);
      setError(null);
      setOpen(false);
    } catch (e) {
      setError(
        "Invalid JSON: " + (e instanceof Error ? e.message : "Unknown error")
      );
    }
  };

  const handleFileUpload = useCallback(
    (file: File) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const text = e.target?.result as string;
        setJsonText(text);
      };
      reader.readAsText(file);
    },
    [setJsonText]
  );

  const handleFileInputChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (file) {
        handleFileUpload(file);
      }
      event.target.value = ""; // Reset input value to allow re-uploading the same file
    },
    [handleFileUpload]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.stopPropagation();
      e.preventDefault();
      e.currentTarget.classList.remove("drag-over");

      const files = Array.from(e.dataTransfer.files);
      const csvFile = files.find(
        (file) =>
          file.type === "text/csv" ||
          file.name.endsWith(".csv") ||
          file.name.endsWith(".txt")
      );

      if (csvFile) {
        handleFileUpload(csvFile);
      }
    },
    [handleFileUpload]
  );

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger className={buttonVariants({ variant: "secondary" })}>
        <FaFileImport className="inline-block mr-1" />
        Import
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Import JSON configuration</DialogTitle>
          <DialogDescription>
            Use an existing JSON configuration to populate the form or load one
            of the examples.
          </DialogDescription>
        </DialogHeader>
        <Select
          onValueChange={(v) =>
            setJsonText(JSON.stringify(examples[v].config, null, 2))
          }
          name="example"
        >
          <SelectTrigger className="w-full">
            <SelectValue placeholder="Import an example" />
          </SelectTrigger>
          <SelectContent>
            <SelectGroup>
              <SelectLabel>Import an example</SelectLabel>
              {Object.entries(examples).map(([key, value]) => (
                <SelectItem key={key} value={key}>
                  {value.name}
                </SelectItem>
              ))}
            </SelectGroup>
          </SelectContent>
        </Select>
        <Textarea
          aria-invalid={!!error}
          className="h-64 border font-mono text-sm"
          value={jsonText}
          onChange={(e) => setJsonText(e.target.value)}
          placeholder={`{
  "verbose": 3,
  "seed": -1, 
  ...
}`}
          autoFocus
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        />
        <small className="text-red-500">{error}</small>
        <DialogFooter>
          <FileUpload
            label="Upload JSON"
            name="json-file-upload"
            accept=".json"
            id="json-file-upload"
            onChange={handleFileInputChange}
          />
          <Button onClick={handleImport}>
            <FaFileImport className="mr-2" />
            Import
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
