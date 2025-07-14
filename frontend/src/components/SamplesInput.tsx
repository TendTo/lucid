import { parseCSVData } from "@/utils/csvParser";
import { useCallback, useRef } from "react";
import { useFormContext } from "react-hook-form";
import { Textarea } from "./ui/textarea";
import {
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "./ui/form";
import { FileUpload } from "./ui/fileupload";
import { handleDragLeave, handleDragOver } from "@/utils/utils";

type SamplesInputProps = {
  name: string;
  label: string;
};

export default function SamplesInput({ name, label }: SamplesInputProps) {
  const { register, setValue, control } = useFormContext();
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const { ref, ...rest } = register(name, {
    setValueAs: parseCSVData,
  });

  const handleFileUpload = useCallback(
    (file: File) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        if (!inputRef.current) return;
        const text = e.target?.result as string;
        setValue(name, parseCSVData(text));
        inputRef.current.value = text;
      };
      reader.readAsText(file);
    },
    [name, setValue]
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
    <FormField
      control={control}
      name={name}
      render={() => (
        <FormItem>
          <div className="flex items-center justify-between w-full">
            <FormLabel htmlFor={name}>{label}</FormLabel>
            <FormMessage />
            <FileUpload
              id={`${name}-file-upload`}
              name={`${name}-file-upload`}
              accept=".csv, .txt"
              onChange={handleFileInputChange}
              label="Upload CSV"
            />
          </div>
          <FormControl>
            <Textarea
              {...rest}
              ref={(e) => {
                ref(e);
                inputRef.current = e;
              }}
              id={name}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              placeholder={`1,2,3
4,5,6

Or drag and drop a CSV file here`}
            />
          </FormControl>
        </FormItem>
      )}
    />
  );
}
