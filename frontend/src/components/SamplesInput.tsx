import { ErrorMessage } from "@hookform/error-message";
import { useFormContext } from "react-hook-form";
import { useCallback, useRef } from "react";
import { FaUpload } from "react-icons/fa6";
import { parseCSVData } from "@app/utils/csvParser";

type SamplesInputProps = {
  name: string;
  label: string;
};

function handleDragOver(e: React.DragEvent<HTMLElement>) {
  e.preventDefault();
  e.currentTarget.classList.add("drag-over");
}

function handleDragLeave(e: React.DragEvent<HTMLElement>) {
  e.preventDefault();
  e.currentTarget.classList.remove("drag-over");
}

export default function SamplesInput({ name, label }: SamplesInputProps) {
  const { formState, register, setValue } = useFormContext();
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
    <div className="form-group">
      <div className="flex items-center justify-between mb-2">
        <label htmlFor={name} className="block font-bold">
          {label}
        </label>
        <div>
          <label
            htmlFor={`${name}-file-upload`}
            className="btn btn-secondary flex items-center justify-center cursor-pointer"
          >
            <FaUpload className="inline-block mr-1 size-3" />
            Upload CSV
          </label>
          <input
            className="hidden"
            type="file"
            id={`${name}-file-upload`}
            name={`${name}-file-upload`}
            accept="text/csv"
            onChange={handleFileInputChange}
          />
        </div>
      </div>

      <textarea
        {...rest}
        ref={(e) => {
          ref(e);
          inputRef.current = e;
        }}
        id={name}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className="form-control w-full h-32"
        placeholder={`1.2,2.4,3.1
4.1,5.2,6.2

Or drag and drop a CSV file here
`}
      />

      <ErrorMessage
        errors={formState.errors}
        name={name}
        render={({ message }) => (
          <small className="text-red-500 block mb-1">{message}</small>
        )}
      />
    </div>
  );
}
