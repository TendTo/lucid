import { useEffect, useState } from "react";
import { useFormContext } from "react-hook-form";
import { jsonSchema } from "@utils/schema";
import { FaX, FaFileImport } from "react-icons/fa6";

interface JSONImportModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function JSONImportModal({
  isOpen,
  onClose,
}: JSONImportModalProps) {
  const [jsonText, setJsonText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const { reset } = useFormContext();

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
      const parsedJson = JSON.parse(jsonText);

      // Validate against schema
      const result = jsonSchema.safeParse(parsedJson);

      if (!result.success) {
        setError("Invalid JSON format: " + result.error.message);
        return;
      }

      // Store the current JSON text in the browser's local storage
      localStorage.setItem("lucid-import-json", jsonText);
      // Reset form with new values
      reset(parsedJson);
      setError(null);
      onClose();
    } catch (e) {
      setError(
        "Invalid JSON: " + (e instanceof Error ? e.message : "Unknown error")
      );
    }
  };

  return isOpen ? (
    <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center">
      <div className="bg-white rounded-lg p-6 max-w-2xl w-full max-h-[90vh] flex flex-col">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold">Import Configuration JSON</h2>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700"
          >
            <FaX className="size-5" />
          </button>
        </div>

        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
            <p>{error}</p>
          </div>
        )}

        <div className="mb-4 flex-grow">
          <label className="block mb-2 font-medium">
            Paste your JSON configuration below:
          </label>
          <textarea
            className="w-full h-64 border border-gray-300 rounded p-2 font-mono text-sm"
            value={jsonText}
            onChange={(e) => setJsonText(e.target.value)}
            placeholder='{"verbose": 3, "seed": -1, ...}'
            autoFocus
          />
        </div>

        <div className="flex justify-end space-x-2">
          <button
            type="button"
            onClick={onClose}
            className="btn btn-secondary"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleImport}
            className="btn btn-primary flex items-center"
          >
            <FaFileImport className="mr-2" />
            Import
          </button>
        </div>
      </div>
    </div>
  ) : null;
}
