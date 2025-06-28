import { useState } from "react";
import { FaCopy, FaCheck } from "react-icons/fa6";

type JsonPreviewProps = {
  formData: object;
};

export default function JsonPreview({ formData }: JsonPreviewProps) {
  const [copied, setCopied] = useState(false);
  const formattedJson = JSON.stringify(formData, null, 2);

  const copyToClipboard = () => {
    navigator.clipboard.writeText(formattedJson);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="h-full flex flex-col">
      <div className="flex mb-2 justify-between items-center">
        <h4>Current Configuration (JSON)</h4>
        <button
          onClick={copyToClipboard}
          className="bg-gray-800 p-2 text-white rounded cursor-pointer hover:bg-gray-700"
          title="Copy to clipboard"
        >
          {copied ? (
            <FaCheck className="size-4" />
          ) : (
            <FaCopy className="size-4" />
          )}
        </button>
      </div>

      <pre className="rounded overflow-auto flex-1 text-xs p-4 font-mono bg-gray-100">
        {formattedJson}
      </pre>
    </div>
  );
}
