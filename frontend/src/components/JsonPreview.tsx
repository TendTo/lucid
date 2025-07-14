import { useCallback, useState } from "react";
import { FaCheck, FaCopy, FaDownload } from "react-icons/fa6";
import { Button } from "./ui/button";

type JsonPreviewProps = {
  formData: object;
};

export default function JsonPreview({ formData }: JsonPreviewProps) {
  const [copied, setCopied] = useState(false);
  const formattedJson = JSON.stringify(formData, null, 2);

  const copyToClipboard = useCallback(() => {
    navigator.clipboard.writeText(formattedJson);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [formattedJson]);

  const downloadJson = useCallback(() => {
    const blob = new Blob([formattedJson], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "config.json";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [formattedJson]);

  return (
    <div className="h-full flex flex-col">
      <pre className="relative rounded overflow-auto flex-1 text-xs p-4 font-mono bg-gray-100">
        <Button
          className="absolute right-16 top-4"
          onClick={copyToClipboard}
          title="Copy to clipboard"
        >
          {copied ? <FaCheck /> : <FaCopy />}
        </Button>
        <Button
          className="absolute right-4 top-4"
          onClick={downloadJson}
          title="Download"
        >
          <FaDownload />
        </Button>
        {formattedJson}
      </pre>
    </div>
  );
}
