export function onChangeNumber(
  onChange: (value: number | "") => void
): (e: React.ChangeEvent<HTMLInputElement>) => void {
  return (e) => onChange(e.target.value !== "" ? Number(e.target.value) : "");
}

export function handleDragOver(e: React.DragEvent<HTMLElement>) {
  e.stopPropagation();
  e.preventDefault();
  e.currentTarget.classList.add("drag-over");
}

export function handleDragLeave(e: React.DragEvent<HTMLElement>) {
  e.stopPropagation();
  e.preventDefault();
  e.currentTarget.classList.remove("drag-over");
}

export function parseNumberList(value: string) {
  const numbers = Array.from(value.matchAll(/-?\d+(\.\d+)?/g)).map((match) =>
    Number(match[0])
  );
  return numbers;
}

export function formatNumber(value: number | undefined) {
  if (value === undefined) return "N/A";
  return typeof value === "number" ? value.toFixed(6) : value;
}
