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
