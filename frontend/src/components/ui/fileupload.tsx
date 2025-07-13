import * as React from "react";
import { type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";
import { buttonVariants } from "./button";
import { FaUpload } from "react-icons/fa6";

function FileUpload({
  className,
  variant,
  size,
  label,
  ...props
}: React.ComponentProps<"input"> &
  VariantProps<typeof buttonVariants> & {
    label: string;
  }) {
  return (
    <>
      <label
        htmlFor={props.id}
        className={cn(buttonVariants({ variant, size, className }))}
      >
        <FaUpload className="inline-block mr-1 size-3" />
        {label}
      </label>
      <input className="hidden" type="file" {...props} />
    </>
  );
}

export { FileUpload };
