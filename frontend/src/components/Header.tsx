import { Disclosure } from "@headlessui/react";
import { FaX, FaCheck } from "react-icons/fa6";
import Logo from "@/assets/logo.svg";
import type { FormStepName, FormSteps } from "@/types/types";
import Examples from "@/components/Examples";
import type { FieldValues, UseFormReset } from "react-hook-form";
import JSONImportModal from "@/components/JSONImportModal";

export type HeaderProps = {
  errors: object;
  steps: FormSteps;
  setCurrentStep: (step: FormStepName) => void;
  reset: UseFormReset<FieldValues>;
};

export default function Header({
  steps,
  setCurrentStep,
  errors,
  reset,
}: HeaderProps) {
  return (
    <header>
      <Disclosure as="nav" className="bg-gray-800">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="flex h-16 items-center justify-between">
            <div className="flex items-center">
              <div className="shrink-0">
                <img alt="Lucid logo" src={Logo} className="size-8" />
              </div>
              <h1 className="text-white">Lucid</h1>
              <div className="ml-10 flex items-center space-x-4">
                {Object.entries(steps).map(([key, item]) => (
                  <a
                    key={item.name}
                    href={item.href}
                    aria-current={item.current ? "page" : undefined}
                    className={
                      "rounded-md px-4 py-2 text-sm font-medium bg-gray-700 text-white hover:bg-gray-600 decoration-wavy underline-offset-4" +
                      (item.current ? " underline" : "")
                    }
                    onClick={(e) => {
                      e.preventDefault();
                      setCurrentStep(key as FormStepName);
                    }}
                  >
                    {item.name}
                    {item.error(errors) ? (
                      <FaX className="inline-block ml-1 text-red-500" />
                    ) : (
                      <FaCheck className="inline-block ml-1 text-green-500" />
                    )}
                  </a>
                ))}
                <JSONImportModal reset={reset} />
                <Examples reset={reset} />
              </div>
            </div>
          </div>
        </div>
      </Disclosure>
    </header>
  );
}
