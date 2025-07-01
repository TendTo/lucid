import { Disclosure } from "@headlessui/react";
import { FaX, FaCheck, FaFileImport } from "react-icons/fa6";
import Logo from "@assets/logo.svg";
import type { FormStepName, FormSteps } from "@components/App";

export type HeaderProps = {
  errors: object;
  steps: FormSteps;
  setCurrentStep: (step: FormStepName) => void;
  setIsImportModalOpen: (isOpen: boolean) => void; // Optional for import modal
};

export default function Header({
  steps,
  setCurrentStep,
  errors,
  setIsImportModalOpen,
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
                <button
                  type="button"
                  onClick={() => setIsImportModalOpen(true)}
                  className="btn btn-secondary flex items-center justify-center"
                >
                  <FaFileImport className="inline-block mr-1" />
                  Import JSON
                </button>
              </div>
            </div>
          </div>
        </div>
      </Disclosure>
    </header>
  );
}
