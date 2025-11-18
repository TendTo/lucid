import Logo from "@/assets/logo.svg";
import JsonImportModal from "@/components/JsonImportModal";
import type { FormStepName, FormSteps } from "@/types/types";
import type { Configuration } from "@/utils/schema";
import { Disclosure } from "@headlessui/react";
import type { FieldErrors, useForm, UseFormReset } from "react-hook-form";
import {
  Drawer,
  DrawerClose,
  DrawerContent,
  DrawerDescription,
  DrawerFooter,
  DrawerHeader,
  DrawerTitle,
  DrawerTrigger,
} from "@/components/ui/drawer";
import { Button } from "@/components/ui/button";
import { FaArrowDown, FaShare } from "react-icons/fa6";
import { Form } from "@/components/ui/form";
import { useMemo } from "react";
import ConfigAdvanced from "@/components/ConfigAdvanced";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import JsonPreview from "@/components/JsonPreview";
import { ScrollArea } from "@/components/ui/scroll-area";

export type HeaderProps = {
  methods: ReturnType<typeof useForm<Configuration>>;
  errors: object;
  steps: FormSteps;
  setCurrentStep: (step: FormStepName) => void;
  reset: UseFormReset<Configuration>;
};

export function advancedError(errors: FieldErrors<Configuration>): boolean {
  return Boolean(
    errors.system_dynamics ||
      errors.X_bounds ||
      errors.X_init ||
      errors.X_unsafe ||
      errors.x_samples ||
      errors.xp_samples
  );
}

export default function Header({ reset, methods }: HeaderProps) {
  const { formState } = methods;

  const errors = useMemo(() => {
    return [
      "verbose",
      "gamma",
      "C_coeff",
      "lambda",
      "num_samples",
      "time_horizon",
      "sigma_f",
      "sigma_l",
      "num_frequencies",
      "oversample_factor",
      "lattice_resolution",
      "noise_scale",
      "estimator",
      "kernel",
      "feature_map",
      "optimiser",
    ].some((key) => Object.hasOwn(formState.errors, key));
  }, [formState]);

  return (
    <header className="sticky top-0 h-16 z-50">
      <Disclosure as="nav" className="bg-gray-800">
        <div className="flex items-center justify-between h-16 px-6">
          <div className="flex items-center">
            <img alt="Lucid logo" src={Logo} className="size-8" />
            <h1 className="text-white">Lucid</h1>
          </div>
          <Drawer fixed={false}>
            <DrawerTrigger asChild>
              <Button
                variant="outline"
                className={errors ? "text-red-500 border-red-500" : ""}
              >
                Advanced
              </Button>
            </DrawerTrigger>
            <DrawerContent className="px-4">
              <DrawerHeader>
                <DrawerTitle>Advanced configuration</DrawerTitle>
                <DrawerDescription>
                  Customize the scenario further with advanced settings.
                </DrawerDescription>
              </DrawerHeader>
              <ScrollArea className="mx-auto w-full p-4 h-[50vh]">
                <Form {...methods}>
                  <ConfigAdvanced />
                </Form>
              </ScrollArea>
              <DrawerFooter>
                <DrawerClose asChild>
                  <Button variant="outline">
                    <FaArrowDown />
                  </Button>
                </DrawerClose>
              </DrawerFooter>
            </DrawerContent>
          </Drawer>
          <div className="ml-10 flex items-center space-x-4">
            <JsonImportModal reset={reset} />
            <Sheet>
              <SheetTrigger asChild>
                <Button variant="outline">
                  <FaShare />
                  Export
                </Button>
              </SheetTrigger>
              <SheetContent className="min-w-md">
                <SheetHeader>
                  <SheetTitle>Export configuration</SheetTitle>
                  <SheetDescription>
                    Export the current configuration as a JSON file.
                  </SheetDescription>
                </SheetHeader>
                <JsonPreview formData={methods.watch()} />
              </SheetContent>
            </Sheet>
          </div>
        </div>
      </Disclosure>
    </header>
  );
}
