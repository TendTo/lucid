import { Button } from "@/components/ui/button";
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
import type { Configuration } from "@/utils/schema";
import type { useForm } from "react-hook-form";
import { FaArrowDown } from "react-icons/fa6";
import Logo from "@/assets/logo.svg";
import JsonImportModal from "@/components/JsonImportModal";
import Examples from "@/components/Examples";

type FooterProps = {
  methods: ReturnType<typeof useForm<Configuration>>;
};

export default function Footer({ methods }: FooterProps) {
  const { reset } = methods;
  return (
    <footer className="bg-foreground text-center flex">
      <div className="shrink-0">
        <img alt="Lucid logo" src={Logo} className="size-8" />
      </div>
      <h1 className="text-white">Lucid</h1>
      <JsonImportModal reset={reset} />
      <Examples reset={reset} />
      <Drawer fixed={false}>
        <DrawerTrigger asChild>
          <Button variant="outline">Advanced</Button>
        </DrawerTrigger>
        <DrawerContent>
          <div className="mx-auto w-full">
            <DrawerHeader>
              <DrawerTitle>Move Goal</DrawerTitle>
              <DrawerDescription>
                Set your daily activity goal.
              </DrawerDescription>
            </DrawerHeader>
            <div className="p-4 pb-0">
              <div className="flex items-center justify-center space-x-2">
                <span className="sr-only">Decrease</span>
                <div className="flex-1 text-center">
                  <div className="text-7xl font-bold tracking-tighter">
                    GAOL
                  </div>
                  <div className="text-muted-foreground text-[0.70rem] uppercase">
                    Calories/day
                  </div>
                </div>
                <Button
                  variant="outline"
                  size="icon"
                  className="h-8 w-8 shrink-0 rounded-full"
                >
                  <span className="sr-only">Increase</span>
                </Button>
              </div>
              <div className="mt-3 h-[120px]">CHART?</div>
            </div>
            <DrawerFooter>
              <DrawerClose asChild>
                <Button variant="outline">
                  <FaArrowDown />
                </Button>
              </DrawerClose>
            </DrawerFooter>
          </div>
        </DrawerContent>
      </Drawer>
    </footer>
  );
}
