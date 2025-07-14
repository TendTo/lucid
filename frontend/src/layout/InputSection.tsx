import { useForm } from "react-hook-form";
import { FaPaperPlane, FaSpinner } from "react-icons/fa6";
import { Form } from "@/components/ui/form";
import { Button } from "@/components/ui/button";
import ConfigSystemDynamics from "@/components/ConfigSystemDynamics";
import ConfigSafetySpecification from "@/components/ConfigSafetySpecification";
import type { Configuration } from "@/utils/schema";
import { ScrollArea } from "@/components/ui/scroll-area";

type InputSectionProps = {
  methods: ReturnType<typeof useForm<Configuration>>;
  handleSubmit: (event: React.FormEvent<HTMLFormElement>) => Promise<void>;
  handlePreview: () => Promise<void>;
  submitLoading: boolean;
  submitError: string;
  previewError: string;
  previewLoading: boolean;
};

export default function InputSection({
  methods,
  handleSubmit,
  handlePreview,
  submitError,
  submitLoading,
  previewError,
  previewLoading,
}: InputSectionProps) {
  return (
    <section className="flex-grow basis-0">
      <ScrollArea className="h-full p-8">
        <Form {...methods}>
          <form id="form" onSubmit={handleSubmit}>
            <ConfigSystemDynamics
              onSubmit={handlePreview}
              loading={previewLoading}
              error={previewError}
              disabled={submitLoading || previewLoading}
            />
            <ConfigSafetySpecification />
            <div className="w-full flex justify-end mt-4 items-center gap-2">
              <small className="text-red-500 mt-1">{submitError}</small>
              <Button type="submit" disabled={submitLoading || previewLoading}>
                {submitLoading ? (
                  <>
                    <FaSpinner className="mr-2 animate-spin" />
                    Computing
                  </>
                ) : (
                  <>
                    <FaPaperPlane className="mr-2" />
                    Submit
                  </>
                )}
              </Button>
            </div>
          </form>
        </Form>
      </ScrollArea>
    </section>
  );
}
