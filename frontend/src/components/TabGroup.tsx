import {
  TabGroup as HTabGroup,
  Tab,
  TabList,
  TabPanel,
  TabPanels,
} from "@headlessui/react";

type TabGroupProps = {
  tabs: Record<string, { content: React.ReactNode; onClick?: () => void }>;
  className?: string;
  selectedIndex?: number;
  setSelectedIndex?: (index: number) => void;
};

export default function TabGroup({
  tabs,
  className,
  selectedIndex,
  setSelectedIndex,
}: TabGroupProps) {
  return (
    <HTabGroup
      className={className}
      selectedIndex={selectedIndex}
      onChange={setSelectedIndex}
    >
      <TabList className="flex space-x-1 rounded-xl bg-muted-background p-1">
        {Object.entries(tabs).map(([name, { onClick }]) => (
          <Tab
            key={name}
            className={({ selected }) =>
              "w-full rounded-lg py-2.5 text-sm font-medium leading-5 ring-offset-2 [stroke-dasharray:10] ring-offset-chart-primary" +
              (selected
                ? " bg-background/[0.9] text-foreground shadow ring-2"
                : " text-foreground hover:bg-background/[0.9] hover:text-foreground/[0.7] outline-dotted")
            }
            onClick={onClick}
          >
            {name}
          </Tab>
        ))}
      </TabList>
      <TabPanels className="mt-2 flex flex-col rounded-lg border-2 border-dashed border-gray-300 p-4 focus:border-blue-300 focus:outline-none dark:border-gray-600">
        {Object.values(tabs).map(({ content }, idx) => (
          <TabPanel
            key={idx}
            className={
              "rounded-xl bg-background p-3 ring-white/60 ring-offset-2"
            }
          >
            {content}
          </TabPanel>
        ))}
      </TabPanels>
    </HTabGroup>
  );
}
