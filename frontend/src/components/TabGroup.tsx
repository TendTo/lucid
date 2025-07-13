import {
  TabGroup as HTabGroup,
  Tab,
  TabList,
  TabPanel,
  TabPanels,
} from "@headlessui/react";

type TabGroupProps = {
  tabs: Record<string, React.ReactNode>;
};

export default function TabGroup({ tabs }: TabGroupProps) {
  return (
    <div className="w-full max-w-app px-2 py-16 sm:px-0">
      <HTabGroup>
        <TabList className="flex space-x-1 rounded-xl bg-blue-900/20 p-1">
          {Object.keys(tabs).map((name) => (
            <Tab
              key={name}
              className={({ selected }) =>
                "w-full rounded-lg py-2.5 text-sm font-medium leading-5 ring-white/60 ring-offset-2 ring-offset-blue-400 focus:outline-none focus:ring-2" +
                (selected
                  ? " bg-white text-blue-700 shadow"
                  : " text-blue-100 hover:bg-white/[0.12] hover:text-white")
              }
            >
              {name}
            </Tab>
          ))}
        </TabList>
        <TabPanels className="mt-2 flex flex-col rounded-lg border-2 border-dashed border-gray-300 p-4 focus:border-blue-300 focus:outline-none dark:border-gray-600">
          {Object.values(tabs).map((node, idx) => (
            <TabPanel
              key={idx}
              className={
                "rounded-xl bg-white p-3 ring-white/60 ring-offset-2 ring-offset-blue-400 focus:outline-none focus:ring-2"
              }
            >
              {node}
            </TabPanel>
          ))}
        </TabPanels>
      </HTabGroup>
    </div>
  );
}
