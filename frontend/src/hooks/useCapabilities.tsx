import type { ServerCapabilities } from "@/types/types";
import {
  createContext,
  useContext,
  useState,
  useEffect,
  type ReactNode,
} from "react";
import { getCapabilities } from "@/utils/jslucid";

const defaultCapabilities: ServerCapabilities = {
  ALGLIB: false,
  GUROBI: false,
  HIGHS: true,
  MATPLOTLIB: false,
  PLOT: false,
  VERIFICATION: false,
  GUI: true,
  CUDA: false,
  OMP: false,
  PSOCPP: false,
  SOPLEX: false,
};

const CapabilitiesContext =
  createContext<ServerCapabilities>(defaultCapabilities);

export const useCapabilities = () => {
  const context = useContext(CapabilitiesContext);
  if (context === undefined) {
    throw new Error("Context must be used within a Provider");
  }
  return context;
};

type CapabilitiesProviderProps = {
  reset?: (capabilities: ServerCapabilities) => void;
  children: ReactNode;
};

export default function CapabilitiesProvider({
  children,
}: CapabilitiesProviderProps) {
  const [capabilities, setCapabilities] =
    useState<ServerCapabilities>(defaultCapabilities);

  useEffect(() => {
    // Replace with your API call to fetch capabilities
    async function fetchCapabilities() {
      try {
        const response = await fetch("/api/capabilities");
        const data: ServerCapabilities = await response.json();
        setCapabilities(data);
      } catch (error) {
        console.error("Failed to fetch capabilities:", error);
      }
    }
    async function fetchFromJslucid() {
      try {
        setCapabilities(await getCapabilities());
      } catch (error) {
        console.error("Failed to get capabilities from jslucid:", error);
      }
    }
    if (import.meta.env.VITE_JS_BUILD) {
      fetchFromJslucid();
    } else {
      fetchCapabilities();
    }
  }, []);

  return (
    <CapabilitiesContext.Provider value={capabilities}>
      {children}
    </CapabilitiesContext.Provider>
  );
}
