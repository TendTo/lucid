import App from "@/layout/App.tsx";
import "@/styles/main.css";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import CapabilitiesProvider from "./hooks/useCapabilities";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <CapabilitiesProvider>
      <App />
    </CapabilitiesProvider>
  </StrictMode>
);
