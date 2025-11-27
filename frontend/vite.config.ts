import { defineConfig, searchForWorkspaceRoot } from "vite";
import react from "@vitejs/plugin-react-swc";
import tailwindcss from "@tailwindcss/vite";
import tsconfigPaths from "vite-tsconfig-paths";

// https://vite.dev/config/
export default defineConfig({
  define: {
    __JSLUCID_PATH__: "'./jslucid.js'",
  },
  plugins: [react(), tailwindcss(), tsconfigPaths()],
  base: "./",
  server: {
    fs: {
      allow: [
        searchForWorkspaceRoot(process.cwd()),
        `path/to/bazel-out/k8-opt/bin/bindings/jslucid/jslucid.wasm`,
      ],
    },
    proxy: {
      "/api": {
        target: "http://localhost:3661",
        changeOrigin: true,
        secure: false,
      },
    },
  },
});
