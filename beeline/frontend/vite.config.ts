import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath, URL } from "node:url";

// The shared response contract lives one level up in beeline/shared/.
// It is imported type-only, so nothing from it survives into the bundle —
// but the alias keeps a single source of truth for PathResult.
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@shared": fileURLToPath(new URL("../shared", import.meta.url)),
    },
  },
  server: {
    port: 5173,
    fs: { allow: [".."] },
  },
});
