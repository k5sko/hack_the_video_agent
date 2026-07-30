import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath, URL } from "node:url";

// The shared response contract lives one level up in beeline/shared/.
// It is imported type-only, so nothing from it survives into the bundle —
// but the alias keeps a single source of truth for PathResult.
export default defineConfig({
  // Relative asset URLs. Vite defaults to absolute "/assets/...", which breaks
  // the moment the app is served from anything other than the domain root --
  // deployed under an /app/ prefix in S3, the bundle 404s and the page renders
  // blank with no error on screen. "./" works from any prefix, including root.
  base: "./",
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
