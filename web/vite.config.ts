import { defineConfig } from "vite";
import { resolve } from "path";

export default defineConfig({
  root: ".",
  server: {
    // Allow large model file responses (no body size limit)
    headers: { "Access-Control-Allow-Origin": "*" },
  },
  build: {
    outDir: "dist",
    rollupOptions: {
      input: {
        main: resolve(import.meta.dirname, "index.html"),
        "video-face-swap": resolve(
          import.meta.dirname,
          "tools/video-face-swap/index.html",
        ),
        "video-gen": resolve(import.meta.dirname, "tools/video-gen/index.html"),
        "image-gen": resolve(import.meta.dirname, "tools/image-gen/index.html"),
        "model-smoke": resolve(
          import.meta.dirname,
          "tools/model-smoke/index.html",
        ),
      },
    },
  },
});
