import { defineConfig } from "vite";
import { resolve } from "path";

export default defineConfig({
  root: ".",
  build: {
    outDir: "dist",
    rollupOptions: {
      input: {
        main: resolve(import.meta.dirname, "index.html"),
        "video-face-swap": resolve(
          import.meta.dirname,
          "tools/video-face-swap/index.html",
        ),
      },
    },
  },
});
