import { defineConfig } from "vite";
import { resolve } from "path";
import { existsSync, statSync, createReadStream } from "fs";

const localModelsDir = resolve(import.meta.dirname, "../../notes/models");

export default defineConfig({
  root: ".",
  server: {
    // Allow large model file responses (no body size limit)
    headers: { "Access-Control-Allow-Origin": "*" },
    hmr: false,
    ws: false,
    fs: {
      allow: [
        ".",
        ...(existsSync(localModelsDir) ? [localModelsDir] : []),
      ],
    },
  },
  plugins: [
    {
      name: "local-model-proxy",
      configureServer(server) {
        server.middlewares.use("/local-models/fastwan", (req, res, next) => {
          if (!req.url) return next();
          const filePath = resolve(localModelsDir, "fastwan/hf-repo", req.url.replace(/^\//, ""));
          if (!filePath.startsWith(localModelsDir)) return next();
          if (!existsSync(filePath)) return next();
          const stat = statSync(filePath);
          res.setHeader("Content-Length", stat.size);
          res.setHeader("Content-Type", "application/octet-stream");
          res.setHeader("Access-Control-Allow-Origin", "*");
          createReadStream(filePath).pipe(res);
        });
      },
    },
  ],
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
