import { defineConfig } from "vite";
import { resolve } from "path";
import { existsSync, statSync, createReadStream } from "fs";

// Serve local model files under /local-models/ during dev.
// Drop model files into ../../notes/models/zimage-local/ mirroring the HF
// repo layout (e.g. onnx/transformer_model_q4f16.onnx). Works on --host
// so mobile on LAN can reach them too.
const LOCAL_MODELS_DIR = resolve(import.meta.dirname, "../../notes/models/zimage/hf-repo");

export default defineConfig({
  root: ".",
  server: {
    // Allow large model file responses (no body size limit)
    headers: { "Access-Control-Allow-Origin": "*" },
  },
  plugins: [
    {
      name: "local-model-server",
      configureServer(server) {
        server.middlewares.use("/local-models", (req, res, next) => {
          const filePath = resolve(LOCAL_MODELS_DIR, (req.url ?? "/").slice(1));
          // Don't allow path traversal outside the models dir
          if (!filePath.startsWith(LOCAL_MODELS_DIR)) {
            res.statusCode = 403;
            res.end("Forbidden");
            return;
          }
          if (!existsSync(filePath) || statSync(filePath).isDirectory()) {
            res.statusCode = 404;
            res.end("Not found");
            return;
          }
          const stat = statSync(filePath);
          res.setHeader("Content-Length", stat.size);
          res.setHeader("Content-Type", "application/octet-stream");
          res.setHeader("Access-Control-Allow-Origin", "*");
          // Bump highWaterMark from the 64 KB default to 4 MB so LAN
          // delivery to the phone isn't syscall-bound. On loopback 64 KB
          // is fine, but over WiFi the many small pipeline steps leave
          // a lot of bandwidth unused.
          createReadStream(filePath, { highWaterMark: 4 * 1024 * 1024 }).pipe(res);
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
