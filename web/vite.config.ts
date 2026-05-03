import { defineConfig, type Plugin } from "vite";
import { resolve } from "path";
import { existsSync, statSync, createReadStream } from "fs";
import { createHash } from "crypto";

const localModelsDir = resolve(import.meta.dirname, "../../notes/models");

// CSP injected via <meta http-equiv> on every page, dev and prod. We
// can't ship 'unsafe-inline' for script-src, so the plugin extracts each
// page's inline <script> blocks at transform time and emits sha256
// hashes for them. Self-correcting if inline contents change.
function injectCsp(): Plugin {
  return {
    name: "inject-csp",
    transformIndexHtml: {
      order: "post",
      handler(html) {
        const inlineRe = /<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/gi;
        const hashes: string[] = [];
        for (const m of html.matchAll(inlineRe)) {
          const digest = createHash("sha256").update(m[1], "utf8").digest("base64");
          hashes.push(`'sha256-${digest}'`);
        }
        const scriptSrc = [
          "'self'",
          "'wasm-unsafe-eval'",
          "'unsafe-eval'",
          "https://cdn.jsdelivr.net/npm/@mediapipe/",
          ...hashes,
        ].join(" ");
        const csp = [
          "default-src 'none'",
          `script-src ${scriptSrc}`,
          "worker-src 'self' blob:",
          "style-src 'self' 'unsafe-inline'",
          "img-src 'self' blob: data:",
          "media-src blob:",
          "connect-src 'self' blob: https://huggingface.co https://*.huggingface.co https://*.hf.co https://storage.googleapis.com/mediapipe-models/ https://cdn.jsdelivr.net/npm/@mediapipe/",
          "base-uri 'none'",
          "form-action 'none'",
        ].join("; ");
        const tag = `    <meta http-equiv="Content-Security-Policy" content="${csp}" />`;
        return html.replace(/(<meta charset=[^>]*>)/i, (m) => `${m}\n${tag}`);
      },
    },
  };
}

// Vite's dep pre-bundler injects `import { injectQuery } from "/@vite/client"`
// into prebundled deps that resolve URLs at runtime (e.g. onnxruntime-web
// loading its wasm/worker). The real client.mjs opens a WebSocket on load
// even with server.hmr=false — see vitejs/vite#13994, #18489. We serve a
// minimal stub instead: just enough exports for prebundled imports to
// resolve, no WS, no HMR machinery.
function stubViteClient(): Plugin {
  const stub = [
    "// /@vite/client stub - HMR + WebSocket disabled (see vite.config.ts)",
    "export function injectQuery(url, queryToInject) {",
    "  if (url[0] !== '/' && !/^[a-z]+:/i.test(url)) return url;",
    "  const u = new URL(url.startsWith('//') ? location.protocol + url : url, 'http://_');",
    "  return u.pathname + '?' + queryToInject + (u.search ? '&' + u.search.slice(1) : '') + (u.hash || '');",
    "}",
    "const noop = () => {};",
    "const hot = { accept: noop, acceptExports: noop, dispose: noop, prune: noop, decline: noop, invalidate: noop, on: noop, off: noop, send: noop, data: {} };",
    "export function createHotContext() { return hot; }",
    "export function updateStyle() {}",
    "export function removeStyle() {}",
    "",
  ].join("\n");
  return {
    name: "stub-vite-client",
    apply: "serve",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = req.url ?? "";
        if (url === "/@vite/client" || url.startsWith("/@vite/client?")) {
          res.setHeader("Content-Type", "text/javascript");
          res.setHeader("Cache-Control", "no-cache");
          res.end(stub);
          return;
        }
        next();
      });
    },
  };
}

export default defineConfig({
  root: ".",
  server: {
    // Allow large model file responses (no body size limit)
    headers: { "Access-Control-Allow-Origin": "*" },
    fs: {
      allow: [
        ".",
        ...(existsSync(localModelsDir) ? [localModelsDir] : []),
      ],
    },
  },
  plugins: [
    stubViteClient(),
    injectCsp(),
    {
      name: "local-model-proxy",
      configureServer(server) {
        const mount = (model: string) => {
          server.middlewares.use(`/local-models/${model}`, (req, res, next) => {
            if (!req.url) return next();
            const filePath = resolve(localModelsDir, `${model}/hf-repo`, req.url.replace(/^\//, ""));
            if (!filePath.startsWith(localModelsDir)) return next();
            if (!existsSync(filePath)) return next();
            const stat = statSync(filePath);
            res.setHeader("Content-Length", stat.size);
            res.setHeader("Content-Type", "application/octet-stream");
            res.setHeader("Access-Control-Allow-Origin", "*");
            createReadStream(filePath).pipe(res);
          });
        };
        mount("fastwan");
        mount("ltx");
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
