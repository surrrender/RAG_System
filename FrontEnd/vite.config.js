var _a;
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
var defaultApiTarget = "http://127.0.0.1:8000";
export default defineConfig({
    plugins: [react()],
    server: {
        proxy: {
            "/api": {
                target: (_a = process.env.VITE_API_PROXY_TARGET) !== null && _a !== void 0 ? _a : defaultApiTarget,
                changeOrigin: true,
                rewrite: function (path) { return path.replace(/^\/api/, ""); },
            },
        },
    },
    test: {
        environment: "jsdom",
        globals: true,
        setupFiles: "./vitest.setup.ts",
    },
});
