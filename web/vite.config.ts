import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { viteStaticCopy } from "vite-plugin-static-copy";

// https://vite.dev/config/
export default defineConfig({
  base: "./",
  plugins: [
    react(),
    viteStaticCopy({
      targets: [
        // Ship full Python module set and data files for parity in-browser
        { src: "../*.py", dest: "py" },
        { src: "../*.json", dest: "py" },
        { src: "../geo/**/*", dest: "py/geo" },
        { src: "../countries/**/*", dest: "py/countries" },
        { src: "../trajectories/**/*", dest: "py/trajectories" },
      ],
    }),
  ],
})
