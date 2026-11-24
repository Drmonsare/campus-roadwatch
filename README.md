# Campus RoadWatch

**Design of an AI-Based Pothole Detection System** — Group `CST_05`

- **Developed by:** Satvik Kalra (`00918012723`) & Pranjal Tripathi (`01418012723`)
- **Project Guide:** Mr. Akash Bhardwaj

## Overview
Campus RoadWatch is an offline-capable, browser-based road surface inspection workstation. It runs a bundled `YOLOv8s` ONNX pothole detector locally via WebAssembly, computes affected road area and severity grades, lets reviewers confirm or dismiss individual detections, and exports structured CSV reports and annotated PNGs.

## Run Locally

### Windows (One-Click)
1. Extract the ZIP folder.
2. Double-click **`Start RoadWatch.cmd`** inside `campus-roadwatch/`.

### Terminal (Windows / macOS / Linux)
```bash
node scripts/build.mjs
node scripts/serve.mjs
```
Open the URL printed in the terminal (default `http://127.0.0.1:4173`).

## Deploy to GitHub & Vercel
1. Push the contents of the `campus-roadwatch/` folder to a GitHub repository (`dist/` is ignored via `.gitignore`).
2. Import the repository in [Vercel](https://vercel.com) — `vercel.json` is already configured with `node scripts/build.mjs`, `dist/` output, and proper `.wasm` / `.onnx` headers. No environment variables or backend servers are needed.

## Features
- **Offline ONNX Inference:** Bundled `YOLOv8s` model runs in a Web Worker with zero external API keys.
- **Custom `.onnx` Model Loader:** Load any single-class or multi-class YOLOv8/YOLO11 `.onnx` model directly from the toolbar.
- **Live Controls:** Adjust Confidence Threshold and NMS Overlap sliders in real time without re-running inference.
- **Surface Metrics:** Calculates affected road area (%), bounding box dimensions, severity level (`Low` / `Medium` / `High`), overall road condition, and inference latency (`ms`).
- **Exports & History:** Download annotated PNG images, export formula-safe CSV reports, and revisit saved inspections in IndexedDB.
