import { cp, mkdir, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));
const output = join(root, 'dist');
const assets = [
  'index.html',
  'favicon.svg',
  'favicon.png',
  'app.css',
  'app.js',
  'vision.js',
  'vision-worker.js',
  'storage.js',
  'demo',
  'vendor',
  'models',
];

await rm(output, { recursive: true, force: true });
await mkdir(output, { recursive: true });
for (const asset of assets) {
  await cp(join(root, asset), join(output, asset), { recursive: true });
}
console.log('Build complete: dist/ is ready.');
