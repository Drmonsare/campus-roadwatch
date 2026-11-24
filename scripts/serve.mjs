import { createReadStream } from 'node:fs';
import { realpath, stat } from 'node:fs/promises';
import { createServer } from 'node:http';
import { extname, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';

const root = fileURLToPath(new URL('../dist/', import.meta.url));
await stat(resolve(root, 'index.html')).catch(() => {
  throw new Error('Build the app first with "npm run build" (or "node scripts/build.mjs").');
});
const base = await realpath(root);

const types = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.wasm': 'application/wasm',
  '.onnx': 'application/octet-stream',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png': 'image/png',
  '.webp': 'image/webp',
  '.json': 'application/json',
  '.svg': 'image/svg+xml',
};

const server = createServer(async (req, res) => {
  try {
    if (!['GET', 'HEAD'].includes(req.method)) {
      res.writeHead(405, { Allow: 'GET, HEAD' }).end();
      return;
    }
    const pathname = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
    const filename = await realpath(resolve(base, '.' + (pathname === '/' ? '/index.html' : pathname)));
    if (!filename.startsWith(base + sep)) throw new Error('Outside root');
    const file = await stat(filename);
    if (!file.isFile()) throw new Error('Missing file');
    res.writeHead(200, {
      'Content-Type': types[extname(filename).toLowerCase()] || 'application/octet-stream',
      'Content-Length': file.size,
      'X-Content-Type-Options': 'nosniff',
    });
    if (req.method === 'HEAD') res.end();
    else createReadStream(filename).on('error', () => res.destroy()).pipe(res);
  } catch {
    res.writeHead(404).end('Not found');
  }
});

let port = Number(process.env.PORT || 4173);
const maxAttempts = 10;
let attempts = 0;

function startListening() {
  server.listen(port, '127.0.0.1');
}

server.on('error', err => {
  if (err.code === 'EADDRINUSE' && !process.env.PORT && attempts < maxAttempts) {
    attempts += 1;
    port += 1;
    startListening();
    return;
  }
  console.error(err.code === 'EADDRINUSE' ? `Port ${port} is already in use. Set PORT to another port.` : err.message);
  process.exitCode = 1;
});

server.on('listening', () => {
  const address = 'http://127.0.0.1:' + server.address().port;
  console.log('RoadWatch running at ' + address);
  console.log('Press Ctrl+C to stop the server.');
  if (process.argv.includes('--open') && process.platform === 'win32') {
    spawn('rundll32.exe', ['url.dll,FileProtocolHandler', address], { detached: true, stdio: 'ignore', windowsHide: true }).unref();
  }
});

startListening();
