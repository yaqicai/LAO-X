import { cp, mkdir, readdir, rm } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url));
const out = path.join(root, 'dist');
// The website is native HTML/CSS/JS: building exports deployable static files.
await rm(out, { recursive: true, force: true });
await mkdir(out, { recursive: true });
for (const entry of await readdir(root, { withFileTypes: true })) {
  if (entry.isFile() && (/\.(html|css|js)$/.test(entry.name) || entry.name === '.nojekyll')) {
    await cp(path.join(root, entry.name), path.join(out, entry.name));
  }
}
await cp(path.join(root, 'assets'), path.join(out, 'assets'), { recursive: true });
console.log('Static website exported to dist/');
