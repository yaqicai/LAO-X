import { readFile, access } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url));
for (const page of ['index.html', 'parallel.html', 'single.html']) {
const html = await readFile(path.join(root, page), 'utf8');
const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(m => m[1]);
if (new Set(ids).size !== ids.length) throw new Error('Duplicate HTML IDs');
const links = [...html.matchAll(/(?:src|href)="([^"]+)"/g)].map(m => m[1]);
for (const link of links) {
  if (link === '#' || /^(?:https?:|data:)/.test(link)) continue;
  if (link.startsWith('#')) {
    if (!ids.includes(link.slice(1))) throw new Error(`Missing anchor: ${link}`);
  } else await access(path.join(root, link.split('#')[0]));
}
if (/hero-playback|hero-restart|hero-timeline/.test(html)) throw new Error(`Playback controls present: ${page}`);
if (!html.includes('https://arxiv.org/pdf/2610.07326v1') || !html.includes('https://github.com/yaqicai/LAO-X')) throw new Error(`Missing public resource links: ${page}`);
}
const scenes = ['case', 'jug', 'bottle', 'clutter'];
for (const scene of scenes) for (let i = 0; i < 5; i++) await access(path.join(root, `assets/compare-${scene}-${i}.png`));
for (const name of ['hazard','unseen','shapes','multipart','transparent']) for (let i = 0; i < 2; i++) await access(path.join(root, `assets/${name}-${i}.png`));
console.log('Static checks passed: unique IDs, anchors, local assets, and all interactive image variants.');

const real = JSON.parse(await readFile(path.join(root, 'assets/real/manifest.json'), 'utf8'));
if (real.cases.length !== 60 || real.datasets.length !== 6) throw new Error('Expected 60 curated scans across six datasets');
const resources = new Set();
let outputs = 0;
for (const dataset of real.datasets) {
  const cases = real.cases.filter(c => c.dataset === dataset.id);
  if (cases.length !== 10 || cases.some((c, i) => c.rank !== i + 1 || (i && c.score > cases[i - 1].score))) throw new Error(`Invalid ranking: ${dataset.id}`);
  for (const scan of cases) {
    resources.add(scan.image); resources.add(scan.thumbnail);
    if (!scan.prompts[scan.default_prompt]?.outputs[scan.default_output]) throw new Error('Invalid default selection');
    for (const prompt of scan.prompts) {
      if (prompt.outputs.length !== (prompt.excluded_g1 ? 9 : 10)) throw new Error('Incorrect granularity exclusion');
      if (!prompt.excluded_g1 && prompt.outputs[9].mask_area_pixels < prompt.outputs[8].mask_area_pixels) throw new Error('Shrinking g=1 is still present');
      const best = prompt.outputs.reduce((a, o, i, all) => o.box_iou > all[a].box_iou ? i : a, 0);
      if (prompt.best_output !== best) throw new Error('Incorrect per-point optimum');
      for (const [i, output] of prompt.outputs.entries()) {
        if (output.granularity !== (i + 1) / 10 || output.box_iou < 0 || output.box_iou > 1) throw new Error('Invalid prediction metadata');
        for (const key of ['mask', 'overlay', 'display']) resources.add(output[key]);
        outputs++;
      }
    }
  }
}
await Promise.all([...resources].map(resource => access(path.join(root, resource))));
console.log(`Real data checks passed: 60 scans, ${outputs} outputs, ${resources.size} local resources.`);
