'use strict';

// Public paper, code and model resources.
const RELEASE = { code: 'https://github.com/yaqicai/LAO-X', model: 'https://huggingface.co/YAQICAI/LAO-X' };
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const NS = 'http://www.w3.org/2000/svg';
let scene = 'case';
const sceneNames = { case: 'Everyday items', jug: 'Fine structures', bottle: 'Transparent objects', clutter: 'Heavy clutter' };
function renderComparison() {
  $('#comparison-grid').innerHTML = ['SAM2', 'SAM3', 'UnSAMv2', 'LAO-X'].map((model, i) => `<article class="comparison-card ${i === 3 ? 'ours' : ''}"><div class="comparison-card-header">${model}<span>${i === 3 ? 'OURS' : '0' + (i + 1)}</span></div><div class="comparison-image"><img src="assets/compare-${scene}-${i + 1}.png" alt="${model} top-5 predicted masks: ${sceneNames[scene]}, from Figure 3" loading="lazy"></div></article>`).join('');
  $$('[data-scene]').forEach(button => { const active = button.dataset.scene === scene; button.classList.toggle('is-active', active); button.setAttribute('aria-pressed', active); });
}
$$('[data-scene]').forEach(button => button.addEventListener('click', () => { scene = button.dataset.scene; renderComparison(); }));

// Paper Table 1, page 7. Order within each method: AP50, AP75, AP50:95.
const benchmarks = [
  { name: 'DET-COMPASS', values: [[36.10,29.13,29.74],[11.46,9.93,9.69],[33.25,27.90,27.44],[63.39,55.73,53.33]], raxo: [[54.20,48.80,47.90],[19.80,16.80,16.10]] },
  { name: 'CLCXray', values: [[17.75,14.73,12.80],[28.14,21.74,18.93],[30.71,20.22,19.06],[37.61,27.44,25.29]], raxo: [[29.60,24.40,22.20],[9.70,7.50,7.10]] },
  { name: 'DvXray', values: [[10.38,9.72,8.87],[11.77,9.82,8.55],[19.58,15.34,14.05],[23.56,18.33,16.87]], raxo: [[26.60,24.10,22.60],[14.10,12.90,12.40]] },
  { name: 'PIXray', values: [[6.33,5.45,5.14],[12.04,9.98,9.05],[22.54,17.72,16.97],[30.26,22.40,21.37]], raxo: [[45.00,39.00,36.90],[19.80,16.80,16.10]] },
  { name: 'HiXray', values: [[18.99,11.22,10.58],[24.32,12.15,12.77],[43.16,21.14,22.26],[43.35,21.75,22.47]], raxo: [[27.20,19.40,17.10],[13.00,8.70,7.90]] },
  { name: 'PIDray', values: [[1.97,1.16,1.13],[7.21,3.66,3.79],[9.57,4.79,4.94],[11.61,5.16,5.65]], raxo: [[21.40,17.90,16.50],[17.10,14.30,13.40]] }
];
// Equal-weight average of Table 1 box mAP (AP50:95), not relative percent gain.
const meanBoxMap = [0, 3].map(method => benchmarks.reduce((sum, b) => sum + b.values[method][2], 0) / benchmarks.length);
const meanMapGain = meanBoxMap[1] - meanBoxMap[0];
$('#mean-map-gain').textContent = `+${meanMapGain.toFixed(1)}`;
$('#performance-highlight').title = `Mean box mAP across ${benchmarks.length} test-only benchmarks: LAO-X ${meanBoxMap[1].toFixed(2)} vs. SAM2 ${meanBoxMap[0].toFixed(2)}; +${meanMapGain.toFixed(2)} points. Each dataset has equal weight.`;
let metric = 2;
const metricNames = ['AP₅₀', 'AP₇₅', 'mAP'];
const colors = ['#9aa9bb', '#eca855', '#b497d2', '#2476ef'];
function radarChart(dataset) {
  const max = Math.max(...dataset.values.flat()) * 1.12;
  const vertex = (v, axis) => { const angle = (-90 - axis * 120) * Math.PI / 180; return `${(90 + Math.cos(angle) * 60 * v).toFixed(2)},${(86 + Math.sin(angle) * 60 * v).toFixed(2)}`; };
  const triangle = level => [0,1,2].map(axis => vertex(level, axis)).join(' ');
  const grid = [.25,.5,.75,1].map(v => `<polygon class="radar-grid" points="${triangle(v)}"/>`).join('');
  const axes = [0,1,2].map(axis => `<line class="radar-axis" x1="90" y1="86" x2="${vertex(1,axis).split(',')[0]}" y2="${vertex(1,axis).split(',')[1]}"/>`).join('');
  const plots = dataset.values.map((values,i) => `<polygon points="${values.map((v,a) => vertex(v / max,a)).join(' ')}" stroke="${colors[i]}" fill="${colors[i]}" fill-opacity="${i === 3 ? '.11' : '.025'}" style="stroke-width:${i === 3 ? 1.8 : 1.1}"/>`).join('');
  return `<svg class="radar" viewBox="0 0 180 155" role="img" aria-label="${dataset.name}: AP50, AP75, and mAP for SAM2, SAM3, UnSAMv2, and LAO-X. Exact values are available in the full results table."><title>${dataset.name}, chart range 0–${max.toFixed(1)}%</title>${grid}${axes}${plots}<text x="90" y="16" text-anchor="middle" style="fill:${metric === 0 ? '#2476ef' : '#8698ac'}">AP50</text><text x="15" y="132" text-anchor="middle" style="fill:${metric === 1 ? '#2476ef' : '#8698ac'}">AP75</text><text x="163" y="132" text-anchor="middle" style="fill:${metric === 2 ? '#2476ef' : '#8698ac'}">mAP</text></svg>`;
}
function renderBenchmarks() {
  $('#benchmark-grid').innerHTML = benchmarks.map(d => `<article class="benchmark-card"><h3>${d.name}</h3>${radarChart(d)}<div class="benchmark-value">${d.values[3][metric].toFixed(2)}<small>${metricNames[metric]}</small></div><p class="benchmark-delta">+${(d.values[3][metric] - d.values[0][metric]).toFixed(2)} pts vs. SAM2</p></article>`).join('');
  $$('[data-metric]').forEach(button => { const active = Number(button.dataset.metric) === metric; button.classList.toggle('is-active', active); button.setAttribute('aria-pressed', active); });
  renderFullResults();
}
function renderFullResults() {
  const methods = ['RAXO (in-house descriptors)', 'RAXO (Internet descriptors)', 'SAM2', 'SAM3', 'UnSAMv2', 'LAO-X'];
  const rows = methods.map((method,i) => `<tr><th scope="row">${method}</th>${benchmarks.map(d => `<td>${(i < 2 ? d.raxo[i][metric] : d.values[i - 2][metric]).toFixed(2)}</td>`).join('')}</tr>`).join('');
  $('#full-results').innerHTML = `<h3>Box localization · ${metricNames[metric]} (%)</h3><div class="table-scroll" tabindex="0" role="region" aria-label="Scrollable benchmark table"><table><caption>Table 1, supplied paper · mAP = AP50:95</caption><thead><tr><th scope="col">Method</th>${benchmarks.map(d => `<th scope="col">${d.name}</th>`).join('')}</tr></thead><tbody>${rows}</tbody></table></div><p><strong>Evaluation.</strong> All six datasets are test-only for LAO-X; their images are not used for mining, synthesis, or training. Predicted masks are converted to tight boxes. To account for partial annotations, the paper evaluates proposals whose centers fall inside ground-truth boxes expanded by 1.5×, then applies COCO matching. Appendix F reports the optimal granularity per benchmark. Radar axes use a common scale within each dataset, with a separate range for each dataset.</p><p><strong>Mean across benchmarks.</strong> Equally weighting the six test-only datasets gives box mAP ${meanBoxMap[1].toFixed(2)} for LAO-X and ${meanBoxMap[0].toFixed(2)} for SAM2: +${meanMapGain.toFixed(2)} points.</p><p><strong>Baseline context.</strong> RAXO uses semantic class descriptors and is reproduced from its original paper under conventional COCO evaluation. It is provided here for context and uses a different protocol. LAO-X outperforms the three promptable/self-supervised baselines in box AP on all six datasets; it does not outperform RAXO on every dataset. Results are single-run measurements, as reported in the paper.</p>`;
}
$$('[data-metric]').forEach(button => button.addEventListener('click', () => { metric = Number(button.dataset.metric); renderBenchmarks(); }));
$('#toggle-results').addEventListener('click', () => {
  const expanded = $('#full-results').hidden;
  $('#full-results').hidden = !expanded;
  $('#toggle-results').setAttribute('aria-expanded', expanded);
  $('#toggle-results').innerHTML = `${expanded ? 'Hide' : 'Full'} results & evaluation <span>${expanded ? '−' : '＋'}</span>`;
});

const tiers = { easy: ['2–4','0.05'], hard: ['4–7','0.30'], extreme: ['6–9','0.50'] };
$$('[data-tier]').forEach(button => button.addEventListener('click', () => {
  $$('[data-tier]').forEach(b => { const active = b === button; b.classList.toggle('is-active', active); b.setAttribute('aria-pressed', active); });
  $('.curriculum-card').dataset.stage = button.dataset.tier;
  const [count, overlap] = tiers[button.dataset.tier];
  $('#tier-stats').innerHTML = `<span><strong>${count}</strong> objects</span><span><strong>${overlap}</strong> max overlap</span>`;
}));

const dialog = $('#lightbox');
let dialogTrigger = null;
function openDialog(title, content, className = '') {
  dialogTrigger = document.activeElement;
  $('#dialog-title').textContent = title;
  $('#dialog-body').innerHTML = content;
  dialog.className = className;
  dialog.showModal();
}
$('#close-dialog').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', e => { if (e.target !== dialog) return; const rect = dialog.getBoundingClientRect(); if (e.clientX < rect.left || e.clientX > rect.right || e.clientY < rect.top || e.clientY > rect.bottom) dialog.close(); });
dialog.addEventListener('close', () => dialogTrigger?.focus());
const cases = {
  hazard: ['Unconventional threats', 'LAO-X localizes a 3D-printed plastic pistol. The point-prompted output comes from Figure 1.'],
  unseen: ['Objects beyond a fixed vocabulary', 'An unfamiliar object can still be localized without a category name or text prompt. Lizard example from Figure 1.'],
  shapes: ['Familiar objects. Unfamiliar shapes.', 'Common objects can vary in shape and composition. Scissors examples from Figure 1.'],
  multipart: ['From parts to a whole instance', 'X-ray objectness must account for internal components and projected overlap. Multi-part example from Figure 1.'],
  transparent: ['Almost invisible. Still an object.', 'Low-attenuation plastic can blend into the background. Two point-prompted bottle predictions from Figure 1.']
};
$$('[data-case]').forEach(button => button.addEventListener('click', () => {
  const id = button.dataset.case, [title, description] = cases[id];
  openDialog(title, `<div class="case-pair"><div><img src="assets/${id}-0.png" alt="${id === 'transparent' ? 'First bottle prediction' : 'Original X-ray scan'}"><span>${id === 'transparent' ? 'Point prompt · object 1' : 'Original X-ray'}</span></div><div><img src="assets/${id}-1.png" alt="LAO-X mask prediction"><span>${id === 'transparent' ? 'Point prompt · object 2' : 'LAO-X prediction'}</span></div></div><p>${description}</p>`);
}));
$('#view-pipeline').addEventListener('click', () => openDialog('The LAO-X framework', '<img src="assets/method-overview.png" alt="Figure 2: saliency-guided object mining, physics-guided synthesis, and granularity-conditioned SAM2 curriculum training"><p>Figure 2 · Objects mined from 1,621 COMPASS-XP scans supply 18,000 synthetic training scenes. A SAM2-small localizer learns from holistic and part-level pseudo masks through an Easy → Hard → Extreme curriculum.</p>'));
$('#open-compare').addEventListener('click', () => {
  openDialog(`UnSAMv2 vs. LAO-X · ${sceneNames[scene]}`, `<div class="compare-slider"><img src="assets/compare-${scene}-4.png" alt="LAO-X prediction"><img class="compare-before" src="assets/compare-${scene}-3.png" alt="UnSAMv2 prediction"><span class="wipe-label wipe-left">UnSAMv2</span><span class="wipe-label wipe-right">LAO-X</span><span class="wipe-line"></span><input id="wipe" type="range" min="0" max="100" value="50" aria-label="Compare UnSAMv2 and LAO-X predictions" aria-valuetext="50% UnSAMv2, 50% LAO-X"></div><label class="wipe-control-label" for="wipe">Drag to compare, or use the arrow keys.</label><p>Original predictions from Figure 3. Both images show the same scan; colors distinguish predicted masks.</p>`);
  $('#wipe').addEventListener('input', e => {
    $('.compare-before').style.clipPath = `inset(0 ${100 - Number(e.target.value)}% 0 0)`;
    $('.wipe-line').style.left = `${e.target.value}%`;
    e.target.setAttribute('aria-valuetext', `${e.target.value}% UnSAMv2, ${100 - Number(e.target.value)}% LAO-X`);
  });
});
$$('[data-resource]').forEach(button => button.addEventListener('click', () => {
  const type = button.dataset.resource;
  if (RELEASE[type]) { window.open(RELEASE[type], '_blank', 'noopener,noreferrer'); return; }
  openDialog(type === 'model' ? 'Model checkpoints' : 'Code & models', `<div class="resource-icon">${type === 'model' ? '◇' : '⌘'}</div><h3>Release links coming soon.</h3><p>${type === 'model' ? 'The model checkpoint link will appear here when the public release is available.' : 'The project repository and model links will appear here when the public release is available.'}</p><p>Explore the method, evaluation, and implementation details in the paper.</p><a class="button primary" href="https://arxiv.org/pdf/2610.07326v1" target="_blank" rel="noopener">Read the paper <span>↗</span></a>`, 'resource-modal');
}));

let toastTimeout;
function toast(message) {
  clearTimeout(toastTimeout); $('#toast').textContent = message; $('#toast').classList.add('is-visible');
  toastTimeout = setTimeout(() => $('#toast').classList.remove('is-visible'), 3200);
}
renderComparison();
renderBenchmarks();

// Copy exactly the citation displayed on the page.
let citationCopyTimeout;
$('#copy-bibtex').addEventListener('click', async () => {
  const code = $('#bibtex'), label = $('#copy-bibtex span');
  try {
    await navigator.clipboard.writeText(code.textContent.trim() + '\n');
    label.textContent = 'Copied';
    toast('BibTeX copied');
    clearTimeout(citationCopyTimeout);
    citationCopyTimeout = setTimeout(() => { label.textContent = 'Copy'; }, 2200);
  } catch {
    const range = document.createRange(); range.selectNodeContents(code);
    const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
    toast('Citation selected. Press ⌘C or Ctrl+C to copy.');
  }
});
