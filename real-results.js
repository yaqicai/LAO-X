'use strict';

(() => {
  const data = window.LAOX_REAL;
  const q = selector => document.querySelector(selector);
  const ns = 'http://www.w3.org/2000/svg';
  const svg = (name, attrs) => {
    const node = document.createElementNS(ns, name);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    return node;
  };
  let dataset = data.datasets[0].id;
  let active, promptIndex = 0, outputIndex = 0;
  let display = 'mask', opacity = 62, history = [], version = 0;
  const cache = new Map();
  const preload = src => {
    if (!cache.has(src)) {
      const img = new Image();
      img.src = src;
      cache.set(src, img.decode().then(() => img));
      if (cache.size > 32) cache.delete(cache.keys().next().value);
    }
    return cache.get(src);
  };
  // Blend actual prediction layers without changing their geometry or IoU.
  const mask = q('#real-mask');
  const maskStack = document.createElement('div');
  maskStack.className = 'real-mask-stack';
  mask.before(maskStack);
  maskStack.append(mask);
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  let maskSource = '', maskFrame = 0;
  let maskWeights = new Map();

  function settleMask() {
    cancelAnimationFrame(maskFrame); maskFrame = 0;
    for (const layer of maskWeights.keys()) if (layer !== mask) layer.remove();
    maskWeights = new Map([[mask, 1]]);
    mask.style.opacity = '1';
  }

  function resetMask() {
    settleMask(); maskSource = ''; mask.hidden = true;
  }

  function showMask(src) {
    if (src === maskSource) return;
    cancelAnimationFrame(maskFrame);
    const animate = maskSource && !mask.hidden && display === 'mask' && !reducedMotion.matches;
    if (animate) {
      for (const [layer, weight] of maskWeights) {
        if (layer !== mask && weight < .001) { layer.remove(); maskWeights.delete(layer); }
      }
      // Retarget from the currently displayed blend, even during a fast drag.
      const weight = maskWeights.get(mask) || 0;
      if (weight > 0) {
        const previous = mask.cloneNode();
        previous.removeAttribute('id'); previous.alt = '';
        previous.setAttribute('aria-hidden', 'true');
        mask.before(previous); maskWeights.set(previous, weight);
      }
      maskWeights.delete(mask);
    }
    mask.src = src; mask.hidden = false; maskSource = src;
    if (!animate) { settleMask(); return; }
    maskWeights.set(mask, 0); mask.style.opacity = '0';
    const from = new Map(maskWeights), started = performance.now();
    const frame = now => {
      const progress = Math.min(1, (now - started) / 260);
      const blend = progress * progress * (3 - 2 * progress);
      for (const [layer, initial] of from) {
        const weight = initial + ((layer === mask ? 1 : 0) - initial) * blend;
        maskWeights.set(layer, weight); layer.style.opacity = String(weight);
      }
      if (progress < 1) maskFrame = requestAnimationFrame(frame);
      else settleMask();
    };
    maskFrame = requestAnimationFrame(frame);
  }
  reducedMotion.addEventListener('change', () => { if (reducedMotion.matches) settleMask(); });
  document.addEventListener('visibilitychange', () => { if (document.hidden) settleMask(); });

  const currentPrompt = () => active.prompts[promptIndex];
  const objectIndex = p => active.objects.findIndex(o => String(o.annotation_id) === String(p.annotation_id));
  const pointName = p => p.point_index === 0 ? 'Center' : `Point ${p.point_index + 1}`;
  const remember = () => history.push({promptIndex, outputIndex});

  function renderCurve() {
    const chart = q('#iou-curve'), prompt = currentPrompt();
    chart.replaceChildren();
    const last = prompt.outputs.at(-1).granularity;
    const x = g => 28 + (g - .1) / (last - .1) * 224;
    const y = iou => 106 - iou * 78;
    chart.setAttribute('aria-label', 'IoU with GT across granularity. Computed from the tight predicted box and the GT box.');
    [0, .5, 1].forEach(level => {
      chart.append(svg('line', {x1:28, x2:252, y1:y(level), y2:y(level), stroke:'#e5edf5'}));
      const label = svg('text', {x:20, y:y(level)+4, 'text-anchor':'end', class:'curve-axis'});
      label.textContent = String(level); chart.append(label);
    });
    [.1, .5, last].forEach(g => {
      const label = svg('text', {x:x(g), y:126, 'text-anchor':'middle', class:'curve-axis'});
      label.textContent = g.toFixed(1); chart.append(label);
    });
    const points = prompt.outputs.map(o => `${x(o.granularity)},${y(o.box_iou)}`).join(' ');
    chart.append(svg('polygon', {points:`28,106 ${points} 252,106`, fill:'#2476ef0c'}));
    chart.append(svg('polyline', {points, fill:'none', stroke:'#2476ef', 'stroke-width':2, 'stroke-linejoin':'round'}));
    const selected = prompt.outputs[outputIndex];
    chart.append(svg('line', {x1:x(selected.granularity), x2:x(selected.granularity), y1:22, y2:106, stroke:'#2476ef55', 'stroke-dasharray':'3 4'}));
    prompt.outputs.forEach((o, i) => {
      const group = svg('g', {role:'button', tabindex:'0', 'aria-label':`Granularity ${o.granularity.toFixed(1)}, IoU ${o.box_iou.toFixed(3)}${i === prompt.best_output ? ', best' : ''}`, class:'curve-point'});
      group.append(svg('circle', {cx:x(o.granularity), cy:y(o.box_iou), r:11, fill:'transparent'}));
      group.append(svg('circle', {cx:x(o.granularity), cy:y(o.box_iou), r:i===outputIndex?5:3, fill:i===outputIndex?'#2476ef':'#fff', stroke:'#2476ef', 'stroke-width':1.7}));
      if (i === prompt.best_output) {
        const star = svg('text', {x:x(o.granularity), y:y(o.box_iou)-10, 'text-anchor':'middle', fill:'#149e7c', 'font-size':15});
        star.textContent = '★'; group.append(star);
      }
      const choose = () => { remember(); outputIndex = i; render(); };
      group.addEventListener('click', choose);
      group.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); choose(); } });
      chart.append(group);
    });
    q('#best-match').textContent = `★ Best ${prompt.outputs[prompt.best_output].granularity.toFixed(1)}`;
  }

  function renderOverlay() {
    const overlay = q('#real-overlay');
    overlay.replaceChildren();
    overlay.setAttribute('viewBox', `0 0 ${active.width} ${active.height}`);
    const prompt = currentPrompt(), output = prompt.outputs[outputIndex];
    const obj = active.objects[objectIndex(prompt)];
    const strokeWidth = Math.max(active.width, active.height) / 340;
    if (q('#show-gt').checked) {
      const [x, y, width, height] = obj.bbox;
      overlay.append(svg('rect', {x, y, width, height, fill: 'none', stroke: '#ec8b28', 'stroke-width': strokeWidth, 'stroke-dasharray': `${strokeWidth * 4} ${strokeWidth * 3}`}));
    }
    if (display === 'box' && output.bbox_xyxy) {
      const [x, y, x2, y2] = output.bbox_xyxy;
      overlay.append(svg('rect', {x, y, width: x2 - x, height: y2 - y, fill: '#e80aa110', stroke: '#d10b93', 'stroke-width': strokeWidth}));
    }
    active.prompts.forEach((p, i) => {
      const selected = i === promptIndex;
      const group = svg('g', {'data-prompt': i, class: 'real-prompt'});
      const radius = Math.max(active.width, active.height) / 85;
      group.append(svg('circle', {cx: p.point[0], cy: p.point[1], r: radius * 1.9, fill: 'transparent'}));
      group.append(svg('circle', {cx: p.point[0], cy: p.point[1], r: radius * (selected ? 1 : .72), fill: selected ? '#2476ef' : '#fff', stroke: selected ? '#fff' : '#2476ef', 'stroke-width': strokeWidth}));
      const title = svg('title', {});
      title.textContent = `Object ${objectIndex(p) + 1} · ${pointName(p)}`;
      group.append(title);
      group.addEventListener('click', () => choosePoint(i));
      overlay.append(group);
    });
  }

  async function render() {
    const thisVersion = ++version;
    const prompt = currentPrompt(), output = prompt.outputs[outputIndex];
    q('#granularity').max = prompt.outputs.at(-1).granularity;
    q('#granularity').value = output.granularity;
    q('#granularity-max').textContent = prompt.outputs.at(-1).granularity.toFixed(1);
    q('#granularity-output').value = output.granularity.toFixed(1);
    q('#opacity').value = opacity;
    q('#opacity-output').value = `${opacity}%`;
    q('#opacity').disabled = display === 'box';
    q('#undo').disabled = !history.length;
    q('#selection-status').textContent = `Object ${objectIndex(prompt) + 1} · ${pointName(prompt)}`;
    q('#download-mask').href = output.mask;
    q('#scan-stage').dataset.granularity = output.granularity;
    q('#scan-stage').dataset.case = String(active.id);
    document.querySelectorAll('[data-prompt-choice]').forEach(button => button.setAttribute('aria-pressed', String(Number(button.dataset.promptChoice) === promptIndex)));
    document.querySelectorAll('[data-display]').forEach(button => {
      const selected = button.dataset.display === display;
      button.setAttribute('aria-pressed', String(selected));
      button.classList.toggle('is-active', selected);
    });
    renderOverlay();
    renderCurve();
    maskStack.style.opacity = String(opacity / 100);
    maskStack.hidden = display !== 'mask';
    if (display !== 'mask') settleMask();
    q('#scan-stage').setAttribute('aria-busy', 'true');
    try {
      await Promise.all([preload(active.image), preload(output.display)]);
      if (thisVersion !== version) return;
      q('#real-image').src = active.image;
      showMask(output.display);
      q('#scan-stage').setAttribute('aria-busy', 'false');
    } catch {
      if (thisVersion !== version) return;
      q('#selection-status').textContent = 'Image unavailable. Please reload.';
      q('#scan-stage').setAttribute('aria-busy', 'false');
    }
  }

  function warmPrompt() {
    currentPrompt().outputs.forEach(o => preload(o.display).catch(() => {}));
  }
  function choosePoint(i) {
    remember(); resetMask(); promptIndex = i; outputIndex = Math.min(outputIndex, currentPrompt().outputs.length - 1); warmPrompt(); render();
  }
  function chooseCase(c) {
    active = c; promptIndex = c.default_prompt; outputIndex = c.default_output; history = [];
    q('#real-image').removeAttribute('src');
    resetMask();
    const name = data.datasets.find(d => d.id === dataset).name;
    q('#real-case-title').textContent = `${name} / ${String(c.rank).padStart(2, '0')}`;
    q('#real-image').alt = `${name}: ${c.file_name}`;
    q('#point-count').textContent = `${c.prompts.length} points`;
    const choices = q('#real-point-choices');
    choices.replaceChildren();
    c.objects.forEach((obj, index) => {
      const group = document.createElement('div'); group.className = 'point-group';
      const label = document.createElement('span'); label.textContent = `Object ${index + 1}`;
      group.append(label);
      c.prompts.forEach((p, i) => {
        if (String(p.annotation_id) !== String(obj.annotation_id)) return;
        const button = document.createElement('button');
        button.dataset.promptChoice = i;
        button.textContent = pointName(p);
        button.setAttribute('aria-label', `Object ${index + 1}, ${pointName(p)}`);
        button.addEventListener('click', () => choosePoint(i));
        group.append(button);
      });
      choices.append(group);
    });
    document.querySelectorAll('[data-case-choice]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.caseChoice === String(c.id))));
    warmPrompt(); render();
  }

  function chooseDataset(id) {
    dataset = id;
    document.querySelectorAll('[data-dataset]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.dataset === id)));
    const gallery = q('#real-gallery'); gallery.replaceChildren(); gallery.scrollLeft = 0;
    const scans = data.cases.filter(c => c.dataset === id);
    q('#gallery-label').textContent = `${data.datasets.find(d => d.id === id).name} · 10 scans`;
    scans.forEach(c => {
      const button = document.createElement('button'); button.dataset.caseChoice = c.id;
      button.setAttribute('aria-label', `Scan ${c.rank}: ${c.file_name}`);
      const image = document.createElement('img'); image.src = c.thumbnail; image.alt = ''; image.loading = 'lazy';
      const caption = document.createElement('span'); caption.textContent = String(c.rank).padStart(2, '0');
      button.append(image, caption); button.addEventListener('click', () => chooseCase(c)); gallery.append(button);
    });
    chooseCase(scans[0]);
  }

  data.datasets.forEach(d => {
    const button = document.createElement('button'); button.dataset.dataset = d.id; button.textContent = d.name;
    button.addEventListener('click', () => chooseDataset(d.id)); q('#dataset-tabs').append(button);
  });
  q('#granularity').addEventListener('input', e => { outputIndex = currentPrompt().outputs.findIndex(o => Math.abs(o.granularity - Number(e.target.value)) < .001); render(); });
  q('#opacity').addEventListener('input', e => { opacity = Number(e.target.value); render(); });
  q('#show-gt').addEventListener('change', renderOverlay);
  document.querySelectorAll('[data-display]').forEach(b => b.addEventListener('click', () => { display = b.dataset.display; render(); }));
  q('#best-match').addEventListener('click', () => {
    remember(); outputIndex = currentPrompt().best_output; render();
  });
  q('#undo').addEventListener('click', () => {
    const previous = history.pop(); if (!previous) return;
    if (promptIndex !== previous.promptIndex) resetMask();
    promptIndex = previous.promptIndex; outputIndex = previous.outputIndex; warmPrompt(); render();
  });
  q('#reset').addEventListener('click', () => { display = 'mask'; opacity = 62; q('#show-gt').checked = false; chooseCase(active); });
  const expanded = value => {
    q('.real-explorer').classList.toggle('is-expanded', value);
    document.body.classList.toggle('has-expanded', value);
    q('#expand').setAttribute('aria-expanded', String(value));
    q('#expand').setAttribute('aria-label', value ? 'Collapse demo' : 'Expand demo');
  };
  q('#expand').addEventListener('click', () => expanded(!q('.real-explorer').classList.contains('is-expanded')));
  document.addEventListener('keydown', e => { if (e.key === 'Escape') expanded(false); });
  chooseDataset(dataset);
})();
