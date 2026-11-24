import {
  infer,
  MODELS,
  DEFAULT_THRESHOLD,
  DEFAULT_IOU,
  filterDetections,
  computeSeverity,
  registerCustomModel,
} from './vision.js';
import {
  saveInspection,
  listInspections,
  deleteInspection,
  exportInspection,
} from './storage.js';

const $ = id => document.getElementById(id);
const svgNS = 'http://www.w3.org/2000/svg';

const icons = {
  scan: 'M4 7V5a1 1 0 0 1 1-1h2M17 4h2a1 1 0 0 1 1 1v2M20 17v2a1 1 0 0 1-1 1h-2M7 20H5a1 1 0 0 1-1-1v-2M7 12h10',
  target: 'M12 2v4M12 18v4M2 12h4M18 12h4M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8z',
  image: 'M4 5h16a1 1 0 0 1 1 1v12a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1zm0 11 4.5-4.5 3.5 3.5 3-3 5 5M15 9h.01',
  clock: 'M12 6v6l4 2M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20z',
  upload: 'M12 16V4m-5 5 5-5 5 5M4 17v2a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-2',
  download: 'M12 4v12m-5-5 5 5 5-5M4 17v2a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-2',
  eye: 'M2 12s3.5-6.5 10-6.5S22 12 22 12s-3.5 6.5-10 6.5S2 12 2 12zm10 3a3 3 0 1 0 0-6 3 3 0 0 0 0 6z',
  expand: 'M4 9V4h5M15 4h5v5M20 15v5h-5M9 20H4v-5',
  layers: 'M12 3 3 8l9 5 9-5-9-5zm-9 9 9 5 9-5M3 16l9 5 9-5',
  cpu: 'M9 9h6v6H9zm-5 1H2m2 4H2m20-4h-2m2 4h-2M10 4V2m4 2V2m-4 20v-2m4 2v-2M6 6h12v12H6z',
  camera: 'M4 7h3l2-2h6l2 2h3a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V8a1 1 0 0 1 1-1zm8 10a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7z',
  arrowLeft: 'M19 12H5m6-6-6 6 6 6',
  check: 'M5 13l4 4L19 7',
  x: 'M6 6l12 12M18 6 6 18',
  undo: 'M9 14 4 9l5-5M4 9h10.5a5.5 5.5 0 0 1 0 11H11',
  trash: 'M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3',
};

function renderIcon(name) {
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="${icons[name] || ''}"/></svg>`;
}

for (const el of document.querySelectorAll('[data-icon]')) {
  el.innerHTML = renderIcon(el.dataset.icon);
}

const samples = {
  wet: { src: './demo/road.jpg', name: 'wet-road.jpg' },
  urban: { src: './demo/bengaluru.jpg', name: 'urban-pavement.jpg' },
  asphalt: { src: './demo/asphalt.jpg', name: 'asphalt-surface.jpg' },
};

const sourceLabel = { demo: 'Sample', upload: 'Uploaded', history: 'Saved' };

const state = {
  record: null,
  rawDetections: null,
  selected: null,
  undo: null,
  dirty: false,
  conflict: false,
  busy: false,
  ready: false,
  mode: 'demo',
  history: [],
  outlines: true,
  modelId: 'default',
  threshold: DEFAULT_THRESHOLD,
  iou: DEFAULT_IOU,
};

const photo = $('photo');

const make = (tag, text, className) => {
  const el = document.createElement(tag);
  if (text !== undefined) el.textContent = text;
  if (className) el.className = className;
  return el;
};

function makeBtn(label, action, className = 'btn btn-ghost btn-sm', iconName = null) {
  const btn = make('button', undefined, className);
  btn.type = 'button';
  if (iconName) {
    const ic = make('i');
    ic.dataset.icon = iconName;
    ic.innerHTML = renderIcon(iconName);
    btn.append(ic);
  }
  if (label) btn.append(make('span', label));
  btn.addEventListener('click', action);
  return btn;
}

function tell(msg) {
  $('notice').textContent = msg || '';
  $('noticeWrap').hidden = !msg;
}
$('noticeClose').onclick = () => tell('');

const describe = err => (err instanceof Error ? err.message : String(err));

function syncModelDropdown() {
  const sel = $('modelSelect');
  sel.replaceChildren();
  for (const m of MODELS) {
    const opt = make('option', m.name);
    opt.value = m.id;
    sel.append(opt);
  }
  sel.value = state.modelId;
}

function activeModelName() {
  return MODELS.find(m => m.id === state.modelId)?.name ?? 'YOLOv8s RoadWatch';
}

function setBusy(value) {
  state.busy = value;
  $('demoChoice').disabled = value;
  $('upload').disabled = value;
  $('loadModelBtn').disabled = value;
  $('modelSelect').disabled = value;
  $('analyze').disabled = value || !state.ready;
  $('export').disabled = value || state.dirty || !state.record;
  $('exportPngBtn').disabled = value || !state.record;
  $('downloadPng').disabled = value || !state.record;
  for (const el of document.querySelectorAll('.nav-tab, #back, #decision button, .history-entry button, #reanalyze')) {
    el.disabled = value;
  }
}

function buildCandidates(raw, width, height, threshold, iou, prevStatuses = new Map()) {
  const kept = filterDetections(raw, threshold, iou);
  return kept.map((item, idx) => {
    const id = 'det-' + (idx + 1);
    const sev = computeSeverity(item.box, width, height, item.score);
    return {
      id,
      label: item.label || 'Pothole',
      box: [...item.box],
      score: item.score,
      areaPct: Number(sev.areaPct.toFixed(2)),
      severity: sev.label,
      severityLevel: sev.level,
      status: prevStatuses.get(id) || 'pending',
    };
  });
}

function roadCondition(candidates) {
  const active = candidates.filter(c => c.status !== 'rejected');
  const totalArea = active.reduce((sum, c) => sum + (c.areaPct || 0), 0);
  const highCount = active.filter(c => c.severityLevel === 'high' || c.severity === 'High').length;
  if (active.length === 0) return { label: 'Good', cls: 'cond-good', totalArea };
  if (highCount >= 2 || totalArea >= 12) return { label: 'Severe', cls: 'cond-severe', totalArea };
  if (highCount >= 1 || active.length >= 3 || totalArea >= 5) return { label: 'Moderate', cls: 'cond-moderate', totalArea };
  return { label: 'Fair', cls: 'cond-fair', totalArea };
}

function resetBeforeLoad(mode, name) {
  state.record = null;
  state.rawDetections = null;
  state.selected = null;
  state.undo = null;
  state.dirty = false;
  state.conflict = false;
  state.ready = false;
  state.mode = mode;
  state.outlines = true;
  $('outlines').setAttribute('aria-pressed', 'true');
  $('filename').textContent = name;
  $('dimensions').textContent = 'Loading image…';
  $('source').textContent = sourceLabel[mode] || 'Image';
  $('readyTitle').textContent = 'Ready to inspect';
  $('readyText').textContent = 'Run pothole detection on this road photo, or drag and drop / paste (Ctrl+V) a new image.';
  $('analyze').lastElementChild.textContent = 'Run Detection';
  draw();
}

async function loadPhoto(src, mode, name, previous = null) {
  setBusy(true);
  tell('');
  resetBeforeLoad(mode, name);
  try {
    photo.src = src;
    await photo.decode();
    state.ready = true;
    photo.alt = 'Road surface photo: ' + name;
    $('dimensions').textContent = `${photo.naturalWidth} × ${photo.naturalHeight} px`;
    state.record = previous;
    if (previous) {
      state.selected = previous.candidates[0]?.id ?? null;
      if (previous.threshold) {
        state.threshold = previous.threshold;
        $('thresholdSlider').value = Math.round(previous.threshold * 100);
        $('thresholdValue').textContent = Math.round(previous.threshold * 100) + '%';
      }
      if (previous.iou) {
        state.iou = previous.iou;
        $('iouSlider').value = Math.round(previous.iou * 100);
        $('iouValue').textContent = Math.round(previous.iou * 100) + '%';
      }
    }
  } catch {
    $('readyTitle').textContent = 'Could not load image';
    $('readyText').textContent = 'Choose a valid JPEG, PNG, or WebP road photo.';
    $('dimensions').textContent = 'Unavailable';
    tell('Failed to open the selected image.');
  } finally {
    setBusy(false);
    draw();
  }
}

function readFileAsDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error('Could not read the selected file.'));
    reader.readAsDataURL(file);
  });
}

async function imageDataForRecord() {
  if (photo.src.startsWith('data:')) return photo.src;
  const response = await fetch(photo.src);
  if (!response.ok) throw new Error('Could not read image data.');
  return readFileAsDataURL(await response.blob());
}

function svgEl(tag, attrs) {
  const el = document.createElementNS(svgNS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

function boxColor(candidate, isSelected) {
  if (isSelected) return '#38bdf8';
  if (candidate.status === 'confirmed') return '#10b981';
  if (candidate.status === 'rejected') return '#94a3b8';
  return '#f59e0b';
}

function drawBoxes() {
  const layer = $('boxes');
  layer.replaceChildren();
  if (!state.record || !state.outlines) return;
  const { width, height, candidates } = state.record;
  layer.setAttribute('viewBox', `0 0 ${width} ${height}`);
  const unit = Math.max(width / 700, height / 550, 0.35);

  const sorted = [...candidates].sort((a, b) => Number(a.id === state.selected) - Number(b.id === state.selected));
  for (const c of sorted) {
    const [x, y, right, bottom] = c.box;
    const isSelected = c.id === state.selected;
    const color = boxColor(c, isSelected);
    const group = svgEl('g', {
      class: 'box-group',
      opacity: c.status === 'rejected' && !isSelected ? '0.45' : '1',
    });

    const rectAttrs = {
      class: 'hit-rect',
      x,
      y,
      width: Math.max(1, right - x),
      height: Math.max(1, bottom - y),
      fill: isSelected ? 'rgba(56, 189, 248, 0.14)' : 'rgba(0,0,0,0.01)',
      stroke: color,
      'stroke-width': isSelected ? 3.5 : 2.2,
      'vector-effect': 'non-scaling-stroke',
    };
    if (c.status === 'rejected') rectAttrs['stroke-dasharray'] = '6 4';
    group.append(svgEl('rect', rectAttrs));

    const idxLabel = '#' + String(candidates.indexOf(c) + 1).padStart(2, '0');
    const tagText = `${idxLabel} · ${Math.round(c.score * 100)}%`;
    const tagH = 20 * unit;
    const tagW = (tagText.length * 6.8 + 12) * unit;
    const tagY = y - tagH >= 0 ? y - tagH : Math.min(height - tagH, y);
    const tagX = Math.min(Math.max(0, x), Math.max(0, width - tagW));

    group.append(svgEl('rect', { x: tagX, y: tagY, width: tagW, height: tagH, rx: 3 * unit, fill: color }));
    const text = svgEl('text', {
      x: tagX + 6 * unit,
      y: tagY + 14 * unit,
      fill: '#0f172a',
      'font-size': 11.5 * unit,
      'font-weight': 700,
      'font-family': 'system-ui, sans-serif',
    });
    text.textContent = tagText;
    group.append(text);

    group.addEventListener('click', () => {
      state.selected = c.id;
      draw();
    });
    layer.append(group);
  }
}

async function persist() {
  if (!state.record) return false;
  state.dirty = true;
  try {
    await saveInspection(state.record);
    state.dirty = false;
    await refreshHistory();
    return true;
  } catch (error) {
    if (error.name === 'InspectionConflictError') {
      state.conflict = true;
      await refreshHistory();
      tell('A newer revision was saved in another tab. Reopen from History.');
      return false;
    }
    tell('Save failed: ' + describe(error));
    return false;
  }
}

function statusText(status) {
  return ({ pending: 'Pending', confirmed: 'Confirmed', rejected: 'Dismissed' })[status] ?? 'Pending';
}

function draw() {
  const record = state.record;
  $('ready').hidden = !!record;
  $('results').hidden = !record;
  $('reviewTotal').hidden = !record;
  $('outlines').disabled = !record?.candidates.length;
  $('downloadPng').disabled = state.busy || !record;
  $('exportPngBtn').disabled = state.busy || !record;
  drawBoxes();
  if (!record) return;

  const candidates = record.candidates;
  const reviewed = candidates.filter(c => c.status !== 'pending').length;
  const cond = roadCondition(candidates);

  $('reviewTotal').textContent = `${candidates.length} found`;
  $('metricCount').textContent = String(candidates.filter(c => c.status !== 'rejected').length);
  $('metricArea').textContent = cond.totalArea.toFixed(1) + '%';
  $('metricCondition').textContent = cond.label;
  $('metricCondition').className = 'metric-value ' + cond.cls;
  $('metricSpeed').textContent = record.elapsedMs ? record.elapsedMs + ' ms' : '—';

  $('summary').textContent = candidates.length
    ? `${reviewed} of ${candidates.length} reviewed`
    : 'No potholes detected at current threshold.';

  const list = $('candidateList');
  list.replaceChildren();
  candidates.forEach((c, index) => {
    const isSelected = c.id === state.selected;
    const row = make('button', undefined, 'candidate' + (isSelected ? ' selected' : ''));
    row.type = 'button';
    row.addEventListener('click', () => {
      state.selected = c.id;
      draw();
    });

    const w = Math.round(c.box[2] - c.box[0]);
    const h = Math.round(c.box[3] - c.box[1]);
    const left = make('div', undefined, 'cand-left');
    const title = make('div', undefined, 'cand-title');
    title.append(
      make('span', `#${String(index + 1).padStart(2, '0')} ${c.label || 'Pothole'}`),
      make('span', `${(c.score * 100).toFixed(1)}%`, 'mono')
    );
    const sub = make('span', `${w}×${h} px · ${Number(c.areaPct || 0).toFixed(2)}% area`, 'cand-sub mono');
    left.append(title, sub);

    const right = make('div', undefined, 'cand-right');
    const sevLvl = (c.severityLevel || c.severity || 'low').toLowerCase();
    right.append(
      make('span', c.severity || 'Low', `sev-pill sev-${sevLvl}`),
      make('span', statusText(c.status), `status-pill status-${c.status}`)
    );

    row.append(left, right);
    list.append(row);
  });

  const decisions = $('decision');
  decisions.replaceChildren();
  const selected = candidates.find(c => c.id === state.selected);
  if (selected) {
    const confirmBtn = makeBtn('Confirm', () => review('confirmed'), 'btn btn-confirm btn-sm', 'check');
    const rejectBtn = makeBtn('Dismiss', () => review('rejected'), 'btn btn-dismiss btn-sm', 'x');
    confirmBtn.disabled = state.busy || selected.status === 'confirmed';
    rejectBtn.disabled = state.busy || selected.status === 'rejected';
    decisions.append(confirmBtn, rejectBtn);
    if (state.undo) {
      decisions.append(makeBtn('Undo', undoReview, 'btn btn-ghost btn-sm', 'undo'));
    }
  }
  if (state.conflict) {
    decisions.replaceChildren(
      makeBtn('Load latest revision', async () => {
        await refreshHistory();
        const latest = state.history.find(item => item.id === record.id);
        if (latest) await loadPhoto(latest.imageData, 'history', latest.name, structuredClone(latest));
      }, 'btn btn-primary btn-sm')
    );
  } else if (state.dirty) {
    decisions.append(
      makeBtn('Retry save', async () => {
        if (state.busy) return;
        setBusy(true);
        if (await persist()) tell('Inspection saved.');
        setBusy(false);
        draw();
      }, 'btn btn-primary btn-sm')
    );
  }

  $('export').disabled = state.busy || state.dirty;
}

async function review(status) {
  if (state.busy || !state.record) return;
  const candidate = state.record.candidates.find(c => c.id === state.selected);
  if (!candidate || candidate.status === status) return;
  setBusy(true);
  state.undo = { id: candidate.id, status: candidate.status };
  candidate.status = status;
  await persist();
  setBusy(false);
  draw();
}

async function undoReview() {
  if (state.busy || !state.undo || !state.record) return;
  setBusy(true);
  const candidate = state.record.candidates.find(c => c.id === state.undo.id);
  if (candidate) {
    candidate.status = state.undo.status;
    state.selected = candidate.id;
  }
  state.undo = null;
  await persist();
  setBusy(false);
  draw();
}

async function runAnalysis() {
  if (!state.ready || state.busy) return;
  setBusy(true);
  tell('Running ' + activeModelName() + '…');
  $('analyze').lastElementChild.textContent = 'Analyzing…';
  try {
    await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
    const { raw, elapsedMs } = await infer(photo, state.modelId);
    state.rawDetections = raw;
    const candidates = buildCandidates(raw, photo.naturalWidth, photo.naturalHeight, state.threshold, state.iou);
    const record = {
      id: crypto.randomUUID?.() ?? 'rw-' + Date.now() + '-' + Math.random().toString(36).slice(2, 7),
      name: $('filename').textContent,
      imageData: await imageDataForRecord(),
      width: photo.naturalWidth,
      height: photo.naturalHeight,
      createdAt: new Date().toISOString(),
      model: activeModelName(),
      threshold: state.threshold,
      iou: state.iou,
      elapsedMs,
      candidates,
    };
    state.record = record;
    state.selected = record.candidates[0]?.id ?? null;
    state.undo = null;
    if (await persist()) {
      tell(`Analysis complete in ${elapsedMs} ms — ${candidates.length} detection${candidates.length === 1 ? '' : 's'} found.`);
    }
  } catch (error) {
    tell('Detection failed: ' + describe(error));
  } finally {
    $('analyze').lastElementChild.textContent = 'Run Detection';
    setBusy(false);
    draw();
  }
}

function applyLiveFilters() {
  if (!state.record || !state.rawDetections) return;
  state.record.threshold = state.threshold;
  state.record.iou = state.iou;
  state.record.candidates = buildCandidates(
    state.rawDetections,
    state.record.width,
    state.record.height,
    state.threshold,
    state.iou
  );
  state.selected = state.record.candidates[0]?.id ?? null;
  state.undo = null;
  draw();
  persist();
}

function downloadAnnotatedPng() {
  if (!state.record) return;
  const { width, height, candidates } = state.record;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(photo, 0, 0, width, height);

  const unit = Math.max(width / 700, height / 550, 0.5);
  candidates.forEach((c, idx) => {
    if (c.status === 'rejected') return;
    const [x, y, right, bottom] = c.box;
    const color = c.status === 'confirmed' ? '#10b981' : '#f59e0b';
    ctx.lineWidth = Math.max(2, 3 * unit);
    ctx.strokeStyle = color;
    ctx.strokeRect(x, y, right - x, bottom - y);

    const text = `#${String(idx + 1).padStart(2, '0')} ${c.label || 'Pothole'} ${Math.round(c.score * 100)}% (${c.severity || 'Low'})`;
    ctx.font = `700 ${Math.round(13 * unit)}px system-ui, sans-serif`;
    const padX = 7 * unit;
    const tagH = 22 * unit;
    const tagW = ctx.measureText(text).width + padX * 2;
    const tagY = y - tagH >= 0 ? y - tagH : y;
    const tagX = Math.min(Math.max(0, x), Math.max(0, width - tagW));

    ctx.fillStyle = color;
    ctx.fillRect(tagX, tagY, tagW, tagH);
    ctx.fillStyle = '#0f172a';
    ctx.fillText(text, tagX + padX, tagY + 15.5 * unit);
  });

  canvas.toBlob(blob => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const base = state.record.name.replace(/\.[^.]+$/, '');
    a.href = url;
    a.download = `${base}-annotated.png`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }, 'image/png');
}

async function handleImageFile(file) {
  if (!file) return;
  if (file.size > 16 * 1024 * 1024) {
    tell('Image exceeds 16 MB. Use a smaller photo.');
    return;
  }
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
    tell('Please upload a JPEG, PNG, or WebP road photo.');
    return;
  }
  setBusy(true);
  try {
    await loadPhoto(await readFileAsDataURL(file), 'upload', file.name);
  } catch (error) {
    tell(describe(error));
    setBusy(false);
  }
}

// Event bindings
$('analyze').onclick = runAnalysis;
$('reanalyze').onclick = runAnalysis;
$('upload').onclick = () => { if (!state.busy) $('file').click(); };
$('file').onchange = async e => {
  const file = e.target.files[0];
  e.target.value = '';
  await handleImageFile(file);
};

$('loadModelBtn').onclick = () => { if (!state.busy) $('modelFile').click(); };
$('modelFile').onchange = async e => {
  const file = e.target.files[0];
  e.target.value = '';
  if (!file) return;
  setBusy(true);
  tell('Loading custom ONNX model…');
  try {
    const buffer = await file.arrayBuffer();
    const entry = await registerCustomModel(file.name, buffer);
    state.modelId = entry.id;
    syncModelDropdown();
    tell(`Loaded model "${entry.name}". Click Run Detection or Re-run model to test it.`);
  } catch (err) {
    tell('Could not load ONNX model: ' + describe(err));
  } finally {
    setBusy(false);
  }
};

$('modelSelect').onchange = e => {
  state.modelId = e.target.value;
  if (state.record) tell('Model switched. Click "Re-run model" to analyze with ' + activeModelName() + '.');
};

$('thresholdSlider').oninput = e => {
  const pct = Number(e.target.value);
  state.threshold = pct / 100;
  $('thresholdValue').textContent = pct + '%';
  applyLiveFilters();
};

$('iouSlider').oninput = e => {
  const pct = Number(e.target.value);
  state.iou = pct / 100;
  $('iouValue').textContent = pct + '%';
  applyLiveFilters();
};

$('demoChoice').onchange = async e => {
  const key = e.target.value;
  e.target.value = '';
  if (!samples[key] || state.busy) return;
  displayView('workspace');
  await loadPhoto(samples[key].src, 'demo', samples[key].name);
};

$('outlines').onclick = () => {
  state.outlines = !state.outlines;
  $('outlines').setAttribute('aria-pressed', String(state.outlines));
  drawBoxes();
};

$('downloadPng').onclick = downloadAnnotatedPng;
$('exportPngBtn').onclick = downloadAnnotatedPng;

$('expand').onclick = async () => {
  const viewer = $('viewer');
  if (document.fullscreenElement) { await document.exitFullscreen(); return; }
  if (viewer.classList.contains('expanded')) { viewer.classList.remove('expanded'); return; }
  try { await viewer.requestFullscreen(); } catch { viewer.classList.add('expanded'); }
};

$('export').onclick = async () => {
  if (!state.record || state.dirty) return;
  try {
    await exportInspection(state.record);
    tell('CSV report exported.');
  } catch (error) {
    tell('Export failed: ' + describe(error));
  }
};

// Drag & Drop + Clipboard Paste
const viewerEl = $('viewer');
viewerEl.addEventListener('dragover', e => {
  e.preventDefault();
  $('dropOverlay').hidden = false;
});
viewerEl.addEventListener('dragleave', e => {
  if (!viewerEl.contains(e.relatedTarget)) $('dropOverlay').hidden = true;
});
viewerEl.addEventListener('drop', async e => {
  e.preventDefault();
  $('dropOverlay').hidden = true;
  if (state.busy) return;
  const file = e.dataTransfer?.files?.[0];
  if (file) await handleImageFile(file);
});

window.addEventListener('paste', async e => {
  if (state.busy) return;
  const item = [...(e.clipboardData?.items || [])].find(i => i.type.startsWith('image/'));
  if (!item) return;
  const file = item.getAsFile();
  if (file) await handleImageFile(file);
});

// Keyboard shortcuts
document.addEventListener('keydown', e => {
  if (e.target.matches('input, select, textarea')) return;
  if (e.key === 'Escape' && $('viewer').classList.contains('expanded')) {
    $('viewer').classList.remove('expanded');
  } else if (e.key.toLowerCase() === 'o' && state.record?.candidates.length) {
    $('outlines').click();
  } else if (e.key.toLowerCase() === 'c' && state.selected) {
    review('confirmed');
  } else if (e.key.toLowerCase() === 'd' && state.selected) {
    review('rejected');
  } else if (e.key.toLowerCase() === 'z' && state.undo) {
    undoReview();
  }
});

// History view
function renderHistory() {
  const entries = $('historyList');
  entries.replaceChildren();
  $('historyCount').textContent = state.history.length;
  $('historyCount').hidden = !state.history.length;

  if (!state.history.length) {
    entries.append(make('p', 'No saved inspections yet. Analyze a road photo to create your first record.', 'empty-history'));
    return;
  }

  for (const record of state.history) {
    const row = make('div', undefined, 'history-entry');
    const left = make('div', undefined, 'history-left');
    const thumb = make('img', undefined, 'history-thumb');
    thumb.src = record.imageData;
    thumb.alt = record.name;

    const cond = roadCondition(record.candidates || []);
    const info = make('div', undefined, 'history-info');
    info.append(
      make('strong', record.name),
      make('small', `${new Date(record.createdAt).toLocaleString()} · ${record.candidates.length} detections · Condition: ${cond.label}`)
    );
    left.append(thumb, info);

    const actions = make('div', undefined, 'history-actions');
    actions.append(
      makeBtn('Open', async () => {
        displayView('workspace');
        await loadPhoto(record.imageData, 'history', record.name, structuredClone(record));
      }, 'btn btn-ghost btn-sm'),
      makeBtn('', async () => {
        await deleteInspection(record.id);
        await refreshHistory();
      }, 'btn btn-danger-ghost btn-sm', 'trash')
    );

    row.append(left, actions);
    entries.append(row);
  }
}

async function refreshHistory() {
  try {
    state.history = await listInspections();
    renderHistory();
  } catch (error) {
    $('historyList').replaceChildren(make('p', 'Could not load history: ' + describe(error), 'empty-history'));
  }
}

function displayView(view) {
  $('workspace').hidden = view !== 'workspace';
  $('history').hidden = view !== 'history';
  for (const tab of document.querySelectorAll('.nav-tab')) {
    const active = tab.dataset.view === view;
    tab.classList.toggle('active', active);
    if (active) tab.setAttribute('aria-current', 'page');
    else tab.removeAttribute('aria-current');
  }
  tell('');
}

for (const tab of document.querySelectorAll('.nav-tab')) {
  tab.onclick = async () => {
    if (state.busy) return;
    displayView(tab.dataset.view);
    if (tab.dataset.view === 'history') await refreshHistory();
  };
}

$('back').onclick = () => displayView('workspace');
$('brandLink').onclick = e => {
  e.preventDefault();
  if (!state.busy) displayView('workspace');
};

syncModelDropdown();
displayView(location.hash === '#history' ? 'history' : 'workspace');
Promise.all([loadPhoto(samples.wet.src, 'demo', samples.wet.name), refreshHistory()]);
