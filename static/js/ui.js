/**
 * ui.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Pure UI layer:
 *   • Tabs, ripple, comparison slider
 *   • Drag-and-drop + file input
 *   • Dashboard panel updates (enhance / detect / depth / water)
 *   • Session stats
 *   • Gallery, batch, model status
 *
 * Imports from api.js — no raw fetch() calls here.
 * All animation is delegated to animation.js via window._setRadarDetections.
 * ─────────────────────────────────────────────────────────────────────────────
 */

import { analyzeAll, enhance, fetchStatus } from './api.js';

/* ═══════════════════════════════════════════════════════════
   HELPERS
═══════════════════════════════════════════════════════════ */
function $(id)   { return document.getElementById(id); }
function el(sel) { return document.querySelector(sel); }

function setText(id, val) {
  const e = $(id); if (e) e.textContent = val ?? '—';
}
function setHTML(id, val) {
  const e = $(id); if (e) e.innerHTML = val;
}
function show(id, display = 'block') {
  const e = $(id); if (e) e.style.display = display;
}
function hide(id) {
  const e = $(id); if (e) e.style.display = 'none';
}

/* ═══════════════════════════════════════════════════════════
   TABS
═══════════════════════════════════════════════════════════ */
function initTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      btn.classList.add('active');
      $(`${btn.dataset.tab}-tab`).classList.add('active');
    });
  });
}

/* ═══════════════════════════════════════════════════════════
   RIPPLE
═══════════════════════════════════════════════════════════ */
function initRipple() {
  document.querySelectorAll('.ripple-container').forEach(el => {
    el.addEventListener('click', e => {
      const r = el.getBoundingClientRect();
      const w = Math.max(r.width, r.height);
      const rip = document.createElement('span');
      rip.className = 'ripple-wave';
      Object.assign(rip.style, {
        width:  w + 'px', height: w + 'px',
        left:   (e.clientX - r.left - w / 2) + 'px',
        top:    (e.clientY - r.top  - w / 2) + 'px',
      });
      el.appendChild(rip);
      setTimeout(() => rip.remove(), 800);
    });
  });
}



/* ═══════════════════════════════════════════════════════════
   PIPELINE STATUS STRIP
═══════════════════════════════════════════════════════════ */
function setPipe(id, state) {
  const e = $(`pipe-${id}`);
  if (e) e.className = 'pipeline-step' + (state ? ` ${state}` : '');
}

/* ═══════════════════════════════════════════════════════════
   SESSION STATS
═══════════════════════════════════════════════════════════ */
const session = { count: 0, psnr: 0, ssim: 0, uiqm: 0, time: 0 };

function updateSessionStats(data) {
  session.count++;
  if (data.psnr)            session.psnr += data.psnr;
  if (data.ssim)            session.ssim += data.ssim;
  if (data.uiqm)            session.uiqm += data.uiqm;
  if (data.processing_time) session.time += data.processing_time;

  const n = session.count;
  setText('totalProcessed',  n);
  setText('avgPSNR',         (session.psnr / n).toFixed(2));
  setText('statsProcessed',  n);
  setText('statsAvgPSNR',    (session.psnr / n).toFixed(2));
  setText('statsAvgSSIM',    (session.ssim / n).toFixed(4));
  setText('statsAvgUIQM',    session.uiqm > 0 ? (session.uiqm / n).toFixed(3) : '—');
  setText('statsTotalTime',  session.time.toFixed(1) + 's');
}

/* ═══════════════════════════════════════════════════════════
   DASHBOARD PANEL UPDATERS
═══════════════════════════════════════════════════════════ */
function applyEnhance(file, data) {
  setPipe('enhance', data.success ? 'done' : 'error');
  if (!data.success) return;

  $('imgOriginal').src = URL.createObjectURL(file);
  $('imgEnhancedHybrid').src = data.enhanced_image_hybrid;
  $('imgEnhancedOpencv').src = data.enhanced_image_opencv;
  setText('imgSize',    data.size);
  setText('psnrScore',  data.psnr   ? data.psnr.toFixed(2)   : '—');
  setText('ssimScore',  data.ssim   ? data.ssim.toFixed(4)   : '—');
  setText('uiqmScore',  data.uiqm   ? data.uiqm.toFixed(3)   : '—');
  setText('uciqeScore', data.uciqe  ? data.uciqe.toFixed(3)  : '—');
  setText('epsScore',   data.eps    ? data.eps.toFixed(3)     : '—');
  setText('procTime',   data.processing_time ? data.processing_time.toFixed(2) + 's' : '—');

  if (data.heatmap) {
    $('heatmapImg').src = data.heatmap;
    show('heatmapSection');
  }
  window._lastEnhanced = data;
  updateSessionStats(data);
}

function applyDetect(data) {
  setPipe('detect', data.success ? 'done' : 'error');
  hide('detectPending');

  if (data.annotated_image) {
    const img = $('annotatedImg');
    img.src = data.annotated_image; img.style.display = 'block';
  }

  hide('threatPending');
  show('threatPanel');

  const score = data.threat_score || 0;
  const ring  = $('threatRing');
  const badge = $('alertBadge');
  setText('threatScoreVal', Math.round(score));
  ring.className  = 'threat-score-ring';
  badge.className = 'alert-badge';
  if (data.alert_level === 'Red') {
    ring.classList.add('red'); badge.classList.add('red'); badge.textContent = '⚠ RED ALERT';
  } else if (data.alert_level === 'Yellow') {
    ring.classList.add('yellow'); badge.classList.add('yellow'); badge.textContent = '⚡ YELLOW ALERT';
  } else {
    badge.textContent = '✔ GREEN – CLEAR';
  }

  // Detection list
  const dl = $('detectList');
  dl.innerHTML = '';
  (data.detections || []).forEach(d => {
    const item = document.createElement('div');
    item.className = 'detect-item' + (d.category === 'Security Threat' ? ' di-threat' : '');
    item.innerHTML = `<div><div class="di-class">${d.display_class || d.class}</div><div class="di-cat">${d.category}</div></div><div class="di-conf">${(d.confidence * 100).toFixed(0)}%</div>`;
    dl.appendChild(item);
  });
  if (!(data.detections && data.detections.length))
    dl.innerHTML = '<div style="opacity:.45;font-size:12px;padding:6px;">No objects detected</div>';

  // Recommendations
  const rl = $('recommendList');
  rl.innerHTML = '';
  (data.recommendations || []).forEach(r => {
    const e = document.createElement('div');
    e.className = 'recommend-item'; e.textContent = r;
    rl.appendChild(e);
  });

  // Hand off to animated radar (animation.js owns the canvas tick)
  if (window._setRadarDetections) window._setRadarDetections(data.detections || []);
}

function applyDepth(data) {
  setPipe('depth', data.success ? 'done' : 'error');
  hide('depthPending');
  if (data.depth_map) {
    const img = $('depthMapImg');
    img.src = data.depth_map; img.style.display = 'block';
  }
  setText('avgDepth', data.average_depth != null ? (data.average_depth * 100).toFixed(1) + '%' : '—');
  const zones = data.object_distances || [];
  const total = zones.reduce((s, z) => s + z.pixels, 0) || 1;
  zones.forEach(z => {
    const id = z.zone === 'Near' ? 'dzNear' : z.zone === 'Mid' ? 'dzMid' : 'dzFar';
    const e = $(id); if (e) e.style.flex = String(z.pixels / total * 10);
  });
}

function applyWater(data) {
  setPipe('water', data.success ? 'done' : 'error');
  hide('waterPending');
  show('waterPanel');

  setHTML('visibilityVal', (data.visibility_range_meters || '—') + '<small>m</small>');
  setText('visibilityMetric', (data.visibility_range_meters || '—') + 'm');
  setText('turbidityLevel',   data.turbidity_level || '—');
  setText('waterType',        data.water_type || '—');

  const gauge = $('turbidityGauge');
  if (gauge) gauge.style.setProperty('--gauge-pos', (data.turbidity_index * 100).toFixed(0) + '%');

  const att = data.attenuation || {};
  ['Red', 'Green', 'Blue'].forEach(c => {
    const v    = att[c.toLowerCase()] || 0;
    const fill = $('att' + c); if (fill) fill.style.width = (v * 100).toFixed(0) + '%';
    const lbl  = $('att' + c + 'Val'); if (lbl) lbl.textContent = (v * 100).toFixed(0) + '%';
  });
}

/* ═══════════════════════════════════════════════════════════
   MAIN UPLOAD HANDLER — fires all 4 APIs concurrently
═══════════════════════════════════════════════════════════ */
function handleFile(file) {
  hide('loadingOverlay');
  hide('dropZone');
  show('resultsSection');

  // Reset pipeline
  ['enhance', 'detect', 'depth', 'water'].forEach(p => setPipe(p, 'active'));

  // Reset panel states
  ['detectPending', 'depthPending', 'waterPending', 'threatPending'].forEach(id => show(id, 'flex'));
  ['annotatedImg', 'depthMapImg'].forEach(id => hide(id));
  ['waterPanel', 'threatPanel', 'heatmapSection'].forEach(id => hide(id));
  if (window._setRadarDetections) window._setRadarDetections([]);

  analyzeAll(file, {
    onEnhance: data  => applyEnhance(file, data),
    onDetect:  data  => applyDetect(data),
    onDepth:   data  => applyDepth(data),
    onWater:   data  => applyWater(data),
    onError:   _err  => { /* individual pipe already marked error by each handler */ },
  });
}

/* ═══════════════════════════════════════════════════════════
   DROP ZONE
═══════════════════════════════════════════════════════════ */
function initDropZone() {
  const dropZone  = $('dropZone');
  const fileInput = $('fileInput');
  if (!dropZone) return;

  dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('dragging'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragging'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault(); dropZone.classList.remove('dragging');
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener('change', () => { if (fileInput.files[0]) handleFile(fileInput.files[0]); });
}

/* ═══════════════════════════════════════════════════════════
   ACTIONS (download / gallery / reset)
═══════════════════════════════════════════════════════════ */
function initActions() {
  window.downloadImage = () => {
    if (!window._lastEnhanced) return;
    const a = document.createElement('a');
    a.href = window._lastEnhanced.enhanced_image;
    a.download = 'enhanced.png';
    a.click();
  };

  window.resetUpload = () => {
    hide('resultsSection');
    show('dropZone');
    $('fileInput').value = '';
    window._lastEnhanced = null;
    if (window._setRadarDetections) window._setRadarDetections([]);
  };

  window.saveToGallery = () => {
    if (!window._lastEnhanced) return;
    const data = window._lastEnhanced;
    const grid  = $('galleryGrid');
    const empty = grid.querySelector('.empty-state');
    if (empty) empty.remove();
    const item = document.createElement('div');
    item.className = 'gallery-item';
    item.innerHTML = `
      <img src="${data.enhanced_image}" alt="Enhanced">
      <div class="gallery-meta">
        <div class="gallery-time">${new Date().toLocaleTimeString()}</div>
        <div class="gallery-scores">
          <span>PSNR ${data.psnr ? data.psnr.toFixed(2) : '—'}</span>
          <span>SSIM ${data.ssim ? data.ssim.toFixed(4) : '—'}</span>
        </div>
      </div>`;
    grid.prepend(item);
  };

  window.clearGallery = () => {
    setHTML('galleryGrid', `<div class="empty-state">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" width="56" height="56">
        <rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/>
      </svg><p>No enhanced images yet.</p></div>`);
  };
}

/* ═══════════════════════════════════════════════════════════
   BATCH TAB
═══════════════════════════════════════════════════════════ */
function initBatch() {
  const batchInput = $('batchFileInput');
  if (!batchInput) return;

  batchInput.addEventListener('change', () => {
    const files    = Array.from(batchInput.files);
    if (!files.length) return;

    const progress  = $('batchProgress');
    const fill      = $('progressFill');
    const progressT = $('progressText');
    const results   = $('batchResults');
    const actions   = $('batchActions');

    results.innerHTML = '';
    show('batchProgress');
    hide('batchActions');

    let done = 0;
    window._batchResults = [];

    files.forEach(file => {
      enhance(file).then(data => {
        done++;
        const pct = Math.round((done / files.length) * 100);
        fill.style.width      = pct + '%';
        progressT.textContent = `${done}/${files.length}`;
        if (!data.error) {
          window._batchResults.push(data);
          const item = document.createElement('div');
          item.className = 'batch-item';
          item.innerHTML = `
            <img src="${data.enhanced_image}" alt="">
            <div class="batch-meta">
              <div class="name">${file.name}</div>
              <div class="scores">
                <span>PSNR ${data.psnr ? data.psnr.toFixed(1) : '—'}</span>
                <span>SSIM ${data.ssim ? data.ssim.toFixed(3) : '—'}</span>
              </div>
            </div>`;
          results.appendChild(item);
        }
        if (done === files.length) show('batchActions', 'flex');
      });
    });
  });

  window.downloadAllBatch = () => {
    (window._batchResults || []).forEach((d, i) => {
      setTimeout(() => {
        const a = document.createElement('a');
        a.href = d.enhanced_image; a.download = `enhanced_${i + 1}.png`; a.click();
      }, i * 300);
    });
  };

  window.clearBatch = () => {
    $('batchResults').innerHTML = '';
    hide('batchProgress'); hide('batchActions');
    window._batchResults = [];
  };
}

/* ═══════════════════════════════════════════════════════════
   MODEL STATUS CHECK
═══════════════════════════════════════════════════════════ */
function initStatus() {
  fetchStatus()
    .then(d => {
      const pill = $('status');
      if (d.model_loaded) {
        pill.classList.add('ready');
        pill.querySelector('.status-text').textContent = 'Model Ready';
        setText('deviceInfo', d.device);
        if (d.parameters) setText('paramCount', Number(d.parameters).toLocaleString());
      } else {
        pill.classList.add('error');
        pill.querySelector('.status-text').textContent = 'Model Not Loaded';
      }
      if (d.modules) {
        const mp = $('modulePills');
        if (mp) {
          mp.innerHTML = '';
          [
            { key: 'yolo',   label: 'YOLOv8 Detection' },
            { key: 'midas',  label: 'MiDaS Depth' },
            { key: 'opencv', label: 'OpenCV Video' },
          ].forEach(m => {
            const e = document.createElement('div');
            e.className = 'module-pill ' + (d.modules[m.key] ? 'ok' : 'warn');
            e.innerHTML = `<span class="module-pill-dot"></span>${m.label}`;
            mp.appendChild(e);
          });
        }
      }
    })
    .catch(() => {
      const pill = $('status');
      pill.classList.add('error');
      pill.querySelector('.status-text').textContent = 'Offline';
    });
}

/* ═══════════════════════════════════════════════════════════
   BOOT
═══════════════════════════════════════════════════════════ */
export function initUI() {
  initTabs();
  initRipple();
  initDropZone();
  initActions();
  initBatch();
  initStatus();
}