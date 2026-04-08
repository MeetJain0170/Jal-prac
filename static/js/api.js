/**
 * api.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Thin wrappers around every Flask endpoint.
 * All functions return Promises.  Callers handle UI — no DOM touches here.
 * ─────────────────────────────────────────────────────────────────────────────
 */

const BASE = '';  // Same-origin Flask server.  Change to 'http://localhost:5500' if needed.

function fd(file) {
  const f = new FormData();
  f.append('image', file);
  return f;
}

/**
 * GET /api/status
 * @returns {Promise<{model_loaded, device, parameters, modules}>}
 */
export async function fetchStatus() {
  const r = await fetch(`${BASE}/api/status`);
  return r.json();
}

/**
 * POST /api/enhance
 * @param {File} file
 * @returns {Promise<{success, enhanced_image, psnr, ssim, uiqm, uciqe, eps, size, processing_time, heatmap}>}
 */
export async function enhance(file) {
  const r = await fetch(`${BASE}/api/enhance`, { method: 'POST', body: fd(file) });
  return r.json();
}

/**
 * POST /api/detect
 * @param {File} file
 * @returns {Promise<{success, detections, annotated_image, threat_score, alert_level, recommendations}>}
 */
export async function detect(file) {
  const r = await fetch(`${BASE}/api/detect`, { method: 'POST', body: fd(file) });
  return r.json();
}

/**
 * POST /api/depth
 * @param {File} file
 * @returns {Promise<{success, depth_map, average_depth, depth_range, object_distances}>}
 */
export async function depth(file) {
  const r = await fetch(`${BASE}/api/depth`, { method: 'POST', body: fd(file) });
  return r.json();
}

/**
 * POST /api/analyze-water
 * @param {File} file
 * @returns {Promise<{success, visibility_range_meters, turbidity_level, turbidity_index, water_type, attenuation}>}
 */
export async function analyzeWater(file) {
  const r = await fetch(`${BASE}/api/analyze-water`, { method: 'POST', body: fd(file) });
  return r.json();
}

/**
 * POST /api/process-video
 * @param {File} file
 * @returns {Promise<{status, output_path, frame_count, fps, resolution, processing_time_s}>}
 */
export async function processVideo(file) {
  const f = new FormData();
  f.append('video', file);
  const r = await fetch(`${BASE}/api/process-video`, { method: 'POST', body: f });
  return r.json();
}

/**
 * Run all four analysis APIs in parallel.
 * Each settles independently — use onEnhance / onDetect / onDepth / onWater callbacks
 * so the UI can update progressively without waiting for all four.
 *
 * @param {File}     file
 * @param {Object}   callbacks  { onEnhance, onDetect, onDepth, onWater, onError }
 */
export async function analyzeAll(file, { onEnhance, onDetect, onDepth, onWater, onError } = {}) {
  try {
    const wrap = (promise, cb) =>
      promise.then(data => cb && cb(data)).catch(err => onError && onError(err));

    // 1. Run enhancement FIRST
    const enhanceData = await enhance(file);
    if (onEnhance) onEnhance(enhanceData);

    let targetFile = file;

    // 2. If enhancement succeeded, grab the enhanced image string and convert it to a File blob
    if (enhanceData && enhanceData.success && enhanceData.enhanced_image_hybrid) {
      const res = await fetch(enhanceData.enhanced_image_hybrid);
      const blob = await res.blob();
      targetFile = new File([blob], "enhanced.png", { type: "image/png" });
    }

    // 3. Run all subsequent processing on the new Target File (Original or Enhanced Hybrid)
    await Promise.all([
      wrap(detect(targetFile),       onDetect),
      wrap(depth(targetFile),        onDepth),
      wrap(analyzeWater(targetFile), onWater)
    ]);
  } catch (err) {
    if (onError) onError(err);
  }
}