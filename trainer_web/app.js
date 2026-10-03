'use strict';

const I18N = {
  tr: {
    title: 'MAYA Model Eğitimi',
    subtitle: 'Veri seti seç, modeli yapılandır, eğitimi izle',
    theme: 'Tema',
    resumeTitle: 'Yarım kalan eğitim',
    resume: 'Kaldığı yerden devam et',
    clearSession: "Geçici checkpoint'i sil",
    resumeInfo: 'Adım {step}/{total} · loss {loss} · {samples} örnek · {datasets} veri seti',
    datasets: 'Veri setleri',
    selectAll: 'Tümünü seç',
    deselectAll: 'Seçimi kaldır',
    refresh: 'Yenile',
    builtin: 'Yerleşik sohbet diyalogları (~170 örnek)',
    datasetSummary: '{selected}/{total} dosya seçili · ~{samples} örnek · {size}',
    noDatasets: 'data/corpus altında .jsonl bulunamadı. Aşağıdan indirebilirsiniz.',
    downloadCode: 'Kod veri seti indir (Hugging Face)',
    downloadTurkish: 'Türkçe sohbet seti indir',
    modelConfig: 'Model yapılandırması',
    preset: 'Hazır boyut',
    custom: 'Özel',
    layers: 'Katman', hidden: 'Gizli boyut', heads: 'Head', kvHeads: 'KV head', intermediate: 'FFN boyut',
    maxSeq: 'Maks. dizi', vocab: 'Sözcük (vocab)', steps: 'Eğitim adımı (0 = otomatik)', lr: 'Öğrenme hızı (0 = otomatik)',
    stepBatchHint: 'Her adımda batch size kadar örnek işlenir.',
    ckptEvery: 'Checkpoint aralığı (adım)', seed: 'Seed',
    estimate: '~{params} parametre · FP32 {fp32} · INT8 {int8} · geçici (model + AdamW m/v) {temp} · her {every} adımda diske yazılır',
    start: 'Eğitimi başlat', stop: 'Durdur',
    progress: 'İlerleme', step: 'Eğitim adımı', loss: 'Loss', elapsed: 'Geçen süre', eta: 'Kalan (tahmini)', speed: 'Adım hızı',
    resumedFrom: 'Başlangıç adımı',
    storage: 'Depolama', freeDisk: 'Boş disk', tempCkpt: 'Geçici checkpoint', estimatedCkpt: 'Tahmini yeni ckpt',
    tokenizer: 'Tokenizer',
    diskWarning: 'Tahmini checkpoint boyutu boş disk alanından büyük! Yer açmadan eğitim tamamlanamaz.',
    logs: 'Loglar', clearLogs: 'Temizle',
    confirmDifferent: 'Yarım kalan bir eğitim var ve seçimler farklı. Başlatırsan geçici checkpoint silinir. Devam?',
    confirmClear: 'Geçici checkpoint silinsin mi?',
    errHidden: 'Gizli boyut head sayısına tam bölünmeli.',
    errKv: 'Head sayısı KV head sayısına tam bölünmeli.',
    errNoData: 'En az bir veri seti seç veya yerleşik diyalogları aç.',
    state: { idle: 'Hazır', running: 'Eğitiliyor', stopping: 'Durduruluyor', completed: 'Tamamlandı', stopped: 'Durduruldu', failed: 'Hata' },
  },
  en: {
    title: 'MAYA Model Trainer',
    subtitle: 'Pick datasets, configure the model, watch training',
    theme: 'Theme',
    resumeTitle: 'Unfinished training',
    resume: 'Resume where it left off',
    clearSession: 'Delete temp checkpoint',
    resumeInfo: 'Step {step}/{total} · loss {loss} · {samples} samples · {datasets} dataset(s)',
    datasets: 'Datasets',
    selectAll: 'Select all',
    deselectAll: 'Deselect all',
    refresh: 'Refresh',
    builtin: 'Built-in chat dialogues (~170 samples)',
    datasetSummary: '{selected}/{total} files selected · ~{samples} samples · {size}',
    noDatasets: 'No .jsonl found under data/corpus. Download below.',
    downloadCode: 'Download code dataset (Hugging Face)',
    downloadTurkish: 'Download Turkish chat dataset',
    modelConfig: 'Model configuration',
    preset: 'Preset',
    custom: 'Custom',
    layers: 'Layers', hidden: 'Hidden dim', heads: 'Heads', kvHeads: 'KV heads', intermediate: 'FFN dim',
    maxSeq: 'Max seq len', vocab: 'Vocab', steps: 'Training steps (0 = auto)', lr: 'Learning rate (0 = auto)',
    stepBatchHint: 'Each step processes the configured batch size.',
    ckptEvery: 'Checkpoint interval (steps)', seed: 'Seed',
    estimate: '~{params} params · FP32 {fp32} · INT8 {int8} · temp (model + AdamW m/v) {temp} · written to disk every {every} step(s)',
    start: 'Start training', stop: 'Stop',
    progress: 'Progress', step: 'Training step', loss: 'Loss', elapsed: 'Elapsed', eta: 'ETA', speed: 'Steps/s',
    resumedFrom: 'Starting step',
    storage: 'Storage', freeDisk: 'Free disk', tempCkpt: 'Temp checkpoint', estimatedCkpt: 'Estimated new ckpt',
    tokenizer: 'Tokenizer',
    diskWarning: 'Estimated checkpoint is larger than free disk space! Free up space first.',
    logs: 'Logs', clearLogs: 'Clear',
    confirmDifferent: 'An unfinished session exists with different settings. Starting will delete its temp checkpoint. Continue?',
    confirmClear: 'Delete the temp checkpoint?',
    errHidden: 'Hidden dim must be divisible by heads.',
    errKv: 'Heads must be divisible by KV heads.',
    errNoData: 'Select at least one dataset or enable built-in dialogues.',
    state: { idle: 'Idle', running: 'Training', stopping: 'Stopping', completed: 'Completed', stopped: 'Stopped', failed: 'Failed' },
  },
};

const CONFIG_FIELDS = ['num_layers', 'hidden_dim', 'num_heads', 'num_kv_heads', 'intermediate_dim', 'max_seq_len',
  'vocab_size', 'steps', 'learning_rate', 'batch_size', 'checkpoint_interval', 'seed'];
const FIELD_IDS = {
  num_layers: 'numLayers', hidden_dim: 'hiddenDim', num_heads: 'numHeads', num_kv_heads: 'numKvHeads',
  intermediate_dim: 'intermediateDim', max_seq_len: 'maxSeqLen', vocab_size: 'vocabSize', steps: 'steps',
  learning_rate: 'learningRate', batch_size: 'batchSize', checkpoint_interval: 'checkpointInterval', seed: 'seed',
};
// PRESET_FIELDS: model mimarisini belirleyen alanlar (preset eşleşmesi için)
// learning_rate ve batch_size preset'ten alınır ama mimariyi değiştirmez
const PRESET_FIELDS = ['num_layers', 'hidden_dim', 'num_heads', 'num_kv_heads', 'intermediate_dim', 'max_seq_len', 'vocab_size'];

const $ = (id) => document.getElementById(id);
const state = {
  lang: localStorage.getItem('maya.lang') || (navigator.language.startsWith('tr') ? 'tr' : 'en'),
  datasets: [],
  presets: [],
  selected: null, // Set of dataset paths; null until first load (then all selected)
  status: null,
  lastSeq: 0,
  wasDownloading: false,
};

// ---------- i18n / theme ----------
const t = (key, vars = {}) => {
  const text = I18N[state.lang][key] ?? I18N.en[key] ?? key;
  return typeof text === 'string' ? text.replace(/\{(\w+)\}/g, (_, k) => vars[k] ?? '') : text;
};

const applyI18n = () => {
  document.documentElement.lang = state.lang;
  document.querySelectorAll('[data-i18n]').forEach((el) => { el.textContent = t(el.dataset.i18n); });
  document.querySelectorAll('[data-i18n-title]').forEach((el) => { el.title = t(el.dataset.i18nTitle); });
  document.title = t('title');
  $('langSelect').value = state.lang;
  renderPresets();
  renderDatasets();
  if (state.status) renderStatus(state.status);
};

const applyTheme = (mode) => {
  if (mode) document.documentElement.dataset.theme = mode; else delete document.documentElement.dataset.theme;
  localStorage.setItem('maya.theme', mode || '');
};

// ---------- helpers ----------
const api = async (path, options) => {
  const response = await fetch(path, options);
  const json = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(json.error || `HTTP ${response.status}`);
  return json;
};
const postJson = (path, body) => api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body ?? {}) });

const formatBytes = (bytes) => {
  if (!bytes) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const index = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  return `${(bytes / 1024 ** index).toFixed(index >= 2 ? 2 : 0)} ${units[index]}`;
};
const formatDuration = (seconds) => {
  if (!isFinite(seconds) || seconds < 0) return '–';
  const s = Math.round(seconds);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return h ? `${h}h ${m}m` : m ? `${m}m ${sec}s` : `${sec}s`;
};
const formatCount = (n) => n >= 1e9 ? `${(n / 1e9).toFixed(2)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}K` : `${n}`;

const estimateParams = (c) => {
  const h = c.hidden_dim, heads = Math.max(1, c.num_heads), kvDim = (h / heads) * c.num_kv_heads;
  const attention = 2 * h * h + 2 * h * kvDim;
  const ffn = 3 * h * c.intermediate_dim;
  return c.num_layers * (attention + ffn + 2 * h) + 2 * c.vocab_size * h + h;
};

// ---------- config form ----------
const readConfig = () => {
  const config = { datasets: [...(state.selected ?? [])], include_builtin: $('includeBuiltin').checked };
  for (const field of CONFIG_FIELDS) {
    const numValue = Number($(FIELD_IDS[field]).value);
    // NaN ise (boş input) veya 0'dan küçükse 0 kullan, değilse gerçek değeri kullan
    // ÖNEMLİ: || 0 yerine bu kullanılır çünkü || 0, 0.005 gibi küçük değerleri de 0 yapmaz
    // Ama boş input (NaN) veya negatif değerler için 0 kullanılır
    config[field] = isNaN(numValue) || numValue < 0 ? 0 : numValue;
  }
  return config;
};

const fillConfig = (config) => {
  for (const field of CONFIG_FIELDS) if (config[field] !== undefined) $(FIELD_IDS[field]).value = config[field];
  if (config.include_builtin !== undefined) $('includeBuiltin').checked = config.include_builtin;
  if (config.datasets) state.selected = new Set(config.datasets);
  syncPresetSelect();
  renderDatasets();
  updateEstimate();
};

const validateConfig = (config) => {
  if (config.vocab_size < 262) return state.lang === 'tr'
    ? 'Tokenizer vocab en az 262 olmalı (özel tokenlar ve byte alfabesi için). Öğrenilen merge sayısı tekrarlanan metne bağlıdır.'
    : 'Tokenizer vocab must be at least 262 for special tokens and byte fallback. Learned merge count depends on repeated text.';
  if (config.hidden_dim % config.num_heads !== 0) return t('errHidden');
  if (config.num_heads % config.num_kv_heads !== 0) return t('errKv');
  if (!config.datasets.length && !config.include_builtin) return t('errNoData');
  return null;
};

const showFormError = (message) => {
  $('formError').textContent = message || '';
  $('formError').classList.toggle('hidden', !message);
};

const updateEstimate = () => {
  const config = readConfig();
  const params = estimateParams(config);
  const fp32 = params * 4;
  const tempTotal = fp32 * 3;
  $('estimate').textContent = t('estimate', {
    params: formatCount(params), fp32: formatBytes(fp32), int8: formatBytes(params),
    temp: formatBytes(tempTotal), every: config.checkpoint_interval || 1,
  });
  $('stEstimate').textContent = formatBytes(fp32);
  const free = state.status?.storage?.free_bytes ?? Infinity;
  $('storageWarning').classList.toggle('hidden', tempTotal <= free);
  showFormError(validateConfig(config));
};

const renderPresets = () => {
  const select = $('presetSelect');
  const current = select.value;
  select.innerHTML = '';
  for (const preset of state.presets) {
    const option = document.createElement('option');
    option.value = preset.id;
    option.textContent = preset.name;
    select.appendChild(option);
  }
  const custom = document.createElement('option');
  custom.value = 'custom';
  custom.textContent = t('custom');
  select.appendChild(custom);
  select.value = current || 'custom';
  syncPresetSelect();
};

const syncPresetSelect = () => {
  const config = readConfig();
  const match = state.presets.find((p) => PRESET_FIELDS.every((f) => Number(p[f]) === config[f]));
  $('presetSelect').value = match ? match.id : 'custom';
};

const applyPreset = (id) => {
  const preset = state.presets.find((p) => p.id === id);
  if (!preset) return;
  const patch = {};
  for (const field of PRESET_FIELDS) patch[field] = preset[field];
  // Preset'in kendi learning_rate ve batch_size'ını kullan
  // (eski kod learning_rate = 0 atıyordu - bu bug!)
  if (preset.learning_rate !== undefined) patch.learning_rate = preset.learning_rate;
  if (preset.batch_size !== undefined) patch.batch_size = preset.batch_size;
  fillConfig(patch);
};

// ---------- datasets ----------
const renderDatasets = () => {
  const list = $('datasetList');
  list.innerHTML = '';
  if (!state.datasets.length) {
    list.innerHTML = `<li class="muted small" style="padding:10px 4px">${t('noDatasets')}</li>`;
  }
  let samples = 0, size = 0;
  for (const dataset of state.datasets) {
    const checked = state.selected?.has(dataset.path) ?? false;
    if (checked) { samples += dataset.sample_count; size += dataset.size_bytes; }
    const item = document.createElement('li');
    item.innerHTML = `
      <label>
        <input type="checkbox" data-path="${dataset.path}" ${checked ? 'checked' : ''}>
        <span><span class="name">${dataset.name}</span><span class="desc">${dataset.description || dataset.path}</span></span>
        <span class="meta">${dataset.sample_count.toLocaleString()} · ${formatBytes(dataset.size_bytes)}</span>
      </label>`;
    list.appendChild(item);
  }
  list.querySelectorAll('input[type=checkbox]').forEach((box) => box.addEventListener('change', () => {
    if (box.checked) state.selected.add(box.dataset.path); else state.selected.delete(box.dataset.path);
    renderDatasets();
    updateEstimate();
  }));
  const selectedCount = state.selected?.size ?? 0;
  $('datasetSummary').textContent = t('datasetSummary', {
    selected: selectedCount, total: state.datasets.length, samples: samples.toLocaleString(), size: formatBytes(size),
  });
  $('selectAllBtn').textContent = selectedCount === state.datasets.length && selectedCount > 0 ? t('deselectAll') : t('selectAll');
};

const loadDatasets = async () => {
  const { datasets } = await api('/api/v1/train/datasets');
  state.datasets = datasets;
  const known = new Set(datasets.map((d) => d.path));
  state.selected = state.selected ? new Set([...state.selected].filter((p) => known.has(p))) : new Set(known);
  renderDatasets();
  updateEstimate();
};

// ---------- status / logs ----------
const configsMatch = (a, b) => a.include_builtin === b.include_builtin &&
  PRESET_FIELDS.concat(['steps', 'learning_rate', 'batch_size', 'seed']).every((f) => Number(a[f]) === Number(b[f])) &&
  [...a.datasets].sort().join('|') === [...b.datasets].sort().join('|');

const renderStatus = (status) => {
  const busy = status.state === 'running' || status.state === 'stopping';
  const badge = $('stateBadge');
  badge.textContent = t('state')[status.state] ?? status.state;
  badge.className = `badge ${status.state}`;

  const percent = status.total_steps ? (100 * status.step) / status.total_steps : 0;
  $('progressFill').style.width = `${percent.toFixed(1)}%`;
  $('statStep').textContent = `${status.step.toLocaleString()} / ${status.total_steps.toLocaleString()}`;
  $('statLoss').textContent = status.step ? status.loss.toFixed(4) : '–';
  $('statElapsed').textContent = status.elapsed_seconds ? formatDuration(status.elapsed_seconds) : '–';
  $('statSpeed').textContent = status.steps_per_second ? `${status.steps_per_second.toFixed(2)} steps/s` : '–';
  $('statEta').textContent = busy && status.steps_per_second
    ? formatDuration((status.total_steps - status.step) / status.steps_per_second) : '–';
  $('statResumed').textContent = Number.isFinite(Number(status.resumed_from))
    ? Number(status.resumed_from).toLocaleString() : '–';
  $('statusMessage').textContent = status.message || '';

  const storage = status.storage;
  $('stFree').textContent = `${formatBytes(storage.free_bytes)} / ${formatBytes(storage.total_bytes)}`;
  $('stModel').textContent = formatBytes(storage.model_bytes);
  $('stQuant').textContent = formatBytes(storage.quant_bytes);
  $('stTemp').textContent = formatBytes(storage.temp_checkpoint_bytes);
  $('stTokenizer').textContent = formatBytes(storage.tokenizer_bytes);
  const usedPercent = storage.total_bytes ? (100 * (storage.total_bytes - storage.free_bytes)) / storage.total_bytes : 0;
  const diskFill = $('diskFill');
  diskFill.style.width = `${usedPercent.toFixed(1)}%`;
  diskFill.className = `progress-fill ${usedPercent > 95 ? 'danger' : usedPercent > 85 ? 'warn' : ''}`;

  const pending = status.pending_session;
  $('resumeCard').classList.toggle('hidden', !pending || busy);
  if (pending) {
    $('resumeInfo').textContent = t('resumeInfo', {
      step: pending.completed_steps.toLocaleString(), total: pending.total_steps.toLocaleString(),
      loss: pending.last_loss.toFixed(4), samples: pending.sample_count.toLocaleString(), datasets: pending.datasets.length,
    });
  }

  $('startBtn').disabled = busy || status.downloading;
  $('stopBtn').disabled = status.state !== 'running';
  $('downloadCodeBtn').disabled = busy || status.downloading;
  $('downloadTurkishBtn').disabled = busy || status.downloading;
  document.querySelectorAll('.grid input, #presetSelect, #includeBuiltin').forEach((el) => { el.disabled = busy; });
  if (state.wasDownloading && !status.downloading) loadDatasets().catch(console.error);
  state.wasDownloading = status.downloading;
};

const pollStatus = async () => {
  try {
    const status = await api('/api/v1/train/status');
    state.status = status;
    renderStatus(status);
    updateEstimate();
  } catch (error) {
    $('stateBadge').textContent = 'offline';
    $('stateBadge').className = 'badge failed';
  }
};

const pollLogs = async () => {
  try {
    const { entries } = await api(`/api/v1/train/logs?since=${state.lastSeq}`);
    if (!entries.length) return;
    const view = $('logView');
    const stick = view.scrollTop + view.clientHeight >= view.scrollHeight - 8;
    for (const entry of entries) {
      const time = new Date(entry.time * 1000).toLocaleTimeString();
      view.textContent += `[${time}] ${entry.text}\n`;
      state.lastSeq = entry.seq;
    }
    if (stick) view.scrollTop = view.scrollHeight;
  } catch (_) { /* offline: status badge already reflects it */ }
};

// ---------- actions ----------
const startTraining = async (config) => {
  const error = validateConfig(config);
  if (error) { showFormError(error); return; }
  const pending = state.status?.pending_session;
  if (pending && !configsMatch(config, pending) && !window.confirm(t('confirmDifferent'))) return;
  try {
    await postJson('/api/v1/train/start', config);
    showFormError(null);
    await pollStatus();
  } catch (err) {
    showFormError(err.message);
  }
};

const bindEvents = () => {
  $('langSelect').addEventListener('change', (e) => { state.lang = e.target.value; localStorage.setItem('maya.lang', state.lang); applyI18n(); });
  $('themeToggle').addEventListener('click', () => {
    const order = ['', 'light', 'dark'];
    const current = localStorage.getItem('maya.theme') || '';
    applyTheme(order[(order.indexOf(current) + 1) % order.length]);
  });
  $('presetSelect').addEventListener('change', (e) => applyPreset(e.target.value));
  document.querySelectorAll('.grid input').forEach((input) => input.addEventListener('input', () => { syncPresetSelect(); updateEstimate(); }));
  $('includeBuiltin').addEventListener('change', updateEstimate);
  $('selectAllBtn').addEventListener('click', () => {
    const all = state.datasets.map((d) => d.path);
    state.selected = state.selected.size === all.length ? new Set() : new Set(all);
    renderDatasets();
    updateEstimate();
  });
  $('refreshDatasetsBtn').addEventListener('click', () => loadDatasets().catch((e) => showFormError(e.message)));
  $('startBtn').addEventListener('click', () => startTraining(readConfig()));
  $('resumeBtn').addEventListener('click', () => {
    const pending = state.status?.pending_session;
    if (!pending) return;
    fillConfig(pending);
    startTraining(readConfig());
  });
  $('stopBtn').addEventListener('click', () => postJson('/api/v1/train/stop').then(pollStatus).catch((e) => showFormError(e.message)));
  $('clearSessionBtn').addEventListener('click', () => {
    if (window.confirm(t('confirmClear'))) postJson('/api/v1/train/session/clear').then(pollStatus).catch((e) => showFormError(e.message));
  });
  $('downloadCodeBtn').addEventListener('click', () => postJson('/api/v1/train/download', { script: 'code' }).then(pollStatus).catch((e) => showFormError(e.message)));
  $('downloadTurkishBtn').addEventListener('click', () => postJson('/api/v1/train/download', { script: 'turkish' }).then(pollStatus).catch((e) => showFormError(e.message)));
  $('clearLogsBtn').addEventListener('click', () => { $('logView').textContent = ''; });
};

const init = async () => {
  applyTheme(localStorage.getItem('maya.theme') || '');
  bindEvents();
  try {
    const { presets, defaults } = await api('/api/v1/train/presets');
    state.presets = presets;
    renderPresets();
    fillConfig({ ...defaults, datasets: undefined });
  } catch (error) {
    showFormError(error.message);
  }
  applyI18n();
  await Promise.allSettled([loadDatasets(), pollStatus(), pollLogs()]);
  setInterval(pollStatus, 1000);
  setInterval(pollLogs, 1000);
};

init();
