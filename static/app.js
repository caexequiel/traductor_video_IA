// Estado de la aplicación
const state = {
  selectedVideoPath: null,
  selectedSubPath: null,
  subMode: 'whisper', // 'whisper' | 'file'
  currentExplorerTarget: 'video', // 'video' | 'subtitle'
  currentBrowsePath: null,
  activeJobId: null,
  ws: null
};

// Elementos DOM
const dom = {
  gpuBadge: document.getElementById('gpuBadge'),
  gpuText: document.getElementById('gpuText'),
  selectedFileInfo: document.getElementById('selectedFileInfo'),
  btnOpenExplorer: document.getElementById('btnOpenExplorer'),
  btnOpenWebExplorer: document.getElementById('btnOpenWebExplorer'),
  videoPreviewCard: document.getElementById('videoPreviewCard'),
  origVideoPlayer: document.getElementById('origVideoPlayer'),
  radioWhisper: document.getElementById('radioWhisper'),
  radioFile: document.getElementById('radioFile'),
  subFileSection: document.getElementById('subFileSection'),
  selectedSubInfo: document.getElementById('selectedSubInfo'),
  btnSelectSub: document.getElementById('btnSelectSub'),
  sourceLangSelect: document.getElementById('sourceLangSelect'),
  dialectSelect: document.getElementById('dialectSelect'),
  whisperModelSelect: document.getElementById('whisperModelSelect'),
  checkPreserveBg: document.getElementById('checkPreserveBg'),
  btnStartDubbing: document.getElementById('btnStartDubbing'),
  progressBarFill: document.getElementById('progressBarFill'),
  progressMessage: document.getElementById('progressMessage'),
  progressPercentage: document.getElementById('progressPercentage'),
  segmentsTbody: document.getElementById('segmentsTbody'),
  segmentCountBadge: document.getElementById('segmentCountBadge'),
  resultCard: document.getElementById('resultCard'),
  dubbedVideoPlayer: document.getElementById('dubbedVideoPlayer'),
  btnDownload: document.getElementById('btnDownload'),
  // Modal
  explorerModal: document.getElementById('explorerModal'),
  modalTitle: document.getElementById('modalTitle'),
  btnCloseModal: document.getElementById('btnCloseModal'),
  btnCancelExplorer: document.getElementById('btnCancelExplorer'),
  bookmarksBar: document.getElementById('bookmarksBar'),
  btnGoParent: document.getElementById('btnGoParent'),
  pathDisplay: document.getElementById('pathDisplay'),
  explorerTbody: document.getElementById('explorerTbody'),
  explorerHint: document.getElementById('explorerHint')
};

// Inicialización
document.addEventListener('DOMContentLoaded', () => {
  initGPUStatus();
  initWebSocket();
  initEvents();
  loadBookmarks();
});

// Eventos de interfaz
function initEvents() {
  dom.btnOpenExplorer.addEventListener('click', () => openNativeOrWebExplorer('video'));
  if (dom.btnOpenWebExplorer) {
    dom.btnOpenWebExplorer.addEventListener('click', () => openWebExplorer('video'));
  }
  dom.btnSelectSub.addEventListener('click', () => openNativeOrWebExplorer('subtitle'));
  dom.btnCloseModal.addEventListener('click', closeExplorer);
  dom.btnCancelExplorer.addEventListener('click', closeExplorer);

  dom.radioWhisper.addEventListener('change', () => {
    state.subMode = 'whisper';
    dom.subFileSection.classList.add('hidden');
  });

  dom.radioFile.addEventListener('change', () => {
    state.subMode = 'file';
    dom.subFileSection.classList.remove('hidden');
  });

  dom.btnGoParent.addEventListener('click', () => {
    if (state.parentBrowsePath) {
      browsePath(state.parentBrowsePath);
    }
  });

  dom.btnStartDubbing.addEventListener('click', startDubbingProcess);
}

// WebSocket para actualización en vivo
function initWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/progress`;
  
  state.ws = new WebSocket(wsUrl);

  state.ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      handleJobUpdate(data);
    } catch (e) {
      console.error("Error parseando WS message:", e);
    }
  };

  state.ws.onclose = () => {
    setTimeout(initWebSocket, 3000); // Reconexión automática
  };
}

// Monitoreo de Hardware (GPU / VRAM)
async function initGPUStatus() {
  async function update() {
    try {
      const res = await fetch('/api/system/gpu');
      if (res.ok) {
        const data = await res.json();
        if (data.has_cuda) {
          const usedGb = (data.vram_used_mb / 1024).toFixed(1);
          const totalGb = (data.vram_total_mb / 1024).toFixed(1);
          dom.gpuText.innerText = `${data.device_name} | VRAM: ${usedGb} / ${totalGb} GB (${data.temperature}°C)`;
        } else {
          dom.gpuText.innerText = "Modo CPU (CUDA no detectado)";
        }
      }
    } catch (e) {
      dom.gpuText.innerText = "GPU disponible";
    }
  }
  update();
  setInterval(update, 4000);
}

// Explorador de Archivos Nativo de CachyOS / Web
async function openNativeOrWebExplorer(targetType) {
  state.currentExplorerTarget = targetType;
  
  // Cambiar texto de botón temporalmente para dar feedback visual
  const activeBtn = targetType === 'video' ? dom.btnOpenExplorer : dom.btnSelectSub;
  const originalText = activeBtn.innerText;
  activeBtn.innerText = "⏳ Abriendo CachyOS...";
  activeBtn.disabled = true;

  try {
    const res = await fetch(`/api/fs/native-picker?type=${targetType}`, { method: 'POST' });
    if (res.ok) {
      const data = await res.json();
      if (data.status === 'selected') {
        onFileSelected({
          path: data.path,
          name: data.filename,
          size: data.size
        });
        activeBtn.innerText = originalText;
        activeBtn.disabled = false;
        return;
      } else if (data.status === 'cancelled') {
        // Cancelado por el usuario en el diálogo
        activeBtn.innerText = originalText;
        activeBtn.disabled = false;
        return;
      }
    }
  } catch (err) {
    console.warn("Explorador nativo no disponible, abriendo explorador web:", err);
  } finally {
    activeBtn.innerText = originalText;
    activeBtn.disabled = false;
  }

  // Fallback al explorador web
  openWebExplorer(targetType);
}

function openWebExplorer(targetType) {
  state.currentExplorerTarget = targetType;
  if (targetType === 'video') {
    dom.modalTitle.innerText = "Seleccionar Archivo de Video (Web)";
    dom.explorerHint.innerText = "Formatos soportados: MP4, MKV, AVI, MOV, WEBM, FLV";
  } else {
    dom.modalTitle.innerText = "Seleccionar Archivo de Subtítulos (Web)";
    dom.explorerHint.innerText = "Formatos soportados: SRT, VTT";
  }
  dom.explorerModal.classList.remove('hidden');
  browsePath(state.currentBrowsePath || null);
}

function closeExplorer() {
  dom.explorerModal.classList.add('hidden');
}

async function loadBookmarks() {
  try {
    const res = await fetch('/api/fs/bookmarks');
    if (res.ok) {
      const data = await res.json();
      dom.bookmarksBar.innerHTML = '';
      data.bookmarks.forEach(bm => {
        const btn = document.createElement('button');
        btn.className = 'bookmark-btn';
        btn.innerText = `📁 ${bm.name}`;
        btn.addEventListener('click', () => browsePath(bm.path));
        dom.bookmarksBar.appendChild(btn);
      });
    }
  } catch (e) {
    console.error("Error cargando accesos rápidos:", e);
  }
}

async function browsePath(dirPath) {
  try {
    const url = dirPath ? `/api/fs/browse?path=${encodeURIComponent(dirPath)}` : '/api/fs/browse';
    const res = await fetch(url);
    if (!res.ok) return;

    const data = await res.json();
    state.currentBrowsePath = data.current_path;
    state.parentBrowsePath = data.parent_path;

    dom.pathDisplay.innerText = data.current_path;
    dom.btnGoParent.disabled = !data.parent_path;

    renderExplorerEntries(data.entries);
  } catch (e) {
    console.error("Error explorando carpeta:", e);
  }
}

function renderExplorerEntries(entries) {
  dom.explorerTbody.innerHTML = '';

  if (!entries || entries.length === 0) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td colspan="4" class="text-center text-muted">Carpeta vacía o sin archivos compatibles.</td>`;
    dom.explorerTbody.appendChild(tr);
    return;
  }

  entries.forEach(entry => {
    // Filtrar según el objetivo
    if (state.currentExplorerTarget === 'video' && !entry.is_dir && !entry.is_video) return;
    if (state.currentExplorerTarget === 'subtitle' && !entry.is_dir && !entry.is_subtitle) return;

    const tr = document.createElement('tr');
    tr.className = 'explorer-row';

    let icon = '📄';
    if (entry.is_dir) icon = '📁';
    else if (entry.is_video) icon = '🎬';
    else if (entry.is_subtitle) icon = '📝';

    const sizeStr = entry.is_dir ? '-' : formatBytes(entry.size);
    const typeStr = entry.is_dir ? 'Carpeta' : (entry.ext.toUpperCase().replace('.', '') || 'Archivo');

    tr.innerHTML = `
      <td><span class="file-icon">${icon}</span><strong>${escapeHtml(entry.name)}</strong></td>
      <td>${typeStr}</td>
      <td>${sizeStr}</td>
      <td></td>
    `;

    const actionTd = tr.querySelector('td:last-child');

    if (entry.is_dir) {
      tr.addEventListener('click', () => browsePath(entry.path));
    } else {
      const selectBtn = document.createElement('button');
      selectBtn.className = 'btn btn-primary btn-sm';
      selectBtn.innerText = 'Seleccionar';
      selectBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        onFileSelected(entry);
      });
      actionTd.appendChild(selectBtn);
    }

    dom.explorerTbody.appendChild(tr);
  });
}

async function onFileSelected(entry) {
  closeExplorer();

  if (state.currentExplorerTarget === 'video') {
    state.selectedVideoPath = entry.path;
    dom.selectedFileInfo.innerHTML = `
      <div><strong>${escapeHtml(entry.name)}</strong></div>
      <div class="file-meta-pill">${formatBytes(entry.size)}</div>
    `;

    // Cargar previsualización del video
    dom.origVideoPlayer.src = `/api/media/stream?path=${encodeURIComponent(entry.path)}`;
    dom.videoPreviewCard.classList.remove('hidden');

    dom.btnStartDubbing.disabled = false;
  } else if (state.currentExplorerTarget === 'subtitle') {
    state.selectedSubPath = entry.path;
    dom.selectedSubInfo.innerText = entry.name;
  }
}

// Iniciar Doblaje
async function startDubbingProcess() {
  if (!state.selectedVideoPath) return;

  dom.btnStartDubbing.disabled = true;
  dom.resultCard.classList.add('hidden');
  resetStepper();

  const sourceLang = dom.sourceLangSelect && dom.sourceLangSelect.value !== 'auto'
    ? dom.sourceLangSelect.value
    : null;
  const dialect = dom.dialectSelect ? dom.dialectSelect.value : 'es-latam';

  const payload = {
    video_path: state.selectedVideoPath,
    subtitle_path: state.subMode === 'file' ? state.selectedSubPath : null,
    source_language: sourceLang,
    dialect: dialect,
    whisper_model: dom.whisperModelSelect.value,
    preserve_bg: dom.checkPreserveBg.checked
  };

  try {
    const res = await fetch('/api/process/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (res.ok) {
      const data = await res.json();
      state.activeJobId = data.job_id;
      dom.progressMessage.innerText = "Iniciando pipeline de doblaje...";
      dom.progressBarFill.style.width = "5%";
      dom.progressPercentage.innerText = "5%";
    } else {
      const err = await res.json();
      alert(`Error al iniciar: ${err.detail || 'Error desconocido'}`);
      dom.btnStartDubbing.disabled = false;
    }
  } catch (e) {
    alert(`Error de red: ${e.message}`);
    dom.btnStartDubbing.disabled = false;
  }
}

// Manejar eventos del WebSocket
function handleJobUpdate(data) {
  if (data.type === 'job_update') {
    const stage = data.stage;
    const progress = Math.round(data.progress * 100);

    updateStepper(stage);
    dom.progressBarFill.style.width = `${progress}%`;
    dom.progressPercentage.innerText = `${progress}%`;
    dom.progressMessage.innerText = data.message;

    if (data.extra && data.extra.segments) {
      renderSegmentsTable(data.extra.segments);
    }
  } else if (data.type === 'job_completed') {
    dom.progressBarFill.style.width = '100%';
    dom.progressPercentage.innerText = '100%';
    dom.progressMessage.innerText = '¡Doblaje y renderizado completado con éxito!';
    dom.btnStartDubbing.disabled = false;
    markAllStepsCompleted();

    const result = data.result;
    if (result && result.output_video) {
      dom.dubbedVideoPlayer.src = `/api/media/stream?path=${encodeURIComponent(result.output_video)}`;
      dom.btnDownload.href = `/api/media/stream?path=${encodeURIComponent(result.output_video)}`;
      dom.btnDownload.setAttribute('download', result.filename);
      dom.resultCard.classList.remove('hidden');
      dom.dubbedVideoPlayer.scrollIntoView({ behavior: 'smooth' });
    }
  } else if (data.type === 'job_failed') {
    dom.progressBarFill.style.backgroundColor = 'var(--danger)';
    dom.progressMessage.innerText = `Error: ${data.error}`;
    dom.btnStartDubbing.disabled = false;
  }
}

// Stepper helpers
const STEP_IDS = ['extract', 'separation', 'transcription', 'translation', 'voice_cloning', 'synchronization', 'muxing'];

function resetStepper() {
  STEP_IDS.forEach(id => {
    const el = document.getElementById(`step-${id}`);
    if (el) el.className = 'step-item';
  });
}

function updateStepper(activeStage) {
  let found = false;
  STEP_IDS.forEach(id => {
    const el = document.getElementById(`step-${id}`);
    if (!el) return;

    if (id === activeStage) {
      el.className = 'step-item active';
      found = true;
    } else if (!found) {
      el.className = 'step-item completed';
    } else {
      el.className = 'step-item';
    }
  });
}

function markAllStepsCompleted() {
  STEP_IDS.forEach(id => {
    const el = document.getElementById(`step-${id}`);
    if (el) el.className = 'step-item completed';
  });
}

function renderSegmentsTable(segments) {
  dom.segmentsTbody.innerHTML = '';
  dom.segmentCountBadge.innerText = `${segments.length} frases`;

  segments.forEach((seg, idx) => {
    const tr = document.createElement('tr');
    const start = seg.start !== undefined ? `${seg.start.toFixed(1)}s` : '-';
    const end = seg.end !== undefined ? `${seg.end.toFixed(1)}s` : '-';
    const original = seg.text || '-';
    const trans = seg.translated_text || '<span class="text-muted">Traduciendo...</span>';

    tr.innerHTML = `
      <td>${idx + 1}</td>
      <td><code>${start} - ${end}</code></td>
      <td>${escapeHtml(original)}</td>
      <td><strong style="color: var(--primary);">${escapeHtml(trans)}</strong></td>
    `;
    dom.segmentsTbody.appendChild(tr);
  });
}

// Utilidades
function formatBytes(bytes) {
  if (!bytes || bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function escapeHtml(text) {
  if (!text) return '';
  const div = document.createElement('div');
  div.innerText = text;
  return div.innerHTML;
}
