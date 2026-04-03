// TTS Demo — app.js
// Initializes sherpa-onnx WASM worker, handles text→audio pipeline

const $ = (sel) => document.querySelector(sel);
const textEl    = $('#text');
const speakBtn  = $('#speak');
const speedEl   = $('#speed');
const speedVal  = $('#speed-val');
const statusEl  = $('#status');
const statusTxt = $('#status-text');
const statsEl   = $('#stats');
const audioEl   = $('#audio');
const progressBar  = $('#progress');
const progressFill = $('#progress-fill');

let worker = null;
let ready = false;
let generating = false;

// --- Speed slider ---
speedEl.addEventListener('input', () => {
  speedVal.textContent = parseFloat(speedEl.value).toFixed(1) + 'x';
});

// --- Status helpers ---
function setStatus(cls, text, showSpinner) {
  statusEl.className = cls;
  statusTxt.textContent = text;
  const spinner = statusEl.querySelector('.spinner');
  if (spinner) spinner.style.display = showSpinner ? 'inline-block' : 'none';
}

function showProgress(pct) {
  progressBar.classList.toggle('active', pct !== null);
  if (pct !== null) progressFill.style.width = pct + '%';
}

// --- WAV encoding ---
function samplesToWavBlob(samples, sampleRate) {
  const numChannels = 1;
  const bitsPerSample = 16;
  const byteRate = sampleRate * numChannels * bitsPerSample / 8;
  const blockAlign = numChannels * bitsPerSample / 8;
  const dataSize = samples.length * blockAlign;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  // RIFF header
  writeStr(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeStr(view, 8, 'WAVE');
  // fmt
  writeStr(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);
  // data
  writeStr(view, 36, 'data');
  view.setUint32(40, dataSize, true);

  // Convert float32 [-1,1] to int16
  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    let s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    offset += 2;
  }
  return new Blob([buffer], { type: 'audio/wav' });
}

function writeStr(view, offset, str) {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}

// --- Worker init ---
function initWorker() {
  worker = new Worker('sherpa-onnx-tts.worker.js');

  worker.onmessage = (e) => {
    const msg = e.data;
    switch (msg.type) {
      case 'sherpa-onnx-tts-progress':
        if (!ready) setStatus('loading', msg.status || 'Loading model...', true);
        break;

      case 'sherpa-onnx-tts-ready':
        ready = true;
        textEl.disabled = false;
        speakBtn.disabled = false;
        setStatus('ready', `Model ready (${msg.numSpeakers} speaker${msg.numSpeakers !== 1 ? 's' : ''})`, false);
        showProgress(null);
        break;

      case 'sherpa-onnx-tts-generation-progress':
        if (msg.progress != null) {
          showProgress(Math.round(msg.progress * 100));
        }
        break;

      case 'sherpa-onnx-tts-result': {
        generating = false;
        speakBtn.disabled = false;
        speakBtn.textContent = 'Speak';
        showProgress(null);

        const { samples, sampleRate } = msg;
        const duration = samples.length / sampleRate;
        const genTime = (performance.now() - window._genStart) / 1000;
        const rtf = genTime / duration;

        statsEl.textContent =
          `Generated ${duration.toFixed(2)}s audio in ${genTime.toFixed(2)}s — RTF: ${rtf.toFixed(3)}`;

        // Play via <audio> element (WAV blob)
        const blob = samplesToWavBlob(samples, sampleRate);
        const url = URL.createObjectURL(blob);
        audioEl.src = url;
        audioEl.style.display = 'block';
        audioEl.play().catch(() => {});

        setStatus('ready', 'Done', false);
        break;
      }

      case 'error':
        generating = false;
        speakBtn.disabled = false;
        speakBtn.textContent = 'Speak';
        showProgress(null);
        setStatus('error', 'Error: ' + (msg.message || 'Unknown error'), false);
        break;

      default:
        console.log('Worker message:', msg);
    }
  };

  worker.onerror = (err) => {
    setStatus('error', 'Worker error: ' + err.message, false);
    console.error('Worker error:', err);
  };
}

// --- Speak ---
speakBtn.addEventListener('click', () => {
  const text = textEl.value.trim();
  if (!text || !ready || generating) return;

  generating = true;
  speakBtn.disabled = true;
  speakBtn.textContent = 'Generating...';
  statsEl.textContent = '';
  setStatus('generating', 'Generating speech...', true);
  showProgress(0);

  window._genStart = performance.now();

  worker.postMessage({
    type: 'generate',
    text: text,
    sid: 0,
    speed: parseFloat(speedEl.value),
  });
});

// --- Keyboard shortcut ---
textEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    speakBtn.click();
  }
});

// --- Boot ---
initWorker();
