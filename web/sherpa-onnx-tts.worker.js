let tts = null;

const DATA_CHUNKS = ["chunks/data-aa", "chunks/data-ab", "chunks/data-ac", "chunks/data-ad"];
const DATA_TOTAL_SIZE = 81893213;

async function fetchChunkedData() {
  self.postMessage({ type: "sherpa-onnx-tts-progress", status: "Downloading model data..." });
  const buffers = [];
  let loaded = 0;
  for (let i = 0; i < DATA_CHUNKS.length; i++) {
    const resp = await fetch(DATA_CHUNKS[i]);
    if (!resp.ok) throw new Error("Failed to fetch " + DATA_CHUNKS[i]);
    const buf = await resp.arrayBuffer();
    buffers.push(new Uint8Array(buf));
    loaded += buf.byteLength;
    const pct = Math.round((loaded / DATA_TOTAL_SIZE) * 100);
    self.postMessage({ type: "sherpa-onnx-tts-progress", status: `Downloading model... ${pct}%` });
  }
  const combined = new Uint8Array(loaded);
  let offset = 0;
  for (const buf of buffers) {
    combined.set(buf, offset);
    offset += buf.length;
  }
  return combined.buffer;
}

let preloadedData = null;

fetchChunkedData().then((data) => {
  preloadedData = data;
  initModule();
}).catch((e) => {
  self.postMessage({ type: "error", message: "Download failed: " + e.message });
});

function initModule() {
  self.Module = {
    locateFile: function (path, scriptDirectory) {
      return (scriptDirectory || "") + path;
    },
    getPreloadedPackage: function (name, size) {
      return preloadedData;
    },
    setStatus: function (status) {
      self.postMessage({ type: "sherpa-onnx-tts-progress", status });
    },
    onRuntimeInitialized: function () {
      console.log("Model files downloaded!");
      console.log("Initializing tts ......");
      try {
        tts = createOfflineTts(self.Module);
        self.postMessage({
          type: "sherpa-onnx-tts-ready",
          modelType: getDefaultOfflineTtsModelType(),
          numSpeakers: tts.numSpeakers,
        });
      } catch (e) {
        self.postMessage({
          type: "error",
          message: "TTS Initialization failed: " + e.message,
        });
      }
    },
  };
  importScripts("sherpa-onnx-wasm-main-tts.js");
  importScripts("sherpa-onnx-tts.js");
}

function getErrorMessage(err) {
  if (err instanceof Error) {
    if (err.stack) {
      return `${err.message}\n${err.stack}`;
    }
    return err.message;
  }

  return `${err}`;
}

self.onmessage = async (e) => {
  const { type, text, sid, speed, genConfig } = e.data;
  if (type === "generate") {
    if (!tts) {
      return;
    }
    try {
      const audio = tts.generate({
        text: text,
        sid: sid || 0,
        speed: speed || 1.0,
      });
      const samples = audio.samples;
      const sampleRate = tts.sampleRate;
      self.postMessage(
        {
          type: "sherpa-onnx-tts-result",
          samples: samples,
          sampleRate: sampleRate,
        },
        [samples.buffer],
      );
    } catch (err) {
      self.postMessage({
        type: "error",
        message: "Generation failed: " + getErrorMessage(err),
      });
    }
  } else if (type === "generateWithConfig") {
    if (!tts) {
      return;
    }
    try {
      const config = Object.assign({}, genConfig || {});
      config.callback = (samples, n, progress) => {
        self.postMessage({
          type: "sherpa-onnx-tts-generation-progress",
          progress: progress,
        });
        return 1;
      };

      const audio = tts.generateWithConfig(text, config);
      const samples = audio.samples;
      const sampleRate = audio.sampleRate;
      self.postMessage(
          {
            type: "sherpa-onnx-tts-result",
            samples: samples,
            sampleRate: sampleRate,
          },
          [samples.buffer],
      );
    } catch (err) {
      self.postMessage({
        type: "error",
        message: "Generation failed: " + getErrorMessage(err),
      });
    }
  }
};
