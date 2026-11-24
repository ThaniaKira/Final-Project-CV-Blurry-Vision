/* Blurry Vision Studio: UI only. Calls /api/analyze (the unchanged CV pipeline). */
(() => {
  const $ = id => document.getElementById(id);
  const fileInput = $('imageFile');
  const dropzone = $('dropzone');
  const viewport = $('inputViewport');
  const preview = $('previewImage');
  const video = $('cameraPreview');
  const cameraEmpty = $('cameraEmpty');
  const cameraControl = $('cameraControl');
  const startCamera = $('startCamera');
  const stopCameraButton = $('stopCamera');
  const captureButton = $('captureFrame');
  const analyzeButton = $('analyzeBtn');
  const clearButton = $('clearBtn');
  const output = $('resultImage');
  const outputEmpty = $('resultEmpty');
  const overlay = $('processingOverlay');
  const modelName = $('modelName');
  const modelNote = $('modelNote');
  const resultMessage = $('resultMessage');
  const downloadButton = $('downloadResult');
  const processingState = $('processingState');
  const inputs = [...document.querySelectorAll('.source-tab')];
  let currentFile = null;
  let currentObjectUrl = null;
  let stream = null;
  let source = 'upload';
  let outputDataUrl = null;
  let busy = false;

  const sampleNames = new Set(['open-hand', 'fist', 'peace', 'thumbs-up', 'pointing', 'ok']);

  function setState(message, kind='idle') {
    processingState.innerHTML = `<span class="status-dot ${kind}"></span> ${message}`;
  }
  function say(message, error=false) {
    resultMessage.textContent = message;
    resultMessage.classList.toggle('error-text', error);
  }
  function resetOutput() {
    output.hidden = true;
    output.removeAttribute('src');
    outputDataUrl = null;
    outputEmpty.hidden = false;
    $('gestureLabel').textContent = '—';
    $('confidence').textContent = '—';
    $('timeMetric').textContent = '—';
    $('handMetric').textContent = '—';
    downloadButton.disabled = true;
    setState('AWAITING INPUT');
    say('Ready when you are. Choose an image to begin.');
  }
  function resetInput() {
    currentFile = null;
    fileInput.value = '';
    if (currentObjectUrl) URL.revokeObjectURL(currentObjectUrl);
    currentObjectUrl = null;
    preview.hidden = true;
    preview.removeAttribute('src');
    dropzone.hidden = source !== 'upload';
    cameraEmpty.hidden = source !== 'camera';
    analyzeButton.disabled = true;
    resetOutput();
  }
  function stopCamera() {
    if (stream) stream.getTracks().forEach(track => track.stop());
    stream = null;
    video.srcObject = null;
    video.hidden = true;
    captureButton.disabled = true;
    if (source === 'camera' && !currentFile) cameraEmpty.hidden = false;
  }
  function setImage(file) {
    if (!file) return;
    const mime = file.type || '';
    if (!['image/png', 'image/jpeg', 'image/webp'].includes(mime)) {
      say('Please use a PNG, JPG, or WebP image.', true);
      return;
    }
    if (file.size > 12*1024*1024) {
      say('Please choose an image smaller than 12 MB.', true);
      return;
    }
    if (currentObjectUrl) URL.revokeObjectURL(currentObjectUrl);
    currentFile = file;
    currentObjectUrl = URL.createObjectURL(file);
    preview.src = currentObjectUrl;
    preview.hidden = false;
    dropzone.hidden = true;
    cameraEmpty.hidden = true;
    video.hidden = true;
    if (stream) stopCamera();
    analyzeButton.disabled = false;
    resetOutput();
    say('Image loaded. Select Analyze Gesture to run recognition.');
    setState('READY TO SCAN', 'ready');
  }
  function setSource(next) {
    if (busy || !['upload', 'camera'].includes(next)) return;
    if (source === next) return;
    stopCamera();
    source = next;
    inputs.forEach(tab => {
      const isActive = tab.dataset.tab === next;
      tab.classList.toggle('active', isActive);
      tab.setAttribute('aria-pressed', String(isActive));
    });
    resetInput();
    cameraControl.hidden = next !== 'camera';
  }

  inputs.forEach(tab => tab.addEventListener('click', () => setSource(tab.dataset.tab)));
  dropzone.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
  });
  fileInput.addEventListener('change', () => setImage(fileInput.files?.[0]));
  for (const kind of ['dragenter', 'dragover']) {
    viewport.addEventListener(kind, e => {
      if (source !== 'upload') return;
      e.preventDefault();
      viewport.classList.add('dragging');
    });
  }
  for (const kind of ['dragleave', 'drop']) {
    viewport.addEventListener(kind, e => {
      if (source !== 'upload') return;
      e.preventDefault();
      viewport.classList.remove('dragging');
    });
  }
  viewport.addEventListener('drop', e => {
    if (source === 'upload') setImage(e.dataTransfer?.files?.[0]);
  });
  startCamera.addEventListener('click', async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      say('Camera access is unavailable. Use localhost in a supported browser or upload a photo.', true);
      return;
    }
    try {
      stream = await navigator.mediaDevices.getUserMedia({audio:false,video:{width:{ideal:960},height:{ideal:720},facingMode:'user'}});
      video.srcObject = stream;
      video.hidden = false;
      cameraEmpty.hidden = true;
      dropzone.hidden = true;
      await video.play();
      captureButton.disabled = false;
      say('Camera ready. Show your gesture and click Capture Frame.');
    } catch (err) {
      say('Camera access was blocked or unavailable. Allow permission or upload an image instead.', true);
    }
  });
  stopCameraButton.addEventListener('click', () => {
    stopCamera();
    say('Camera stopped.');
  });
  captureButton.addEventListener('click', () => {
    if (!stream || !video.videoWidth) {
      say('Camera is still loading. Try again.', true);
      return;
    }
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.translate(canvas.width, 0); ctx.scale(-1, 1);
    ctx.drawImage(video,0,0,canvas.width,canvas.height);
    canvas.toBlob(blob => {
      if (blob) setImage(new File([blob], 'captured-gesture.png', {type:'image/png'}));
      else say('Could not capture frame. Try again.', true);
    }, 'image/png');
  });
  clearButton.addEventListener('click', () => {
    stopCamera();
    resetInput();
    say('Cleared. Capture or upload another image.');
  });
  analyzeButton.addEventListener('click', async () => {
    if (!currentFile || busy) return;
    busy = true;
    analyzeButton.disabled = true;
    clearButton.disabled = true;
    inputs.forEach(tab => tab.disabled = true);
    setState('PROCESSING', 'ready');
    overlay.hidden = false;
    say('Running gesture recognition…');
    const form = new FormData();
    form.append('image', currentFile, currentFile.name);
    try {
      const response = await fetch('/api/analyze', {method:'POST',body:form});
      let body;
      try { body = await response.json(); }
      catch { throw new Error(`Server returned HTTP ${response.status}. Check the VS Code terminal.`); }
      if (!response.ok) throw new Error(body.detail || `Request failed (HTTP ${response.status}).`);
      output.src = body.image;
      output.hidden = false;
      outputEmpty.hidden = true;
      outputDataUrl = body.image;
      downloadButton.disabled = false;
      const d = body.details || {};
      const label = String(d.gesture || 'Unknown').replace(/_/g, ' ');
      $('gestureLabel').textContent = label;
      $('confidence').textContent = (d.hand_count > 0 && typeof d.confidence === 'number' && body.status?.mode === 'trained')
        ? `${(d.confidence * 100).toFixed(1)}%*` : 'N/A';
      $('timeMetric').textContent = (typeof d.processing_ms === 'number') ? `${d.processing_ms.toFixed(1)} ms` : '—';
      $('handMetric').textContent = String(d.hand_count ?? '—');
      setState('SCAN COMPLETE', 'ready');
      const isMissing = /no hand/i.test(label);
      say(isMissing
        ? 'No hand landmarks found. Move your hand farther back, include your wrist, face your palm toward the camera, and improve front lighting. This is a detection failure, not a gesture classification.'
        : (body.status?.mode === 'trained'
            ? 'Recognition completed using your existing trained model. *Score is not a calibrated probability.'
            : 'Recognition completed using existing geometric rules; no trained confidence score is available.'));
    } catch (err) {
      setState('SCAN ERROR', 'error');
      say(err.message || 'Could not analyze this image. Check the terminal.', true);
    } finally {
      overlay.hidden = true;
      busy = false;
      clearButton.disabled = false;
      inputs.forEach(tab => tab.disabled = false);
      analyzeButton.disabled = !currentFile;
    }
  });
  downloadButton.addEventListener('click', () => {
    if (!outputDataUrl) return;
    const link = document.createElement('a');
    link.href = outputDataUrl;
    link.download = 'blurry-vision-result.png';
    document.body.appendChild(link);
    link.click();
    link.remove();
  });

  async function useSample(sample) {
    if (!sampleNames.has(sample)) return;
    try {
      const resp = await fetch(`/static/samples/${sample}.png`);
      if (!resp.ok) throw new Error('Could not load sample image.');
      const blob = await resp.blob();
      setSource('upload');
      setImage(new File([blob], `${sample}.png`, {type:'image/png'}));
      window.scrollTo({top:0,behavior:'smooth'});
    } catch (err) { say(err.message, true); }
  }
  document.querySelectorAll('[data-sample]').forEach(button => {
    button.addEventListener('click', () => useSample(button.dataset.sample));
  });
  const requested = new URLSearchParams(window.location.search).get('sample');
  if (requested && sampleNames.has(requested)) useSample(requested);

  fetch('/api/status')
    .then(resp => { if (!resp.ok) throw new Error('Unavailable'); return resp.json(); })
    .then(info => {
      modelName.textContent = info.label;
      modelNote.textContent = info.mode === 'trained' ? 'EXISTING TRAINED WEIGHTS · LOCAL' : 'EXISTING RULE-BASED FALLBACK · LOCAL';
      $('transparencyNote').textContent = info.note + ' The website does not modify training, weights, or model settings.';
    })
    .catch(() => { modelName.textContent = 'Model status unavailable'; modelNote.textContent = 'Please check the local server'; });
  window.addEventListener('beforeunload', stopCamera);
  resetOutput();
})();
