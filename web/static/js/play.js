/* Play & Learn. Every reading comes from /api/analyze/numbers, not from fake data. */
(() => {
  const $ = id => document.getElementById(id);
  const video = $('playVideo'), preview = $('playPreview');
  const start = $('startPlayCamera'), stop = $('stopPlayCamera');
  const check = $('checkGesture'), restart = $('newGame');
  const upload = $('playImage'), auto = $('autoScan');
  const feedback = $('gameFeedback');
  const overlay = $('cameraOverlay');
  const placeholder = $('cameraPlaceholder');
  const stars = [...$('progressStars').children];
  let stream = null, uploaded = null, busy = false, autoTimer = null;
  let round = 0, score = 0, consecutive = 0, complete = false;
  let order = [];
  let gameId = 0;
  let pauseUntil = 0;
  // Conservative vote threshold, NOT a calibrated probability.
  const MIN_VOTE = 0.55;

  function speak(message, type='') {
    feedback.textContent = message;
    feedback.className = 'game-feedback' + (type ? ' '+type : '');
  }
  function shuffle(values) {
    const copy = [...values];
    for(let i=copy.length-1;i>0;i--){
      const j = Math.floor(Math.random()*(i+1));
      [copy[i],copy[j]]=[copy[j],copy[i]];
    }
    return copy;
  }
  function render() {
    $('stars').innerHTML = `${score} <span>/ 5</span>`;
    $('round').innerHTML = `${Math.min(round+1,5)} <span>/ 5</span>`;
    stars.forEach((s,i) => {s.textContent=i<score?'★':'☆';s.classList.toggle('filled',i<score);});
    $('targetNumber').textContent=complete?'★':String(order[round]);
    $('promptText').innerHTML = complete ? '<strong>All five challenges complete!</strong>' : `Show <strong>${order[round]} fingers</strong> and hold your hand steady.`;
    check.disabled = busy || complete || (!stream && !uploaded);
  }
  function newGame() {
    gameId+=1;order=shuffle([1,2,3,4,5]);round=0;score=0;consecutive=0;complete=false;pauseUntil=0;
    $('lastReading').textContent='No scan yet';
    speak('New game! Start your camera or choose a photo.');
    render();
  }
  async function startCamera() {
    if(!navigator.mediaDevices?.getUserMedia){speak('Camera access requires localhost or HTTPS. You can upload a photo instead.','warn');return;}
    try{
      stopCamera();
      const s=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:960},height:{ideal:720},facingMode:'user'},audio:false});
      stream=s;video.srcObject=s;await video.play();
      uploaded=null;upload.value='';preview.hidden=true;video.hidden=false;placeholder.hidden=true;
      start.disabled=true;stop.disabled=false;$('cameraState').textContent='CAMERA READY';
      speak('Camera ready! Show the requested number with your full hand and wrist visible.');
      render();
      startAuto();
    }catch(err){speak('Camera access was blocked or no camera was found. Check permissions, or choose a photo.','warn');}
  }
  function stopCamera(){
    if(stream){stream.getTracks().forEach(t=>t.stop());}
    stream=null;video.srcObject=null;video.hidden=true;start.disabled=false;stop.disabled=true;
    if(!uploaded)placeholder.hidden=false;
    $('cameraState').textContent='STANDBY';
    stopAuto();render();
  }
  function stopAuto(){if(autoTimer!==null){clearInterval(autoTimer);autoTimer=null;} consecutive=0;}
  function startAuto(){
    stopAuto();
    if(stream && auto.checked){
      autoTimer=setInterval(()=>{
        if(!busy && !complete && Date.now()>=pauseUntil) analyze(true);
      },1700);
    }
  }
  function getFrameBlob(){
    if(uploaded)return Promise.resolve(uploaded);
    if(!stream || !video.videoWidth)return Promise.reject(new Error('Start the camera or choose a photo first.'));
    const canvas=document.createElement('canvas');canvas.width=video.videoWidth;canvas.height=video.videoHeight;
    const ctx=canvas.getContext('2d');ctx.translate(canvas.width,0);ctx.scale(-1,1);ctx.drawImage(video,0,0,canvas.width,canvas.height);
    return new Promise((resolve,reject)=>canvas.toBlob(b=>b?resolve(b):reject(new Error('Could not capture the frame.')), 'image/jpeg',0.88));
  }
  async function analyze(automatic=false){
    if(busy || complete || (!stream && !uploaded) || Date.now()<pauseUntil)return;
    busy=true;overlay.hidden=false;render();
    const runningGame=gameId;
    const runningRound=round;
    try{
      const blob=await getFrameBlob();
      const form=new FormData();form.append('image',blob,'play-frame.jpg');
      const response=await fetch('/api/analyze/numbers',{method:'POST',body:form});
      const body=await response.json();
      if(!response.ok)throw new Error(body.detail || `HTTP ${response.status}`);
      if(runningGame !== gameId || runningRound !== round)return;
      const d=body.details||{};
      const label=String(d.gesture||'');
      const number=d.number_prediction?.number;
      const votes=Number(d.confidence||0);
      $('lastReading').textContent = d.hand_count===0 ? 'No hand detected' : (number===null||number===undefined?'Unknown gesture':`${number} fingers · ${(votes*100).toFixed(0)}% votes`);
      if(d.hand_count===0){consecutive=0;speak('I cannot see your hand yet. Move back, show your wrist and palm, and add light in front of you.','warn');return;}
      if(!Number.isInteger(number) || number<1 || number>5){consecutive=0;speak('Try showing a number between 1 and 5.','warn');return;}
      if(votes<MIN_VOTE){consecutive=0;speak('Not quite sure yet. Hold your hand still and try again.','warn');return;}
      const expected=order[round];
      if(number!==expected){consecutive=0;speak(`I saw ${number} fingers. Try showing ${expected} fingers!`,'warn');return;}
      consecutive += 1;
      if(automatic && consecutive<2){speak(`I can see ${expected}! Hold steady for one more check…`,'good');return;}
      score+=1;round+=1;consecutive=0;
      if(round>=5){complete=true;speak('Amazing! You earned all 5 stars. You are a counting champion! 🌟','good');}
      else {speak(`Great job! ⭐ Next: show ${order[round]} fingers.`,'good');pauseUntil=Date.now()+1800;}
    }catch(err){consecutive=0;speak(err?.message || 'Recognition failed. Check the VS Code server terminal.','bad');}
    finally{busy=false;overlay.hidden=true;render();}
  }
  start.addEventListener('click',startCamera);
  stop.addEventListener('click',()=>{stopCamera();speak('Camera stopped. You can start it again or choose a photo.');});
  check.addEventListener('click',()=>analyze(false));
  restart.addEventListener('click',newGame);
  auto.addEventListener('change',startAuto);
  upload.addEventListener('change',()=>{
    const file=upload.files?.[0];if(!file)return;
    if(!['image/png','image/jpeg','image/webp'].includes(file.type) || file.size>12*1024*1024){speak('Choose a JPG, PNG, or WebP image smaller than 12 MB.','bad');upload.value='';return;}
    stopCamera();uploaded=file;video.hidden=true;preview.src=URL.createObjectURL(file);preview.hidden=false;placeholder.hidden=true;
    $('cameraState').textContent='PHOTO READY';
    speak('Photo ready. Click CHECK MY FINGERS to use your number recognition model.');render();
  });
  window.addEventListener('pagehide',stopCamera);
  newGame();
})();
