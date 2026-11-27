/* Real camera games. Predictions are fetched from /api/live, never invented. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const video = $('playVideo');
  const WORDS = ['zero','one','two','three','four','five'];
  const POSES = [
    ['Open_Hand','open-hand','open hand'],['Fist','fist','fist'],
    ['Peace','peace','peace sign'],['Thumbs_Up','thumbs-up','thumbs up'],
    ['Pointing','pointing','pointing'],['OK','ok','OK sign']
  ];
  const POSE_NAMES = Object.fromEntries(POSES.map(p => [p[0],p[2]]));
  const MIN_VOTE = {count:0.53, match:0.55, math:0.53, memory:0.53};
  const SCAN_PAUSE_MS = 650;
  const TIME_LIMITS = {math: 60, memory: 75}; // Gentle countdowns; no stars are taken away.
  const WRONG_CUE_COOLDOWN_MS = 7200;
  let stream = null, running = false, pollTimer = null, inFlight = false;
  let runVersion = 0, voiceWait = 0, feedbackKey = '';
  let gameMode = 'count', plan = [], round = 0, stars = 0, complete = false;
  let memoryIndex = 0, memoryRevealing = false, revealTimer = null, revealSerial = 0;
  let candidate = null, consecutive = 0, previousAccepted = null, requireChange = false;
  let pauseUntil = 0, modelReady = {core:false, numbers:false};
  let lastSpokenAt = 0, lastWrongCueAt = 0, lastPromptAt = 0;
  let timerInterval = null, deadline = 0, secondsLeft = null;
  let availableVoices = [], chosenVoice = null;

  function shuffle(values) {
    const copy = [...values];
    for (let i=copy.length-1;i>0;i--) {
      const j=Math.floor(Math.random()*(i+1)); [copy[i],copy[j]]=[copy[j],copy[i]];
    }
    return copy;
  }
  function mathProblems() {
    const examples = [
      {a:1,b:2,op:'+',answer:3},{a:2,b:2,op:'+',answer:4},
      {a:3,b:2,op:'+',answer:5},{a:5,b:3,op:'−',answer:2},
      {a:4,b:3,op:'−',answer:1},{a:1,b:1,op:'+',answer:2},
      {a:5,b:1,op:'−',answer:4}
    ];
    return shuffle(examples).slice(0,5);
  }
  function sequence(length) {
    const nums=[];
    while(nums.length<length) {
      const value=1+Math.floor(Math.random()*5);
      if (value!==nums[nums.length-1]) nums.push(value);
    }
    return nums;
  }
  function totalRounds() { return plan.length; }
  function expected() {
    if (gameMode==='count') return plan[round];
    if (gameMode==='match') return plan[round][0];
    if (gameMode==='math') return plan[round].answer;
    return plan[round][memoryIndex];
  }
  function feedback(text, kind='') {
    const key=`${text}|${kind}`;
    if (key===feedbackKey) return;
    feedbackKey=key;
    $('gameFeedback').textContent=text;
    $('gameFeedback').className='game-feedback'+(kind?' '+kind:'');
  }
  // Speak warmly using the most natural English voice installed on the device.
  // Browser speech voices vary by OS/browser; parents may choose another voice below.
  function refreshVoices() {
    if (!('speechSynthesis' in window)) return;
    availableVoices=window.speechSynthesis.getVoices().filter(v=>/^en([-_]|$)/i.test(v.lang));
    const pick=$('voiceChoice');
    if (!pick) return;
    const previous=pick.value;
    pick.replaceChildren();
    const auto=document.createElement('option');auto.value='auto';auto.textContent='Friendly voice (auto)';pick.append(auto);
    availableVoices.forEach((v,i)=>{const o=document.createElement('option');o.value=String(i);o.textContent=v.name;pick.append(o);});
    if(previous==='auto'||availableVoices[Number(previous)])pick.value=previous;
    chosenVoice=pick.value;
  }
  if ('speechSynthesis' in window) {
    refreshVoices();
    window.speechSynthesis.addEventListener?.('voiceschanged',refreshVoices);
  }
  function friendlyVoice() {
    const index=Number(chosenVoice);
    if(chosenVoice!==null&&chosenVoice!=='auto'&&Number.isInteger(index)&&availableVoices[index])return availableVoices[index];
    return availableVoices.find(v=>/Jenny|Aria|Samantha|Ava|Serena|Google US English|Zira|Natural|Neural|Premium|Enhanced/i.test(v.name)) || availableVoices.find(v=>/female/i.test(v.name)) || availableVoices[0] || null;
  }
  function speak(text, onFinished=null) {
    if (!('speechSynthesis' in window)) {onFinished?.();return false;}
    try {
      window.speechSynthesis.cancel();
      const utterance=new SpeechSynthesisUtterance(text);
      utterance.lang='en-US';utterance.rate=0.91;utterance.pitch=1.16;utterance.volume=1;
      const voice=friendlyVoice();if(voice)utterance.voice=voice;
      utterance.onstart=()=>window.BlurryAudio?.duck?.(true);
      utterance.onend=()=>{window.BlurryAudio?.duck?.(false);onFinished?.();};
      utterance.onerror=()=>{window.BlurryAudio?.duck?.(false);onFinished?.();};
      window.speechSynthesis.speak(utterance);return true;
    } catch (_) { window.BlurryAudio?.duck?.(false);onFinished?.();return false; }
  }
  function coachWrongGesture(recognized, confidence) {
    if(!running||memoryRevealing||complete||Date.now()<pauseUntil||Date.now()-lastPromptAt<2700||confidence<0.50||recognized===expected())return;
    const now=Date.now();if(now-lastWrongCueAt<WRONG_CUE_COOLDOWN_MS)return;
    lastWrongCueAt=now;
    if(gameMode==='match')speak(`Good try! ${POSE_NAMES[expected()]} gesture, please. You can do it!`);
    else if(gameMode==='math') {
      const answer=expected();speak(`Almost there! Show ${WORDS[answer]} ${answer===1?'finger':'fingers'}, please!`);
    } else if(gameMode==='count') {
      const n=expected();speak(`Nice try! Show ${WORDS[n]} ${n===1?'finger':'fingers'}, please!`);
    } else speak('Good try! Remember the next number in your pattern. You can do it!');
  }
  function showTimer() {
    const box=$('challengeTimer');if(!box)return;
    const enabled=Number.isFinite(TIME_LIMITS[gameMode]);
    box.hidden=!enabled;
    if(!enabled)return;
    const total=secondsLeft===null?TIME_LIMITS[gameMode]:secondsLeft;
    $('timerValue').textContent=`${Math.floor(total/60)}:${String(total%60).padStart(2,'0')}`;
    box.classList.toggle('timer-urgent',total<=12);
    $('timerNote').textContent=running?(memoryRevealing?'Listen first — timer starts after the pattern':'Take your time! You can retry if needed'):'Timer begins when your camera starts';
  }
  function clearChallengeTimer() {
    if(timerInterval!==null){clearInterval(timerInterval);timerInterval=null;}
    deadline=0;secondsLeft=null;showTimer();
  }
  function startChallengeTimer() {
    if(!running||complete||memoryRevealing||!TIME_LIMITS[gameMode])return;
    clearChallengeTimer();secondsLeft=TIME_LIMITS[gameMode];deadline=Date.now()+secondsLeft*1000;showTimer();
    let lastTimerTick=Date.now();
    timerInterval=setInterval(()=>{
      const now=Date.now();
      if(!running||document.hidden){deadline+=now-lastTimerTick;lastTimerTick=now;return;}
      lastTimerTick=now;
      const remain=Math.max(0,Math.ceil((deadline-now)/1000));
      if(remain===secondsLeft)return;
      secondsLeft=remain;showTimer();
      if(remain===0){
        clearChallengeTimer();
        feedback('Time for a little reset! You keep your stars. Let’s try again!','warn');
        speak('That is okay! Let’s try one more time!');
        pauseUntil=Date.now()+1900;
        if(gameMode==='memory')setTimeout(()=>{if(running&&gameMode==='memory'&&!complete)revealMemory();},2100);
        else setTimeout(()=>{if(running&&gameMode==='math'&&!complete){lastSpokenAt=0;announce();startChallengeTimer();}},2100);
      }
    },300);
  }
  function currentInstruction() {
    if (complete) return 'Amazing! You finished the game. You are a star!';
    if (gameMode==='count') {
      const n=expected(); return `Let's count! ${WORDS[n]}! Show ${WORDS[n]} ${n===1?'finger':'fingers'}, please!`; 
    }
    if (gameMode==='match') return `Let's copy a hand sign! ${POSE_NAMES[expected()]} gesture, please!`; 
    if (gameMode==='math') {
      const q=plan[round];return `You can do it! ${WORDS[q.a]} ${q.op==='+'?'plus':'minus'} ${WORDS[q.b]}. What do we get? Show your answer with your fingers!`; 
    }
    if (memoryRevealing) return 'Remember the sequence. '+plan[round].map(n=>WORDS[n]).join(', ')+'.';
    return 'Now show the numbers in the same order. Start with the first one!';
  }
  function announce() {
    if (complete) return;
    const now=Date.now();
    if (now-lastSpokenAt<350) return;
    lastSpokenAt=now;lastPromptAt=now;
    speak(currentInstruction());
  }
  function render() {
    const total=totalRounds() || 5;
    $('stars').innerHTML=`${stars} <span>/ ${total}</span>`;
    $('round').innerHTML=`${Math.min(round+1,total)} <span>/ ${total}</span>`;
    const progress=$('progressStars');progress.replaceChildren();
    for(let i=0;i<total;i++) {
      const s=document.createElement('span');s.textContent=i<stars?'★':'☆';
      if(i<stars)s.className='filled';progress.append(s);
    }
    $('gameTypeLabel').textContent=({count:'COUNT & SHINE',match:'COPY THE POSE',math:'FINGER MATH',memory:'MEMORY TRAIL'})[gameMode];
    const titles={count:'HOW MANY FINGERS?',match:'COPY THIS GESTURE',math:'SOLVE WITH FINGERS',memory:'FOLLOW THE PATTERN'};
    $('challengeTitle').textContent=complete?'YOU DID IT!':titles[gameMode];
    $('challengeSub').textContent=complete?'A wonderful job! Play again anytime.':
      (gameMode==='memory'?'Look, remember, then show each number in order!': 'Show the answer with one hand.');
    const orbit=$('numberOrbit'),word=$('spokenWord'),image=$('posePicture'),pattern=$('memoryPattern'),prompt=$('promptText');
    orbit.hidden=word.hidden=image.hidden=pattern.hidden=false;
    if(complete) {
      orbit.hidden=false; $('targetNumber').textContent='★';
      word.hidden=true;image.hidden=true;pattern.hidden=true;
      prompt.textContent=`You earned ${stars} stars. Try another game!`;
      $('repeatSequence').hidden=true;return;
    }
    if(gameMode==='count') {
      const n=expected();$('targetNumber').textContent=String(n);word.textContent=WORDS[n].toUpperCase();
      image.hidden=true;pattern.hidden=true;
      prompt.textContent=`Show ${WORDS[n]} ${n===1?'finger':'fingers'} and listen to the number!`;
    } else if(gameMode==='match') {
      orbit.hidden=true;word.textContent=POSE_NAMES[expected()].toUpperCase();image.hidden=false;pattern.hidden=true;
      image.src=`/static/img/pose-guides/${plan[round][1]}.png`;
      image.alt=`New illustrated guide: ${POSE_NAMES[expected()]} hand pose`;
      prompt.textContent='Make the same hand sign. Hold it steady!';
    } else if(gameMode==='math') {
      const q=plan[round];orbit.hidden=false;orbit.classList.add('math-orbit');
      $('targetNumber').textContent=`${q.a}${q.op}${q.b}`;
      word.textContent='WHAT IS THE ANSWER?';image.hidden=true;pattern.hidden=true;
      prompt.textContent='Show the answer with your fingers!';
    } else {
      orbit.hidden=true;word.textContent=memoryRevealing?'WATCH AND REMEMBER':'YOUR TURN!';image.hidden=true;pattern.hidden=false;
      const seq=plan[round];pattern.replaceChildren();
      seq.forEach((n,i)=>{
        const tile=document.createElement('span');tile.className='memory-tile';
        if(memoryRevealing) tile.textContent=n;
        else if(i<memoryIndex) {tile.textContent='✓';tile.classList.add('done');}
        else {tile.textContent='?';tile.classList.add('hidden-number');}
        pattern.append(tile);
      });
      prompt.textContent=memoryRevealing?'Remember these numbers!':`Show number ${memoryIndex+1} of ${seq.length} from memory.`;
    }
    if(gameMode!=='math')orbit.classList.remove('math-orbit');
    $('repeatSequence').hidden=gameMode!=='memory';
    showTimer();
  }
  function clearReveal() {if(revealTimer!==null){clearTimeout(revealTimer);revealTimer=null;}}
  function revealMemory() {
    clearReveal();clearChallengeTimer();
    if(gameMode!=='memory'||complete)return;
    const thisReveal=++revealSerial;
    memoryRevealing=true;candidate=null;consecutive=0;render();
    const words=plan[round].map(n=>WORDS[n]).join(', ');
    feedback('Listen carefully! The number sequence is read twice before your turn.','good');
    // Read the entire pattern twice. Keep it visible while speech finishes (or until the fallback).
    const version=runVersion, whichRound=round, started=Date.now();
    const minRevealMs=Math.max(6200,plan[round].length*1400+2300);
    const finishReveal=()=>{
      if(version!==runVersion||whichRound!==round||thisReveal!==revealSerial||!memoryRevealing)return;
      clearReveal();memoryRevealing=false;pauseUntil=Date.now()+700;render();
      feedback('Your turn! Show each number in order. You can replay the sequence if needed.','good');
      speak('Your turn! Show the numbers in the same order!');
      startChallengeTimer();
    };
    const spoken=`Listen closely, little star! First time: ${words}. Now, one more time! ${words}. Great remembering!`;
    lastPromptAt=Date.now();
    const voiceDone=()=>{
      if(version!==runVersion||whichRound!==round||thisReveal!==revealSerial||!memoryRevealing)return;
      clearReveal();
      revealTimer=setTimeout(finishReveal,Math.max(0,minRevealMs-(Date.now()-started)));
    };
    // Speech callbacks determine the natural end; the timer is a fallback if an OS voice stalls.
    revealTimer=setTimeout(finishReveal,Math.max(18000,plan[round].length*4500));
    speak(spoken,voiceDone);
  }
  function resetGame(mode=gameMode) {
    runVersion+=1;clearReveal();clearChallengeTimer();lastWrongCueAt=0;
    gameMode=mode;round=0;stars=0;complete=false;memoryIndex=0;memoryRevealing=false;
    candidate=null;consecutive=0;previousAccepted=null;requireChange=false;pauseUntil=0;
    plan=mode==='count'?shuffle([1,2,3,4,5]):mode==='match'?shuffle(POSES):mode==='math'?mathProblems():[sequence(2),sequence(3),sequence(4)];
    document.querySelectorAll('[data-game]').forEach(b=>{
      const selected=b.dataset.game===mode;
      b.classList.toggle('active',selected);b.setAttribute('aria-pressed',String(selected));
    });
    $('lastReading').textContent=stream?'Waiting for the next hand sign':'Camera not started';
    feedback('New game! Start your camera and follow the prompt. Your hand is checked automatically.');
    render();
    if(mode==='memory')revealMemory();
    else {announce();startChallengeTimer();}
  }
  function startNextRound() {
    clearChallengeTimer();pauseUntil=Date.now()+1100;candidate=null;consecutive=0;lastWrongCueAt=0;
    if(gameMode==='memory') {
      const version=runVersion;render();
      setTimeout(()=>{if(runVersion===version&&!complete)revealMemory();},950);
    } else {render();const version=runVersion;setTimeout(()=>{if(runVersion===version&&!complete){announce();startChallengeTimer();}},950);}
  }
  function acceptGesture(recognized) {
    if(memoryRevealing||complete||Date.now()<pauseUntil)return;
    if(requireChange) {
      if(recognized!==previousAccepted) requireChange=false;
      else {feedback('Great! Change your fingers or lower your hand briefly for the next answer.','good');return;}
    }
    if (recognized!==expected()) {
      candidate=null;consecutive=0;
      feedback('Not quite yet. Try the prompt again and hold your hand steady.','warn');
      return;
    }
    if(candidate===recognized)consecutive+=1;
    else {candidate=recognized;consecutive=1;}
    if(consecutive<2){feedback('I see it! Hold that pose for one more check.','good');return;}
    previousAccepted=recognized;requireChange=true;candidate=null;consecutive=0;
    if(gameMode==='memory') {
      memoryIndex+=1;
      if(memoryIndex<plan[round].length) {
        render();feedback('Wonderful! Remember the next number in the sequence.','good');
        speak('Great job! Next number.');pauseUntil=Date.now()+1100;return;
      }
      memoryIndex=0;
    }
    stars+=1;round+=1;
    if(round>=plan.length) {
      complete=true;clearReveal();clearChallengeTimer();render();feedback(`Hooray! ${stars} stars earned. You finished the game!`,'good');
      speak('Amazing! You did it! You are a counting champion!');
    } else {
      render();feedback('Awesome! You earned a star! Get ready for the next challenge.','good');
      speak('Well done! You earned a star!');
      startNextRound();
    }
  }
  function processPrediction(details, status) {
    if(!details) return;
    const core=gameMode==='match';
    const modelAvailable=core?status?.mode==='trained':status?.number_model?.mode==='trained';
    if(!modelAvailable) {
      feedback('Trained weights for this game are missing. Please check your models folder.','bad');
      $('lastReading').textContent='Model unavailable';return;
    }
    if(!details.hand_count) {
      $('lastReading').textContent='No hand detected';
      $('readyText').textContent='Move your hand back; show the wrist and palm';
      $('readyDot').classList.remove('active');
      requireChange=false;candidate=null;consecutive=0;
      if(!memoryRevealing && !complete) feedback('I cannot see your hand yet. Move back and use brighter front lighting.','warn');
      return;
    }
    const recognized=core?String(details.gesture||''):(Number.isInteger(details.number_prediction?.number)?details.number_prediction.number:null);
    const conf=core?details.confidence:details.number_prediction?.confidence;
    $('lastReading').textContent=recognized===null?'Unknown gesture':`${core?(POSE_NAMES[recognized]||'Unknown pose'):(WORDS[recognized]||'Unknown')} · ${typeof conf==='number'?Math.round(conf*100)+'% votes':'unscored'}`;
    $('readyText').textContent='Hand found · observing gesture';$('readyDot').classList.add('active');
    if(typeof conf==='number')coachWrongGesture(recognized,conf);
    if(typeof conf!=='number'||conf<MIN_VOTE[gameMode]) {
      candidate=null;consecutive=0;
      if(!memoryRevealing&&!complete) feedback('Keep your hand still and in good light. Trying to recognize your sign…','warn');
      return;
    }
    acceptGesture(recognized);
  }
  async function frameBlob() {
    if(!stream || !video.videoWidth) throw new Error('Camera is not ready yet.');
    const canvas=document.createElement('canvas');
    const ratio=Math.min(1,520/video.videoWidth);
    canvas.width=Math.max(1,Math.round(video.videoWidth*ratio));
    canvas.height=Math.max(1,Math.round(video.videoHeight*ratio));
    const ctx=canvas.getContext('2d');
    ctx.translate(canvas.width,0);ctx.scale(-1,1);
    ctx.drawImage(video,0,0,canvas.width,canvas.height);
    return new Promise((resolve,reject)=>canvas.toBlob(blob=>blob?resolve(blob):reject(Error('Could not capture the live frame')),'image/jpeg',.76));
  }
  function scheduleTick(delay=SCAN_PAUSE_MS) {
    if(pollTimer!==null)clearTimeout(pollTimer);
    if(running)pollTimer=setTimeout(scanTick,delay);
  }
  async function scanTick() {
    if(!running || !stream) return;
    if(inFlight || document.hidden){scheduleTick(950);return;}
    inFlight=true;const version=runVersion;
    try {
      const blob=await frameBlob();
      const form=new FormData();form.append('image',blob,'live-hand.jpg');
      const path=gameMode==='match'?'/api/live?mode=core':'/api/live?mode=numbers';
      const response=await fetch(path,{method:'POST',body:form,cache:'no-store'});
      const body=await response.json();
      if(!response.ok)throw Error(body.detail||`Server response ${response.status}`);
      if(version===runVersion && running)processPrediction(body.details,body.status);
    } catch(error) {
      feedback(error.message||'Could not recognize the live camera image.','bad');
      $('readyDot').classList.remove('active');
    } finally {inFlight=false;if(running)scheduleTick();}
  }
  async function startCamera() {
    if(stream)return;
    if(!navigator.mediaDevices?.getUserMedia) {
      feedback('Camera access requires localhost or HTTPS and a supported browser.','bad');return;
    }
    const button=$('startPlayCamera');button.disabled=true;
    // The button click is the user's audio gesture: speak while it is active.
    lastSpokenAt=0;announce();
    try {
      const s=await navigator.mediaDevices.getUserMedia({audio:false,video:{width:{ideal:640},height:{ideal:480},facingMode:'user'}});
      stream=s;video.srcObject=s;await video.play();
      video.hidden=false;$('cameraPlaceholder').hidden=true;$('scanIndicator').hidden=false;
      $('cameraState').textContent='LIVE · AUTOMATIC';
      $('startPlayCamera').hidden=true;$('stopPlayCamera').disabled=false;
      $('lastReading').textContent='Looking for your hand…';
      $('readyText').textContent='Camera is live · checking automatically';
      running=true;feedback('Camera ready! Show the gesture for the challenge.','good');
      if(gameMode==='memory')revealMemory();else {announce();startChallengeTimer();}
      scheduleTick(300);
      s.getVideoTracks().forEach(track=>track.addEventListener('ended',stopCamera,{once:true}));
    } catch(error) {feedback('Camera permission denied or camera unavailable. Allow access and try again.','bad');button.disabled=false;}
  }
  function stopCamera() {
    running=false;clearChallengeTimer();if(pollTimer!==null){clearTimeout(pollTimer);pollTimer=null;}
    if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;}
    video.srcObject=null;video.hidden=true;$('cameraPlaceholder').hidden=false;
    $('scanIndicator').hidden=true;$('cameraState').textContent='CAMERA OFF';
    $('startPlayCamera').hidden=false;$('startPlayCamera').disabled=false;
    $('stopPlayCamera').disabled=true;$('readyText').textContent='Camera off';
    $('readyDot').classList.remove('active');$('lastReading').textContent='Camera stopped';
    requireChange=false;candidate=null;consecutive=0;
  }
  document.querySelectorAll('[data-game]').forEach(button=>button.addEventListener('click',()=>{
    if(button.dataset.game!==gameMode)resetGame(button.dataset.game);
  }));
  $('startPlayCamera').addEventListener('click',startCamera);
  $('stopPlayCamera').addEventListener('click',()=>{stopCamera();feedback('Camera stopped. Start it again when you are ready.');});
  $('newGame').addEventListener('click',()=>resetGame());
  $('repeatPrompt').addEventListener('click',()=>{lastSpokenAt=0;announce();});
  $('repeatSequence').addEventListener('click',()=>{if(gameMode==='memory')revealMemory();});
  $('voiceChoice').addEventListener('change',e=>{chosenVoice=e.target.value;speak('Hello, little star! Let’s play and learn together!');});
  document.addEventListener('visibilitychange',()=>{if(!document.hidden && running)scheduleTick(200);});
  window.addEventListener('pagehide',stopCamera);
  fetch('/api/status').then(r=>r.ok?r.json():Promise.reject(Error('Status unavailable'))).then(s=>{
    modelReady={core:s.mode==='trained',numbers:s.number_model?.mode==='trained'};
    if(!modelReady.core||!modelReady.numbers)feedback('Some HaGRID model files are missing. Check the models folder before playing.','warn');
  }).catch(()=>feedback('Please start your local Python server to play.','bad'));
  const initialGame=new URLSearchParams(window.location.search).get('game');
  resetGame(['count','match','math','memory'].includes(initialGame)?initialGame:'count');
})();
