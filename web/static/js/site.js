/* Original Blurry Vision navigation + opt-in original instrumental background music. */
(() => {
  'use strict';
  const header=document.querySelector('.site-header');
  const menu=document.getElementById('mobileNav');
  const toggle=document.querySelector('.mobile-menu');
  if(toggle&&menu) {
    toggle.addEventListener('click',()=>{
      const isOpen=toggle.getAttribute('aria-expanded')==='true';
      toggle.setAttribute('aria-expanded',String(!isOpen));menu.hidden=isOpen;
    });
    document.addEventListener('keydown',e=>{
      if(e.key==='Escape'&&!menu.hidden){menu.hidden=true;toggle.setAttribute('aria-expanded','false');toggle.focus();}
    });
    document.addEventListener('click',e=>{
      if(!menu.hidden&&!menu.contains(e.target)&&!toggle.contains(e.target)){
        menu.hidden=true;toggle.setAttribute('aria-expanded','false');
      }
    });
  }
  if(!header)return;
  const button=document.createElement('button');
  button.type='button';button.className='music-toggle';button.id='musicToggle';
  button.setAttribute('aria-pressed','false');
  button.setAttribute('aria-label','Play original background music');
  button.title='Optional original background music';
  button.innerHTML='<span aria-hidden="true">♫</span> <span class="music-text">MUSIC OFF</span>';
  header.insertBefore(button,header.querySelector('.nav-action')||toggle);
  const music=new Audio('/static/audio/star_steps_original.wav');
  music.loop=true;music.volume=0.17;music.preload='none';
  let playing=false;let ducked=false;
  function reflect(){
    button.setAttribute('aria-pressed',String(playing));
    button.setAttribute('aria-label',playing?'Pause background music':'Play original background music');
    button.querySelector('.music-text').textContent=playing?'MUSIC ON':'MUSIC OFF';
    button.classList.toggle('playing',playing);
  }
  button.addEventListener('click',async()=>{
    if(playing){music.pause();playing=false;reflect();return;}
    try {await music.play();playing=true;music.volume=ducked?0.04:0.17;reflect();}
    catch {playing=false;reflect();button.title='Browser prevented audio playback. Click again.';}
  });
  window.BlurryAudio={duck(value){ducked=Boolean(value);music.volume=ducked?0.04:0.17;}};
  const spoken=['zero','one','two','three','four','five'];
  document.querySelectorAll('[data-say-number]').forEach(node=>node.addEventListener('click',()=>{
    if(!window.speechSynthesis)return;
    const num=Number(node.dataset.sayNumber);
    if(!Number.isInteger(num)||num<1||num>5)return;
    window.speechSynthesis.cancel();
    const utterance=new SpeechSynthesisUtterance(spoken[num]);
    utterance.lang='en-US';utterance.rate=0.8;utterance.pitch=1.07;
    utterance.onstart=()=>window.BlurryAudio.duck(true);
    utterance.onend=()=>window.BlurryAudio.duck(false);
    window.speechSynthesis.speak(utterance);
  }));
  document.addEventListener('visibilitychange',()=>{if(document.hidden&&playing){music.pause();playing=false;reflect();}});
})();
