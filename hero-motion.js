'use strict';

// Operator workstation → monitor zoom → recorded predictions with visual crossfades.
(() => {
  const film = document.querySelector('.hero-film');
  const data = window.LAOX_REAL;
  const container = film.querySelector('.hero-panels');
  const scanner = film.querySelector('.scanner-scene');
  const luggage = film.querySelector('#ct-luggage');
  const beam = film.querySelector('#ct-scan-beam');
  const rollers = film.querySelector('#ct-rollers');
  const screen = film.querySelector('#ct-monitor-screen');
  const hand = film.querySelector('#ct-operator-hand');
  const instruction = film.querySelector('#film-instruction-text');
  let monitor = {};
  const reduce = matchMedia('(prefers-reduced-motion: reduce)');
  const clamp = n => Math.max(0, Math.min(1, n));
  const ease = n => { const t = clamp(n); return t*t*(3-2*t); };
  const lerp = (a,b,t) => a+(b-a)*t;
  const duration = 22000;
  let elapsed = reduce.matches ? 16400 : 0;
  let previous = null, frame = null, paused = reduce.matches, visible = true, ready = false;
  const choices = [['DvXray','4214',0],['PIXray','101556960',0],['pidray','7420',0]];
  const panels = choices.map(([dataset,id,pi]) => {
    const scan = data.cases.find(c => c.dataset === dataset && String(c.id) === id);
    const prompt = scan.prompts[pi];
    const name = data.datasets.find(d => d.id === dataset).name;
    const card = document.createElement('div'); card.className = 'film-panel';
    card.innerHTML = `<div class="film-panel-heading"><span>${name}</span><i></i></div><div class="film-picture"><img class="film-source" alt=""><div class="film-masks"></div><svg><circle class="film-click" fill="none" stroke="white" stroke-width="3"/><circle class="film-dot" fill="#2476ef" stroke="white" stroke-width="2"/></svg></div><div class="film-panel-control"><div class="film-slider-label"><span>Granularity</span><span class="film-g-value">0.1</span></div><div class="film-rail"><span class="film-fill"></span><span class="film-knob"></span></div></div><div class="film-cursor"><svg viewBox="0 0 40 52"><path d="M4 3v37l10-10 10 18 8-5-11-17h15Z"/></svg></div>`;
    container.append(card);
    const q = s => card.querySelector(s);
    q('.film-source').src = scan.image;
    const masks = prompt.outputs.map(output => {
      const image = document.createElement('img');
      image.className = 'film-mask'; image.alt = ''; image.src = output.display;
      q('.film-masks').append(image); return image;
    });
    q('.film-picture svg').setAttribute('viewBox',`0 0 ${scan.width} ${scan.height}`);
    for (const node of [q('.film-dot'),q('.film-click')]) {
      node.setAttribute('cx',prompt.point[0]); node.setAttribute('cy',prompt.point[1]);
    }
    q('.film-dot').setAttribute('r',Math.max(scan.width,scan.height)/110);
    card.dataset.scan = `${dataset}/${id}`;
    return {scan,prompt,card,picture:q('.film-picture'),masks,weights:masks.map(()=>0),from:masks.map(()=>0),changedAt:0,dot:q('.film-dot'),ring:q('.film-click'),rail:q('.film-rail'),fill:q('.film-fill'),knob:q('.film-knob'),value:q('.film-g-value'),cursor:q('.film-cursor'),shown:-1,geometry:{}};
  });
  function measure() {
    scanner.querySelector('.ct-scanner').setAttribute('viewBox',matchMedia('(max-width:640px)').matches?'0 0 600 620':'0 0 1280 400');
    scanner.style.transform = 'none';
    const f = film.getBoundingClientRect(), r = screen.getBoundingClientRect();
    const fit=Math.min(r.width/container.offsetWidth,r.height/container.offsetHeight);
    monitor = {x:r.left-f.left-container.offsetLeft+(r.width-container.offsetWidth*fit)/2,
      y:r.top-f.top-container.offsetTop+(r.height-container.offsetHeight*fit)/2,sx:fit,sy:fit};
    const scene = scanner.getBoundingClientRect();
    scanner.style.transformOrigin = `${r.left-scene.left+r.width/2}px ${r.top-scene.top+r.height/2}px`;
    panels.forEach(p => {
      const scale = Math.min(p.picture.offsetWidth/p.scan.width,p.picture.offsetHeight/p.scan.height);
      p.geometry = {
        x:p.picture.offsetLeft+(p.picture.offsetWidth-p.scan.width*scale)/2+p.prompt.point[0]*scale,
        y:p.picture.offsetTop+(p.picture.offsetHeight-p.scan.height*scale)/2+p.prompt.point[1]*scale,
        railX:p.rail.offsetLeft,railY:p.rail.offsetTop+p.rail.offsetHeight/2,railWidth:p.rail.offsetWidth
      };
    });
  }
  // Fast curved travel with a small final correction, independent of slider timing.
  function move(from,to,progress,bend=10) {
    const t=clamp(progress), u=t<.8 ? .98*(1-Math.pow(1-t/.8,3)) : .98+.02*ease((t-.8)/.2);
    return [lerp(from[0],to[0],u)+Math.sin(Math.PI*u)*bend,
      lerp(from[1],to[1],u)-Math.sin(Math.PI*u)*bend*.55];
  }
  function renderPanel(p, i, t) {
    p.card.style.opacity = '1';
    const local = t-6500-i*160;
    const best = p.prompt.outputs[p.prompt.best_output].granularity;
    const max = p.prompt.outputs.at(-1).granularity;
    const g = p.geometry;
    const target=[g.x,g.y], atRail=v=>[g.railX+g.railWidth*(v-.1)/(max-.1),g.railY];
    let grain=.1, pointer=target;
    if(local<460) pointer=move([g.x+56,g.y+38],target,local/460,7);
    else if(local<1200) pointer=target;
    else if(local<1850) pointer=move(target,atRail(.1),(local-1200)/650,-14);
    else if(local<7750) {
      grain=lerp(.1,best,ease((local-1850)/5900));pointer=atRail(grain);
    } else if(local<9050) {grain=best;pointer=atRail(grain);
    } else if(local<10650) {
      grain=lerp(best,Math.min(.3,best),ease((local-9050)/1600));pointer=atRail(grain);
    } else {grain=lerp(Math.min(.3,best),best,ease((local-10650)/1400));pointer=atRail(grain);}
    const index=Math.min(p.prompt.outputs.length-1,Math.max(0,Math.round(grain*10)-1));
    const output=p.prompt.outputs[index];
    // Blend only opacity between actual masks; never generate or alter their geometry.
    if(local<0) {p.shown=-1;p.weights.fill(0);p.from.fill(0);}
    if(index!==p.shown){p.from=[...p.weights];p.changedAt=local;p.shown=index;}
    const blend=paused?1:ease((local-p.changedAt)/280),maskReveal=ease((local-560)/260);
    p.masks.forEach((image,j)=>{
      p.weights[j]=lerp(p.from[j],j===index?1:0,blend);
      image.style.opacity=String(p.weights[j]);
    });
    p.card.querySelector('.film-masks').style.opacity=String(.65*maskReveal);
    p.dot.style.opacity=String(maskReveal);
    const ripple=clamp((local-560)/650);
    p.ring.setAttribute('r',String((8+32*ripple)*Math.max(p.scan.width,p.scan.height)/800));
    p.ring.style.opacity=String(local>=560&&local<1210?1-ripple:0);
    const percent=(grain-.1)/(max-.1)*100;
    p.fill.style.width=`${percent}%`;p.knob.style.left=`${percent}%`;p.value.textContent=output.granularity.toFixed(1);
    p.cursor.style.opacity=String(ease(local/180));
    p.cursor.style.transform=`translate(${pointer[0]}px,${pointer[1]}px) scale(${local>=560&&local<700?.85:1})`;
    p.card.dataset.granularity=output.granularity.toFixed(1);
    p.card.dataset.blending=String(blend<1&&local>560);
  }
  function render() {
    const t=elapsed%duration,zoom=ease((t-3800)/2400),endFade=1-ease((t-21000)/1000);
    const travel=ease((t-350)/3700);
    luggage.setAttribute('transform',`translate(${lerp(140,660,travel)} 171)`);
    luggage.style.opacity=String(1-ease((travel-.76)/.24));
    scanner.style.opacity=String((1-ease((zoom-.15)/.65))*endFade);
    scanner.style.transform=`scale(${1+zoom*2.8})`;
    hand.setAttribute('transform',`translate(${Math.sin(t/260)*1.3} ${Math.sin(t/180)*.7})`);
    beam.style.opacity=String(t>2000&&t<4200?(.4+.3*Math.sin(t/140))*ease((t-2000)/500):0);
    rollers.setAttribute('patternTransform',`translate(${(t/35)%24} 0)`);
    container.style.transform=`translate(${monitor.x*(1-zoom)}px,${monitor.y*(1-zoom)}px) scale(${lerp(monitor.sx,1,zoom)},${lerp(monitor.sy,1,zoom)})`;
    container.style.opacity=String(endFade);
    panels.forEach((p,i)=>renderPanel(p,i,t));
    const phase=t<3800?'inspection':t<6500?'monitor':t<8050?'select':'adjust';
    film.dataset.phase=phase;
    const text=phase==='inspection'||phase==='monitor'?'Review the scans':phase==='select'?'Click to select':'Drag to refine';
    if(instruction.textContent!==text)instruction.textContent=text;
    instruction.parentElement.style.opacity=String(endFade);
  }
  function tick(time) {
    frame=null;
    if(paused||!visible||document.hidden||!ready)return;
    if(previous!==null)elapsed+=Math.min(time-previous,100);
    previous=time;render();frame=requestAnimationFrame(tick);
  }
  function sync() {
    if(frame!==null)cancelAnimationFrame(frame);
    frame=null;previous=null;
    const running=!paused&&visible&&!document.hidden&&ready;
    film.dataset.playing=String(running);
    if(running)frame=requestAnimationFrame(tick);
  }
  document.addEventListener('visibilitychange',sync);
  reduce.addEventListener('change',e=>{paused=e.matches;if(paused)elapsed=16400;render();sync();});
  new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;sync();}).observe(film);
  new ResizeObserver(()=>{measure();render();}).observe(film);
  measure();render();sync();
  const loaded=[];
  Promise.all(panels.flatMap(p=>[p.scan.image,...p.prompt.outputs.map(o=>o.display)]).map(src=>{
    const image=new Image();image.src=src;loaded.push(image);return image.decode();
  })).then(()=>{ready=true;sync();}).catch(()=>{film.dataset.loadError='true';film.setAttribute('aria-label','Animation unavailable. Please reload.');});
})();
