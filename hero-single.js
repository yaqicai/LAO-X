'use strict';

// Previous single-scan timeline, using only the retained real predictions.
(() => {
  const film=document.querySelector('.hero-film'),data=window.LAOX_REAL;
  const q=s=>film.querySelector(s);
  const scene=q('.hero-film-scene'),base=q('.hero-source-image'),mask=q('.hero-real-mask'),overlay=q('.hero-film-overlay');
  const dot=q('#hero-prompt'),ring=q('#hero-click-ring'),cursor=q('.hero-film-cursor');
  const rail=q('.hero-granularity-track'),fill=q('.hero-granularity-fill'),knob=q('.hero-granularity-knob'),value=q('.hero-granularity-value');
  const reduce=matchMedia('(prefers-reduced-motion: reduce)');
  const sequence=[['DvXray','4214',0],['det_compass_xp','1873',0],['pidray','7420',0]].map(([dataset,id,pi])=>{
    const scan=data.cases.find(c=>c.dataset===dataset&&String(c.id)===id);
    return {scan,prompt:scan.prompts[pi]};
  });
  const duration=14500,clamp=n=>Math.max(0,Math.min(1,n));
  const ease=n=>{const t=clamp(n);return t*t*(3-2*t);},lerp=(a,b,t)=>a+(b-a)*t;
  let elapsed=reduce.matches?8800:0,previous=null,frame=null,selected=-1,shown=-1;
  let visible=true,ready=false,paused=reduce.matches,geometry={};
  function measure(){
    const parent=film.getBoundingClientRect(),picture=scene.getBoundingClientRect(),track=rail.getBoundingClientRect();
    const current=sequence[Math.max(0,selected)];
    geometry={x:picture.left-parent.left,y:picture.top-parent.top,scale:picture.width/current.scan.width,railX:track.left-parent.left,railY:track.top-parent.top+track.height/2,railWidth:track.width};
  }
  function setObject(index){
    selected=index;shown=-1;
    const {scan,prompt}=sequence[index];
    scene.style.aspectRatio=`${scan.width} / ${scan.height}`;
    base.src=scan.image;base.width=scan.width;base.height=scan.height;
    overlay.setAttribute('viewBox',`0 0 ${scan.width} ${scan.height}`);
    for(const node of [dot,ring]){node.setAttribute('cx',prompt.point[0]);node.setAttribute('cy',prompt.point[1]);}
    film.dataset.object=`${scan.dataset}/${scan.id}`;
    q('.hero-granularity-max').textContent=prompt.outputs.at(-1).granularity.toFixed(1);
    measure();
  }
  function render(){
    const index=Math.floor(elapsed/duration)%sequence.length;
    if(index!==selected)setObject(index);
    const t=elapsed%duration,{scan,prompt}=sequence[index],g=geometry;
    const best=prompt.outputs[prompt.best_output].granularity,max=prompt.outputs.at(-1).granularity;
    const target=[g.x+prompt.point[0]*g.scale,g.y+prompt.point[1]*g.scale];
    const atRail=v=>[g.railX+g.railWidth*(v-.1)/(max-.1),g.railY];
    let grain=.1,pointer=target,phase='select',opacity=1;
    if(t<1400){const u=ease(t/1400);pointer=[lerp(target[0]+115,target[0],u),lerp(target[1]+90,target[1],u)];}
    else if(t<2800){phase='local';}
    else if(t<4000){phase='adjust';const u=ease((t-2800)/1200),start=atRail(.1);pointer=[lerp(target[0],start[0],u),lerp(target[1],start[1],u)];}
    else if(t<8500){phase='grow';grain=lerp(.1,best,ease((t-4000)/4500));pointer=atRail(grain);}
    else if(t<10300){phase='hold';grain=best;pointer=atRail(grain);}
    else if(t<12100){phase='shrink';grain=lerp(best,Math.min(.3,best),ease((t-10300)/1800));pointer=atRail(grain);}
    else if(t<13300){phase='grow';grain=lerp(Math.min(.3,best),best,ease((t-12100)/1200));pointer=atRail(grain);}
    else{phase='transition';grain=best;opacity=1-ease((t-13700)/800);const release=ease((t-13300)/400),end=atRail(best);pointer=[end[0]+15*release,end[1]+35*release];}
    const oi=Math.min(prompt.outputs.length-1,Math.max(0,Math.round(grain*10)-1)),output=prompt.outputs[oi];
    if(oi!==shown){mask.src=output.display;shown=oi;}
    const reveal=ease((t-1400)/400)*opacity;
    mask.style.opacity=String(reveal*.65);dot.style.opacity=String(reveal);dot.setAttribute('r',scan.width/130);
    const ripple=clamp((t-1400)/850);ring.setAttribute('r',(8+ripple*30)*scan.width/800);ring.style.opacity=String(t>=1400&&t<2250?1-ripple:0);
    const percent=(grain-.1)/(max-.1)*100;fill.style.width=`${percent}%`;knob.style.left=`${percent}%`;value.textContent=output.granularity.toFixed(1);
    cursor.style.transform=`translate3d(${pointer[0]}px,${pointer[1]}px,0) scale(${t>1400&&t<1600?.85:1})`;cursor.style.opacity=String(opacity);
    film.dataset.phase=phase;film.dataset.granularity=output.granularity.toFixed(1);
  }
  function tick(time){
    frame=null;if(paused||!visible||document.hidden||!ready)return;
    if(previous!==null)elapsed+=Math.min(time-previous,100);
    previous=time;render();frame=requestAnimationFrame(tick);
  }
  function sync(){
    if(frame!==null)cancelAnimationFrame(frame);frame=null;previous=null;
    const running=!paused&&visible&&!document.hidden&&ready;film.dataset.playing=String(running);
    if(running)frame=requestAnimationFrame(tick);
  }
  document.addEventListener('visibilitychange',sync);
  reduce.addEventListener('change',e=>{paused=e.matches;if(paused)elapsed=8800;render();sync();});
  new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;sync();}).observe(film);
  new ResizeObserver(()=>{measure();render();}).observe(film);
  measure();render();sync();
  const images=[];
  Promise.all(sequence.flatMap(s=>[s.scan.image,...s.prompt.outputs.map(o=>o.display)]).map(src=>{const im=new Image();im.src=src;images.push(im);return im.decode();}))
    .then(()=>{ready=true;sync();}).catch(()=>{film.dataset.loadError='true';film.setAttribute('aria-label','Animation unavailable. Please reload.');});
})();
