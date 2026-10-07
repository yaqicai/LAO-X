'use strict';

// Three short, isolated walkthroughs. They never change the explorer selection.
(() => {
  const data = window.LAOX_REAL;
  const scan = data.cases.find(c => c.dataset === 'DvXray' && String(c.id) === '4214');
  const prompt = scan.prompts[0];
  const det = data.cases.find(c => c.dataset === 'det_compass_xp');
  const other = data.cases.filter(c => c.dataset === 'DvXray' && c.id !== scan.id).slice(0, 2);
  const thumbnails = [other[0], scan, other[1]];
  const motion = matchMedia('(prefers-reduced-motion: reduce)');
  const opener = document.querySelector('#explorer-help-open');
  const dialog = document.createElement('dialog');
  dialog.id = 'explorer-help'; dialog.className = 'explorer-help-dialog';
  dialog.setAttribute('aria-labelledby', 'explorer-help-title');
  dialog.innerHTML = `<div class="help-heading"><h2 id="explorer-help-title">How to explore</h2><button class="icon-button help-close" aria-label="Close help">×</button></div>
    <div class="help-tabs" role="tablist" aria-label="Tutorial steps">
      <button role="tab" id="help-tab-0" aria-controls="help-content" data-help-step="0">1 <span>Choose a scan</span></button>
      <button role="tab" id="help-tab-1" aria-controls="help-content" data-help-step="1">2 <span>Select a point</span></button>
      <button role="tab" id="help-tab-2" aria-controls="help-content" data-help-step="2">3 <span>Refine the mask</span></button>
    </div>
    <div id="help-content" role="tabpanel">
      <div class="help-demo" aria-hidden="true">
        <div class="help-datasets"><span data-help-dataset="det">DET-COMPASS</span><span data-help-dataset="dv">DvXray</span><span>HiXray</span></div>
        <div class="help-demo-main"><div class="help-picture"><img class="help-source" alt=""><div class="help-masks"></div><svg class="help-points" viewBox="0 0 ${scan.width} ${scan.height}">${scan.prompts.map((p,i)=>`<circle data-help-point="${i}" cx="${p.point[0]}" cy="${p.point[1]}" r="${Math.max(scan.width,scan.height)/65}"/>`).join('')}<circle class="help-click" cx="${prompt.point[0]}" cy="${prompt.point[1]}" r="0"/></svg></div>
        <div class="help-controls"><div class="help-control-title">Point prompt</div><div class="help-point-options"><span>Center</span><span>Point 2</span><span>Point 3</span></div><div class="help-slider-label"><span>Granularity</span><span class="help-grain">0.1</span></div><div class="help-rail"><i class="help-fill"></i><i class="help-knob"></i></div><div class="help-range"><span>0.1</span><span>${prompt.outputs.at(-1).granularity.toFixed(1)}</span></div><div class="help-chart-heading"><span>IoU with GT</span><span class="help-best">★ Best ${prompt.outputs[prompt.best_output].granularity.toFixed(1)}</span></div><svg class="help-chart" viewBox="0 0 200 82"></svg></div></div>
        <div class="help-thumbnails">${thumbnails.map((c,i)=>`<span data-help-thumb="${i}"><img src="${c.thumbnail}" alt=""></span>`).join('')}</div>
        <div class="help-cursor"><svg viewBox="0 0 40 52"><path d="M4 3v37l10-10 10 18 8-5-11-17h15Z"/></svg></div>
      </div>
      <p class="help-caption"></p>
    </div>
    <div class="help-footer"><button class="help-replay" aria-label="Replay this tutorial">↻ Replay</button><button class="button primary help-try">Try it <span>↗</span></button></div>`;
  document.body.append(dialog);
  const q = s => dialog.querySelector(s);
  const picture = q('.help-picture'), source = q('.help-source'), cursor = q('.help-cursor');
  const layers = prompt.outputs.map(o => {
    const image = document.createElement('img'); image.src=o.display; image.alt='';
    q('.help-masks').append(image); return image;
  });
  const x = g => 10+(g-.1)/(prompt.outputs.at(-1).granularity-.1)*180;
  const y = v => 72-v*58;
  q('.help-chart').innerHTML = `<path d="M10 14H190M10 43H190M10 72H190" stroke="#e0e8f2" fill="none"/><polyline points="${prompt.outputs.map(o=>`${x(o.granularity)},${y(o.box_iou)}`).join(' ')}" stroke="#2476ef" stroke-width="2" fill="none"/><text x="${x(prompt.outputs[prompt.best_output].granularity)}" y="${y(prompt.outputs[prompt.best_output].box_iou)-7}" fill="#169879" font-size="14" text-anchor="middle">★</text><circle class="help-chart-current" r="4" fill="#2476ef" stroke="white" stroke-width="2"/>`;
  const captions = ['Choose a dataset, then a thumbnail.', 'Click a marked point to select an object.', 'Drag to change the mask. ★ Best selects the highest GT IoU.'];
  const durations = [4800, 3600, 6800];
  const clamp = n => Math.max(0,Math.min(1,n));
  const ease = n => {const v=clamp(n);return v*v*(3-2*v);};
  const move = (a,b,n) => {const u=1-Math.pow(1-clamp(n),3);return [a[0]+(b[0]-a[0])*u+Math.sin(Math.PI*u)*8,a[1]+(b[1]-a[1])*u];};
  let step=0,elapsed=0,frame=null,previous=null,ready=false,shown=-1;
  function center(el) {
    const r=el.getBoundingClientRect(),b=q('.help-demo').getBoundingClientRect();
    return [r.left-b.left+r.width/2,r.top-b.top+r.height/2];
  }
  function pointPosition() {
    const r=picture.getBoundingClientRect(),b=q('.help-demo').getBoundingClientRect();
    const scale=Math.min(r.width/scan.width,r.height/scan.height);
    return [r.left-b.left+(r.width-scan.width*scale)/2+prompt.point[0]*scale,r.top-b.top+(r.height-scan.height*scale)/2+prompt.point[1]*scale];
  }
  function setSource(c) {if(source.getAttribute('src')!==c.image)source.src=c.image;}
  function showMask(index,visible) {
    if(index!==shown){layers.forEach((el,i)=>{el.style.opacity=i===index?'1':'0';});shown=index;}
    q('.help-masks').style.opacity=visible?'.65':'0';
  }
  function render() {
    const t=elapsed,target=pointPosition();
    let pointer=target,click=false,grain=.1;
    if(step===0) {
      const choseDataset=t>=1100,choseScan=t>=2600;
      setSource(!choseDataset?det:choseScan?scan:other[0]);
      q('[data-help-dataset="det"]').classList.toggle('active',!choseDataset);
      q('[data-help-dataset="dv"]').classList.toggle('active',choseDataset);
      dialog.querySelectorAll('[data-help-thumb]').forEach((el,i)=>el.classList.toggle('active',i===(choseScan?1:0)));
      q('.help-thumbnails').style.opacity=choseDataset?'1':'.25';
      const datasetTarget=center(q('[data-help-dataset="dv"]')),thumbTarget=center(q('[data-help-thumb="1"]'));
      pointer=t<1300?move([datasetTarget[0]+55,datasetTarget[1]+50],datasetTarget,t/900):move(datasetTarget,thumbTarget,(t-1700)/650);
      click=(t>=1100&&t<1260)||(t>=2600&&t<2760);
      showMask(0,false);
    } else {
      setSource(scan);
      q('.help-point-options').classList.toggle('selected',step===2||t>=1000);
      q('.help-points').classList.toggle('selected',step===2||t>=1000);
      const ripple=clamp((t-1000)/600);
      q('.help-click').setAttribute('r',String((9+ripple*30)*scan.width/800));
      q('.help-click').style.opacity=step===1&&t>=1000&&t<1600?String(1-ripple):'0';
      if(step===1) {
        pointer=move([target[0]+80,target[1]+55],target,t/700);
        click=t>=1000&&t<1160;
        grain=prompt.outputs[prompt.best_output].granularity;
        showMask(prompt.best_output,t>=1000);
      } else {
        const rail=q('.help-rail').getBoundingClientRect(),b=q('.help-demo').getBoundingClientRect();
        const railAt=g=>[rail.left-b.left+(g-.1)/(prompt.outputs.at(-1).granularity-.1)*rail.width,rail.top-b.top+rail.height/2];
        const max=prompt.outputs.at(-1).granularity,best=prompt.outputs[prompt.best_output].granularity;
        grain=t<5000?.1+(max-.1)*ease((t-1100)/3800):max;
        pointer=t<1100?move(target,railAt(.1),t/650):railAt(grain);
        if(t>=5000)pointer=move(railAt(max),center(q('.help-best')),(t-5000)/600);
        if(t>=5800)grain=best;
        click=t>=5800&&t<5960;
        const index=Math.min(prompt.outputs.length-1,Math.max(0,Math.round(grain*10)-1));
        showMask(index,true);
      }
    }
    const output=prompt.outputs[Math.min(prompt.outputs.length-1,Math.max(0,Math.round(grain*10)-1))];
    const percent=(grain-.1)/(prompt.outputs.at(-1).granularity-.1)*100;
    q('.help-fill').style.width=`${percent}%`;q('.help-knob').style.left=`${percent}%`;
    q('.help-grain').textContent=output.granularity.toFixed(1);
    q('.help-chart-current').setAttribute('cx',x(output.granularity));
    q('.help-chart-current').setAttribute('cy',y(output.box_iou));
    cursor.style.transform=`translate(${pointer[0]}px,${pointer[1]}px) scale(${click?.84:1})`;
    cursor.style.opacity=ready?'1':'0';
    q('.help-demo').dataset.phase=step===0?(t<1100?'dataset':t<2600?'thumbnail':'selected'):step===1?(t<1000?'point':'selected'):t<1100?'slider':t<5800?'drag':'best';
  }
  function tick(time) {
    frame=null;if(!dialog.open||document.hidden||motion.matches||!ready)return;
    if(previous!==null)elapsed=Math.min(durations[step],elapsed+Math.min(100,time-previous));
    previous=time;render();
    if(elapsed<durations[step])frame=requestAnimationFrame(tick);
  }
  function play() {
    if(frame!==null)cancelAnimationFrame(frame);frame=null;previous=null;
    if(motion.matches)elapsed=durations[step];
    render();
    if(dialog.open&&ready&&!motion.matches&&!document.hidden&&elapsed<durations[step])frame=requestAnimationFrame(tick);
  }
  function choose(index,focus=false) {
    step=index;elapsed=0;shown=-1;dialog.dataset.step=String(index);
    q('#help-content').setAttribute('aria-labelledby',`help-tab-${index}`);
    q('.help-caption').textContent=captions[index];
    dialog.querySelectorAll('[data-help-step]').forEach((tab,i)=>{tab.setAttribute('aria-selected',String(i===index));tab.tabIndex=i===index?0:-1;});
    if(focus)q(`#help-tab-${index}`).focus();
    play();
  }
  dialog.querySelectorAll('[data-help-step]').forEach((tab,i)=>{
    tab.addEventListener('click',()=>choose(i));
    tab.addEventListener('keydown',e=>{let next;if(e.key==='ArrowRight')next=(i+1)%3;else if(e.key==='ArrowLeft')next=(i+2)%3;else if(e.key==='Home')next=0;else if(e.key==='End')next=2;else return;e.preventDefault();choose(next,true);});
  });
  q('.help-close').addEventListener('click',()=>dialog.close());
  q('.help-try').addEventListener('click',()=>dialog.close());
  q('.help-replay').addEventListener('click',()=>choose(step));
  dialog.addEventListener('click',e=>{const r=dialog.getBoundingClientRect();if(e.target===dialog&&(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom))dialog.close();});
  dialog.addEventListener('close',()=>{if(frame!==null)cancelAnimationFrame(frame);frame=null;opener.focus();});
  document.addEventListener('visibilitychange',()=>{if(dialog.open)play();});
  motion.addEventListener('change',()=>{if(dialog.open)play();});
  new ResizeObserver(()=>{if(dialog.open)render();}).observe(dialog);
  let loaded;
  opener.addEventListener('click',()=>{
    dialog.showModal();choose(0);
    if(!loaded)loaded=Promise.all([det.image,scan.image,...thumbnails.map(c=>c.image),...prompt.outputs.map(o=>o.display)].map(src=>{const im=new Image();im.src=src;return im.decode();}));
    loaded.then(()=>{ready=true;if(dialog.open)play();}).catch(()=>{q('.help-caption').textContent='Images could not load. Close and reopen Help to try again.';loaded=null;});
  });
})();
