import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';

const data=window.DATA,host=document.getElementById('cloud'),status=document.getElementById('cloudStatus');
const pick=document.getElementById('cloudChoice'),quality=document.getElementById('cloudQuality');
const start=document.getElementById('cloudLoad'),stop=document.getElementById('cloudStop'),download=document.getElementById('cloudDownload');
const scene=new THREE.Scene();scene.background=new THREE.Color('#edf1f4');
const camera=new THREE.PerspectiveCamera(50,1,.001,3000);camera.up.set(0,0,1);
const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));host.append(renderer.domElement);
const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;
let group=new THREE.Group();scene.add(group);let material=null,abort=null,generation=0,loaded=0,index=0,descriptor=null,dirty=true,busy=false,framed=false;
controls.addEventListener('change',()=>dirty=true);

function resize(){if(host.clientWidth){renderer.setSize(host.clientWidth,host.clientHeight);camera.aspect=host.clientWidth/host.clientHeight;camera.updateProjectionMatrix();dirty=true}}
new ResizeObserver(resize).observe(host);window.addEventListener('resize',resize);resize();
function frame(){
  if(!descriptor)return;
  let box=new THREE.Box3();
  if(descriptor.bounds_min&&descriptor.bounds_max)box.set(new THREE.Vector3(...descriptor.bounds_min),new THREE.Vector3(...descriptor.bounds_max));
  else box.setFromObject(group);
  if(box.isEmpty())return;
  const sphere=box.getBoundingSphere(new THREE.Sphere()),r=Math.max(sphere.radius,.1);
  const vertical=THREE.MathUtils.degToRad(camera.fov),horizontal=2*Math.atan(Math.tan(vertical/2)*camera.aspect);
  const distance=r/Math.sin(Math.min(vertical,horizontal)/2)*1.08;
  controls.target.copy(sphere.center);camera.position.copy(sphere.center).add(new THREE.Vector3(.9,-1.2,.8).normalize().multiplyScalar(distance));
  camera.near=Math.max(.001,r/100000);camera.far=Math.max(3000,r*200);camera.updateProjectionMatrix();controls.update();dirty=true;framed=true;
}
document.getElementById('cloudFit').onclick=frame;
function clear(){
  abort?.abort();generation++;busy=false;loaded=0;index=0;framed=false;
  for(const p of group.children)p.geometry.dispose();group.clear();material?.dispose();material=null;dirty=true;
}
function choose(){
  clear();const cloud=data.clouds.find(c=>c.key===pick.value);const full=quality.value==='full';
  quality.querySelector('option[value="quick"]').disabled=!cloud.file;
  if(!cloud.file&&!full)quality.value='full';
  if(quality.value==='full'&&cloud.full){descriptor={...cloud.full,isFull:true};}
  else if(cloud.file){
    descriptor={points:cloud.display_points,chunks:[{file:cloud.file,points:cloud.display_points}],layout:'xyzrgb_f32',
      isFull:cloud.display_points===cloud.preview_file_points,
      note:quality.value==='full'?'该历史项只提供其预览文件；当前显示该文件已有点数。':'快速预览最多12万点，保留原有预览处理。'};
  }else{descriptor=null;status.textContent='这份全点数据尚未导出。';return;}
  document.getElementById('cloudNote').textContent=descriptor.note||'';
  download.hidden=!descriptor.download;if(descriptor.download){download.href=new URL(descriptor.download.replace(/^\/viewer\//,''),location.href).href;download.textContent='下载对应全点PCD / 原点云文件';}
  const size=Number(document.getElementById('pointSize').value);
  if(descriptor.layout==='xyz_f32'){
    const low=descriptor.bounds_min?.[2]??0,high=descriptor.bounds_max?.[2]??10;
    material=new THREE.ShaderMaterial({uniforms:{pointSize:{value:size},low:{value:low},span:{value:Math.max(high-low,.001)}},
      vertexShader:'uniform float pointSize; uniform float low; uniform float span; varying float h; void main(){ h=clamp((position.z-low)/span,0.0,1.0); gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0); gl_PointSize=pointSize; }',
      fragmentShader:'varying float h; void main(){vec3 a=vec3(.22,.16,.45),b=vec3(.1,.65,.62),c=vec3(.96,.9,.18); gl_FragColor=vec4(h<.5?mix(a,b,h*2.0):mix(b,c,(h-.5)*2.0),1.0);}'
    });
  }else material=new THREE.PointsMaterial({size,sizeAttenuation:false,vertexColors:true});
  start.hidden=false;stop.hidden=false;stop.disabled=true;start.disabled=false;start.textContent='加载全部点';
  if(descriptor.points>15000000){
    const bytes=descriptor.chunks.reduce((s,c)=>s+(c.bytes||0),0);
    status.textContent=`全部 ${descriptor.points.toLocaleString()} 点，约 ${(bytes/1024**3).toFixed(2)} GiB。点击“加载全部点”分块加载，可随时停止；已加载的每块保留所有点。`;
  }else loadRemaining();
}
async function addChunk(chunk,token){
  if(token!==generation)return false;
  const response=await fetch(chunk.file,{signal:abort.signal});if(!response.ok)throw Error(`点云读取失败 HTTP ${response.status}`);
  const buffer=await response.arrayBuffer();if(token!==generation)return false;
  const raw=new Float32Array(buffer),channels=descriptor.layout==='xyz_f32'?3:6,n=raw.length/channels;
  if(!Number.isInteger(n)||n!==chunk.points)throw Error('分块点数与清单不一致，已停止加载');
  let positions,colors;
  if(channels===3)positions=raw;
  else{
    positions=new Float32Array(n*3);colors=new Float32Array(n*3);
    for(let i=0;i<n;i++){positions.set(raw.subarray(i*6,i*6+3),i*3);colors.set(raw.subarray(i*6+3,i*6+6),i*3);}
  }
  const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(positions,3));
  if(colors)geometry.setAttribute('color',new THREE.BufferAttribute(colors,3));geometry.computeBoundingSphere();
  group.add(new THREE.Points(geometry,material));loaded+=n;dirty=true;
  if(!framed)frame();return true;
}
async function loadRemaining(){
  if(busy||!descriptor)return;busy=true;const token=generation;abort=new AbortController();start.disabled=true;stop.disabled=false;
  try{
    while(token===generation&&index<descriptor.chunks.length){
      if(!await addChunk(descriptor.chunks[index],token))return;
      index++;
      status.textContent=`${descriptor.isFull?'原文件全部点模式':'快速预览'} · 已显示 ${loaded.toLocaleString()} / ${descriptor.points.toLocaleString()} 点 · 分块 ${index}/${descriptor.chunks.length}`;
      await new Promise(requestAnimationFrame);
    }
    if(token===generation){start.textContent='已加载全部';start.disabled=true;stop.disabled=true;}
  }catch(error){
    if(token!==generation)return;
    status.textContent=error.name==='AbortError'?`已停止，保留 ${loaded.toLocaleString()} 点。点击继续加载剩余原点。`:error.message;
    start.disabled=false;start.textContent='继续加载';stop.disabled=true;
  }finally{if(token===generation)busy=false;}
}
start.onclick=loadRemaining;stop.onclick=()=>abort?.abort();
data.clouds.forEach(cloud=>{const option=document.createElement('option');option.value=cloud.key;option.textContent=cloud.title;pick.append(option)});
const requestedCloud=new URLSearchParams(location.search).get('cloud');
pick.value=data.clouds.some(c=>c.key===requestedCloud)?requestedCloud:'lidar';
quality.value='full';pick.onchange=choose;quality.onchange=choose;
document.getElementById('pointSize').oninput=e=>{if(material?.uniforms)material.uniforms.pointSize.value=Number(e.target.value);else if(material)material.size=Number(e.target.value);dirty=true;};
let previous=0;
function animate(now){requestAnimationFrame(animate);controls.update();if(dirty&&(loaded<15000000||now-previous>100)){renderer.render(scene,camera);dirty=false;previous=now}}
renderer.domElement.addEventListener('webglcontextlost',e=>{e.preventDefault();abort?.abort();status.textContent='浏览器图形内存不足。请重新打开页面并选择快速预览或较小点云。';});
requestAnimationFrame(animate);choose();
