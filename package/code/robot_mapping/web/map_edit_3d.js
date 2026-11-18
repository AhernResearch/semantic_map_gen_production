import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';

const el=id=>document.getElementById(id),host=el('view3d');
let renderer;
try{renderer=new THREE.WebGLRenderer({antialias:true})}catch(error){el('view3dStatus').textContent='此浏览器无法创建三维视图：'+error.message;throw error}
renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));host.append(renderer.domElement);
const scene=new THREE.Scene();scene.background=new THREE.Color('#edf2ef');
const camera=new THREE.PerspectiveCamera(55,1,.01,10000);camera.position.set(8,6,8);
const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.dampingFactor=.12;
const regions=new THREE.Group();scene.add(regions);scene.add(new THREE.AxesHelper(3));
const grid=new THREE.GridHelper(100,100,0x8fa79d,0xd4e0db);scene.add(grid);
let points=null,currentCloud=null,state=null,roll=0,indexSignature='';
const colors={display_region:0x229678,exclude_points:0xcc4b45,dynamic_obstacle_region:0xdf7032,propose_free:0x518ce0};
function fit(useTrajectory=false){
 if(!state?.cloud)return;
 let bounds=new THREE.Box3().setFromBufferAttribute(points.geometry.getAttribute('position'));
 if(useTrajectory&&state.trajectory?.length){
  const b=new THREE.Box3();for(let i=0;i<state.trajectory.length;i+=2)b.expandByPoint(new THREE.Vector3(state.trajectory[i],0,state.trajectory[i+1]));
  b.expandByScalar(3);bounds=b;
 }
 const center=bounds.getCenter(new THREE.Vector3()),size=bounds.getSize(new THREE.Vector3()),distance=Math.max(size.x,size.y,size.z,2)*1.15;
 controls.target.copy(center);camera.position.copy(center).add(new THREE.Vector3(.55,.65,.8).normalize().multiplyScalar(distance));
 camera.near=Math.max(.005,distance/10000);camera.far=Math.max(1000,distance*20);camera.updateProjectionMatrix();roll=0;el('roll3d').value='0';controls.update();
}
function volume(polygon,yRange,color){
 if(polygon.length<3||!yRange.every(Number.isFinite))return;
 const lines=[],triangles=[];
 for(let i=0;i<polygon.length;i++){
  const a=polygon[i],b=polygon[(i+1)%polygon.length],lo=yRange[0],hi=yRange[1];
  lines.push(a[0],lo,a[1],b[0],lo,b[1],a[0],hi,a[1],b[0],hi,b[1],a[0],lo,a[1],a[0],hi,a[1]);
  triangles.push(a[0],lo,a[1],b[0],lo,b[1],b[0],hi,b[1],a[0],lo,a[1],b[0],hi,b[1],a[0],hi,a[1]);
 }
 const geom=new THREE.BufferGeometry();geom.setAttribute('position',new THREE.Float32BufferAttribute(lines,3));
 regions.add(new THREE.LineSegments(geom,new THREE.LineBasicMaterial({color})));
 const face=new THREE.BufferGeometry();face.setAttribute('position',new THREE.Float32BufferAttribute(triangles,3));
 regions.add(new THREE.Mesh(face,new THREE.MeshBasicMaterial({color,transparent:true,opacity:.13,side:THREE.DoubleSide,depthWrite:false})));
}
function inside(x,z,polygon){let result=false;for(let i=0,j=polygon.length-1;i<polygon.length;j=i++){
 const a=polygon[j],b=polygon[i];if((a[1]>z)!==(b[1]>z)&&x<(b[0]-a[0])*(z-a[1])/(b[1]-a[1])+a[0])result=!result;
}return result}
function sync(next){
 state=next;if(!state?.cloud)return;
 if(currentCloud!==state.cloud){
  if(points){scene.remove(points);points.geometry.dispose();points.material.dispose()}
  currentCloud=state.cloud;const geometry=new THREE.BufferGeometry();
  geometry.setAttribute('position',new THREE.BufferAttribute(currentCloud.xyz,3));
  geometry.setAttribute('color',new THREE.Uint8BufferAttribute(currentCloud.rgb,3,true));
  points=new THREE.Points(geometry,new THREE.PointsMaterial({size:.035,vertexColors:true,sizeAttenuation:true}));scene.add(points);
  indexSignature='';fit(true);
 }
 const signature=JSON.stringify([state.limit,state.hide,state.catalog?.head_event_id]);
 if(signature!==indexSignature){
  const xyz=currentCloud.xyz,indices=[];for(let i=0;i<xyz.length;i+=3){
   const x=xyz[i],y=xyz[i+1],z=xyz[i+2];if(y>state.limit)continue;
   if(state.hide&&(state.catalog?.regions||[]).some(r=>r.kind==='exclude_points'&&y>=r.y_range_m[0]&&y<=r.y_range_m[1]&&inside(x,z,r.polygon_xz)))continue;
   indices.push(i/3);
  }
  points.geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(indices),1));indexSignature=signature;
  el('view3dStatus').textContent='三维显示 '+indices.length.toLocaleString()+' / '+(xyz.length/3).toLocaleString()+' 个底图点；圈选体积按世界X/Z及Y高度范围显示。';
 }
 while(regions.children.length){const object=regions.children[0];regions.remove(object);object.geometry.dispose();object.material.dispose()}
 for(const r of state.catalog?.regions||[])volume(r.polygon_xz,r.y_range_m,colors[r.kind]);
 volume(state.polygon||[],state.yRange||[],0x203a41);
}
function setAngles(){
 const yaw=THREE.MathUtils.degToRad(Number(el('yaw3d').value)),pitch=THREE.MathUtils.degToRad(Math.max(-89.8,Math.min(89.8,Number(el('pitch3d').value))));
 const r=Math.max(.1,camera.position.distanceTo(controls.target));camera.position.copy(controls.target).add(new THREE.Vector3(Math.sin(yaw)*Math.cos(pitch),Math.sin(pitch),Math.cos(yaw)*Math.cos(pitch)).multiplyScalar(r));controls.update();
}
el('yaw3d').onchange=el('pitch3d').onchange=setAngles;
el('roll3d').oninput=()=>roll=THREE.MathUtils.degToRad(Number(el('roll3d').value));
el('fit3d').onclick=()=>fit(false);el('capture3d').onclick=()=>fit(true);
el('top3d').onclick=()=>{el('pitch3d').value='89.8';setAngles()};
el('reset3d').onclick=()=>fit(true);
controls.addEventListener('change',()=>{const delta=camera.position.clone().sub(controls.target);el('yaw3d').value=THREE.MathUtils.radToDeg(Math.atan2(delta.x,delta.z)).toFixed(1);el('pitch3d').value=THREE.MathUtils.radToDeg(Math.asin(delta.y/Math.max(delta.length(),1e-8))).toFixed(1)});
new ResizeObserver(()=>{const w=host.clientWidth,h=host.clientHeight;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix()}).observe(host);
window.addEventListener('mapedit-view',event=>sync(event.detail));
sync(window.mapEditViewState);
function animate(){requestAnimationFrame(animate);controls.update();const original=camera.quaternion.clone();camera.rotateZ(roll);renderer.render(scene,camera);camera.quaternion.copy(original)}animate();
