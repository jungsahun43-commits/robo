// Isolated DOM-event stubs. This is not a browser/rendering test or human annotation.
"use strict";
const test=require("node:test"),assert=require("node:assert/strict"),fs=require("node:fs"),vm=require("node:vm"),path=require("node:path");
const packagePath=process.env.SAFELOG_RELATION_QA_PACKAGE;
function boot(p,store=new Map()){
  const ids=[...fs.readFileSync(path.join(__dirname,"../relation-review-ui/index.html"),"utf8").matchAll(/id="([^"]+)"/g)].map(x=>x[1]);
  class Element{constructor(){this.listeners={};this.value="";this.checked=false;this.files=[];this.textContent="";this.dataset={};this.classList={toggle(){}};}addEventListener(k,f){this.listeners[k]=f;}async fire(k,e={}){return this.listeners[k]?.(e);}getBoundingClientRect(){return {left:10,top:20,width:400,height:200};}getContext(){return new Proxy({},{get:()=>()=>{},set:()=>true});}click(){this.clicked=true;}}
  const elements=new Map(ids.map(x=>[x,new Element()]));const modes=["inspect","group","left","right"].map(x=>{const e=new Element();e.dataset.mode=x;return e;});
  let lastAnchor,exportedBlob;const context={console,structuredClone,Blob,TextDecoder,Date,Math,Number,Map,Set,JSON,Promise,Error,Object,Array,RegExp,Uint8Array,localStorage:{getItem:k=>store.get(k)||null,setItem:(k,v)=>store.set(k,v)},document:{getElementById:id=>{assert.ok(elements.has(id),`Missing HTML element: ${id}`);return elements.get(id);},querySelectorAll:()=>modes,createElement:()=>{lastAnchor=new Element();return lastAnchor;}},window:{REVIEW_PACKAGE:p,confirm:()=>true},URL:{createObjectURL:b=>{exportedBlob=b;return "blob:test";},revokeObjectURL(){}},setTimeout:f=>{f();},Image:class{set src(s){this.source=s;this.onload?.();}},module:{exports:{}}};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname,"../relation-review-ui/review.js"),"utf8"),context);
  return {elements,modes,store,context,get exportedBlob(){return exportedBlob;},get lastAnchor(){return lastAnchor;}};
}
test("actual package handler smoke: required fields, normalized points, persistence, edit demotion, export/import",{skip:!packagePath},async()=>{
  const p=JSON.parse(fs.readFileSync(packagePath,"utf8")),ui=boot(p),e=id=>ui.elements.get(id),change=async(id,value)=>{e(id).value=value;await e(id).fire("input");};
  await e("confirm").fire("click");assert.match(e("message").textContent,/검수자/);
  await change("reviewer","QA_SIMULATED_NOT_HUMAN");await change("note","Synthetic handler test only. Not actual human labels.");
  const s=p.cases[0];for(const side of ["left","right"]){const parts=s.connector_shapes.filter(sh=>s.proposed_groups.find(a=>a.shape_id===sh.shape_id).side===side);await change(side+"_count",String(parts.length));await change(side+"_pin","1");const pts=parts[0].points;const x=pts.reduce((a,b)=>a+b[0],0)/pts.length,y=pts.reduce((a,b)=>a+b[1],0)/pts.length;await ui.modes.find(b=>b.dataset.mode===side).fire("click");await e("photo").fire("click",{clientX:10+x*400,clientY:20+y*200});}
  await change("wire_count",String(s.cable_shapes.length));await change("wire_color","blue");e("polygons").checked=true;await e("polygons").fire("change");await e("confirm").fire("click");assert.match(e("case-status").textContent,/관찰 확인/);
  await e("export").fire("click");assert.equal(ui.lastAnchor.download,"safelog-connector-review.json");assert.ok(ui.lastAnchor.clicked);const text=await ui.exportedBlob.text(),d=JSON.parse(text);assert.equal(d.cases[0].status,"confirmed");assert.equal(d.cases[0].left_endpoint.length,2);assert.equal(d.cases.length,24);
  if(process.env.SAFELOG_RELATION_QA_EXPORT)fs.writeFileSync(process.env.SAFELOG_RELATION_QA_EXPORT,text);
  const restored=boot(p,ui.store);assert.match(restored.elements.get("case-status").textContent,/관찰 확인/);restored.elements.get("left_count").value="9";await restored.elements.get("left_count").fire("input");assert.match(restored.elements.get("case-status").textContent,/초안/);
  const load=restored.elements.get("import");load.files=[{size:Buffer.byteLength(text),arrayBuffer:async()=>Uint8Array.from(Buffer.from(text)).buffer}];await load.fire("change");assert.match(restored.elements.get("case-status").textContent,/관찰 확인/);await restored.elements.get("next").fire("click");assert.equal(restored.elements.get("progress").textContent,"2 / 24");
  const bad=JSON.parse(text);bad.cases[0].image_sha256="wrong";load.files=[{size:100,arrayBuffer:async()=>Uint8Array.from(Buffer.from(JSON.stringify(bad))).buffer}];await load.fire("change");assert.match(restored.elements.get("message").textContent,/원본/);assert.equal(restored.elements.get("progress").textContent,"2 / 24");
});
