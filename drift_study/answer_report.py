"""Offline HTML for prefix and answer hidden-state drift; no model loading."""
import json
from pathlib import Path

GROUPS=('acoustic','semantic','text','answer_all','answer_masked','answer_decoded','answer_changed')
MAP=dict(acoustic='acoustic',semantic='semantic',text='text',
         answer_content_all='answer_all',answer_content_masked_stable='answer_masked',
         answer_content_decoded_stable='answer_decoded',answer_content_changed='answer_changed')


def collect(out):
    out=Path(out);examples=[];found=False
    for result in sorted(out.glob('samples/*/result.json')):
        r=json.loads(result.read_text());frames={}
        answer=result.parent/'answer_drift.jsonl';found|=answer.exists()
        for path in [result.parent/'drift.jsonl',answer]:
            if not path.exists():continue
            with path.open() as stream:
                for line in stream:
                    row=json.loads(line)
                    if not row['step'] or row['group'] not in MAP:continue
                    key=(row['reference'],row['step']);frame=frames.setdefault(key,
                        dict(step=row['step'],ref=row['reference'],v=[[None]*len(GROUPS) for _ in range(32)],
                             counts=[[0]*len(GROUPS) for _ in range(32)]))
                    g=GROUPS.index(MAP[row['group']]);layer=row['layer']
                    if frame['v'][layer][g] is not None:raise ValueError('Duplicate drift observation')
                    frame['v'][layer][g]=row['metrics']['mean'];frame['counts'][layer][g]=row['count']
        examples.append(dict(id=r['id'],cohort=r.get('domain','audio'),n=r['forward_count'],
            activeEnd=r['forward_count']-1,answer_available=answer.exists(),
            scope='MMAU: final visible answer span, excluding EOS and later slots; includes reasoning',
            frames=[frames[k] for k in sorted(frames)]))
    return dict(examples=examples,groups=GROUPS) if found else None


def write_html(data,target):
    html=TEMPLATE.replace('__DATA__',json.dumps(data,separators=(',',':'),allow_nan=False).replace('<','\\u003c'))
    Path(target).parent.mkdir(parents=True,exist_ok=True);Path(target).write_text(html)
    return Path(target)


def build(out):
    data=collect(out)
    return write_html(data,Path(out)/'analysis'/'answer_drift_by_forward.html') if data else None


TEMPLATE=r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Answer and prefix representation drift</title><style>
*{box-sizing:border-box}body{font:16px system-ui,sans-serif;color:#203047;background:#f5f7fb;max-width:1450px;margin:auto;padding:24px}h1{font-size:26px;margin:0 0 10px}p{line-height:1.5}.controls,main,.notes{background:white;border:1px solid #dbe2ec;border-radius:12px;padding:18px;margin:16px 0}.controls,#groups{display:flex;gap:16px;flex-wrap:wrap;align-items:center}label{display:inline-flex;align-items:center;gap:7px}select,button{font:inherit;border:1px solid #b9c5d7;border-radius:6px;padding:7px;max-width:280px}button{background:#edf3fe;cursor:pointer}canvas{width:100%;height:550px;display:block}#hover{min-height:42px;font:14px ui-monospace,monospace;white-space:pre-wrap}.note{font-size:14px;color:#52627a}.warning{background:#fff4dd;padding:10px;border-radius:6px}[hidden]{display:none!important}@media(max-width:650px){body{padding:12px}h1{font-size:23px}select{max-width:250px}main{padding:10px}}</style></head><body>
<h1>Answer and prefix internal-representation drift</h1>
<p>Post-transformer-block hidden-state change at the same token positions. <b>This is hidden-state drift, not attention or output probabilities.</b></p>
<div class="controls"><label>Cohort <select id="cohort"></select></label><label>Example <select id="example"></select></label><label>Layers <select id="layers"><option value="0,31">All32</option><option value="0,7">0–7</option><option value="8,15">8–15</option><option value="16,23">16–23</option><option value="24,31" selected>24–31</option></select></label><label>Reference <select id="reference"><option value="previous">Previous forward</option><option value="initial">Initial state</option><option value="block_start">Current block start (GSM)</option></select></label><label>Window <select id="window"><option value="1">Each forward</option><option value="5">Five-forward averages</option></select></label><label><input id="active" type="checkbox" checked> Stop at answer-content completion</label><label><input id="matched" type="checkbox"> Fixed example cohort per curve</label><button id="download">Export CSV</button></div>
<p id="availability" class="note warning"></p><main><div id="groups"></div><p id="summary"></p><canvas id="plot"></canvas><div id="hover">Move over a point for drift, contributing examples and token counts.</div></main>
<section class="notes"><p id="scope" class="note"></p><p class="note">Stable masked/decoded groups include only positions whose input token ID is identical in current and reference forwards. They help separate contextual hidden-state movement from the effect of committing a new token. “Changed answer IDs” deliberately includes token replacement. “All answer positions” combines changed and unchanged positions and should not be read as pure contextual drift.</p>
<p class="note">Population sizes change as decoding progresses. Empty populations are omitted, never filled with zero. Each curve averages selected layers within an example, then forwards within a window, then examples equally. Each curve retains its own contributor counts. The lower panel shows the minimum/maximum example count over visible curves. Initial stable-decoded comparisons can be absent because the initial state contains only MASK tokens.</p>
<p class="note">The source is fixed512-slot, two-commits-per-forward GSM decoding or128-slot confidence-adaptive MMAU FPD. Cohorts are descriptive and not a controlled decoder comparison. GSM groups span the recorded answer canvas; new MMAU recordings restrict to the final visible span. Attention files do not contain answer hidden vectors and cannot supply missing answer drift.</p></section>
<script id="data" type="application/json">__DATA__</script><script>
'use strict';const DATA=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id),keys=DATA.groups;
const labels=['Acoustic audio','Semantic audio','Fixed text','All answer positions','Stable MASK answer','Stable decoded answer','Changed answer IDs'];const colors=['#1976d2','#e47813','#16805c','#777','#b58900','#8649c5','#d23f58'];
const cohorts=[...new Set(DATA.examples.map(e=>e.cohort))];cohorts.forEach(c=>$('cohort').add(new Option(c,c)));for(let l=0;l<32;l++)$('layers').add(new Option('Layer '+l,l+','+l));
labels.forEach((label,g)=>{const el=document.createElement('label');el.style.color=colors[g];const input=document.createElement('input');input.type='checkbox';input.checked=g<3||g===4||g===5;input.value=g;input.onchange=update;el.append(input,document.createTextNode(label));$('groups').append(el)});
let state={points:[]},active=[],geom={},hoverIndex=null;
function examples(){return DATA.examples.filter((e,i)=>e.cohort===$('cohort').value&&($('example').value==='all'||i===+$('example').value))}
function populate(){$('example').replaceChildren(new Option('Equal-weight average','all'));DATA.examples.forEach((e,i)=>{if(e.cohort===$('cohort').value)$('example').add(new Option(e.id,i))});return update()}
function aggregate(){const selected=examples(),[lo,hi]=$('layers').value.split(',').map(Number),width=+$('window').value,reference=$('reference').value,pool=new Map();
 for(const e of selected){const local=new Map();for(const frame of e.frames){if(frame.ref!==reference||!frame.step||($('active').checked&&frame.step>e.activeEnd))continue;const bin=Math.floor(frame.step/width);let p=local.get(bin);if(!p){p={s:Array(7).fill(0),n:Array(7).fill(0),tokens:Array(7).fill(0),first:frame.step,last:frame.step};local.set(bin,p)}p.first=Math.min(p.first,frame.step);p.last=Math.max(p.last,frame.step);
 for(let g=0;g<7;g++){let sum=0,n=0,tokens=0;for(let l=lo;l<=hi;l++){const v=frame.v[l][g];if(v!==null){sum+=v;tokens+=frame.counts[l][g];n++}}if(n){p.s[g]+=sum/n;p.tokens[g]+=tokens/n;p.n[g]++}}}
 for(const [bin,p] of local){if(!pool.has(bin))pool.set(bin,{bin,x:width===1?bin:bin*width+(width-1)/2,v:Array(7).fill(0),n:Array(7).fill(0),tokens:Array(7).fill(0),first:p.first,last:p.last});const q=pool.get(bin);q.first=Math.min(q.first,p.first);q.last=Math.max(q.last,p.last);for(let g=0;g<7;g++)if(p.n[g]){q.v[g]+=p.s[g]/p.n[g];q.tokens[g]+=p.tokens[g]/p.n[g];q.n[g]++}}}
 const points=[...pool.values()].sort((a,b)=>a.bin-b.bin);for(const p of points)for(let g=0;g<7;g++){const valid=p.n[g]&&(!$('matched').checked||p.n[g]===selected.length);p.v[g]=valid?p.v[g]/p.n[g]:null;p.tokens[g]=valid?p.tokens[g]/p.n[g]:null}
 return {points,selected:selected.length,lo,hi,width,reference};}
function update(){state=aggregate();active=[...$('groups').querySelectorAll('input:checked')].map(e=>+e.value);const selected=examples(),have=selected.filter(e=>e.answer_available).length;
 $('availability').textContent=`${have}/${selected.length} selected examples have recorded answer hidden-state drift. `+(have===selected.length?'Stable-token controls are available.':'Missing answer curves cannot be reconstructed from attention or prefix-only summaries.');$('scope').textContent=[...new Set(selected.map(e=>e.scope))].join(' | ');
 const counts=state.points.flatMap(p=>active.filter(g=>p.v[g]!==null).map(g=>p.n[g]));$('summary').textContent=`${state.selected} selected examples · layers${state.lo}–${state.hi} · ${state.reference} reference · ${state.width===1?'per forward':'five-forward means'} · ${counts.length?Math.min(...counts)+'–'+Math.max(...counts):0} contributing examples per visible curve/point`;
 hoverIndex=null;draw();return state;}
function draw(){const canvas=$('plot'),w=canvas.clientWidth,h=550,dpr=devicePixelRatio||1;canvas.width=w*dpr;canvas.height=h*dpr;const c=canvas.getContext('2d');c.scale(dpr,dpr);const m={l:76,r:22,t:20},pw=w-m.l-m.r,ph=350,by=415,bh=65;
 const points=state.points,xmax=Math.max(1,...points.map(p=>p.x)),values=points.flatMap(p=>active.map(g=>p.v[g]).filter(v=>v!==null));let ymax=Math.max(1e-6,...values)*1.1;const unit=10**Math.floor(Math.log10(ymax/5));ymax=Math.ceil(ymax/5/unit)*unit*5;const x=v=>m.l+v/xmax*pw,y=v=>m.t+ph-v/ymax*ph;geom={m,pw,xmax,x,y};c.font='13px system-ui';c.lineWidth=1;
 for(let i=0;i<=5;i++){const v=ymax*i/5;c.strokeStyle='#e3e9f1';c.beginPath();c.moveTo(m.l,y(v));c.lineTo(w-m.r,y(v));c.stroke();c.fillStyle='#52627a';c.textAlign='right';c.fillText(Number(v.toPrecision(3)),m.l-10,y(v)+4)}for(let i=0;i<=8;i++){const v=Math.round(xmax*i/8);c.textAlign='center';c.fillStyle='#52627a';c.fillText(v,x(v),by+bh+23)}c.fillText('Actual forward index',m.l+pw/2,h-7);c.save();c.translate(17,m.t+ph/2);c.rotate(-Math.PI/2);c.fillText('Mean per-token cosine distance',0,0);c.restore();
 const width=Math.max(1,pw/(xmax/state.width+1)*.7);for(const p of points){const ns=active.filter(g=>p.v[g]!==null).map(g=>p.n[g]);if(!ns.length)continue;const low=Math.min(...ns),high=Math.max(...ns);c.fillStyle='#d6e0ec';c.fillRect(x(p.x)-width/2,by+bh-high/Math.max(1,state.selected)*bh,width,high/Math.max(1,state.selected)*bh);c.fillStyle='#8faaca';c.fillRect(x(p.x)-width/2,by+bh-low/Math.max(1,state.selected)*bh,width,low/Math.max(1,state.selected)*bh)}c.fillStyle='#52627a';c.textAlign='right';c.fillText(state.selected,m.l-10,by+10);c.fillText('0',m.l-10,by+bh);
 for(const g of active){c.strokeStyle=colors[g];c.lineWidth=2;c.beginPath();let prev=null;for(const p of points){if(p.v[g]===null){prev=null;continue}if(prev===null||p.bin!==prev+1)c.moveTo(x(p.x),y(p.v[g]));else c.lineTo(x(p.x),y(p.v[g]));prev=p.bin}c.stroke()}
 if(hoverIndex!==null&&points[hoverIndex]){const p=points[hoverIndex];c.strokeStyle='#8291a7';c.setLineDash([4,4]);c.beginPath();c.moveTo(x(p.x),m.t);c.lineTo(x(p.x),by+bh);c.stroke();c.setLineDash([])}}
$('plot').onmousemove=e=>{if(!state.points.length)return;const rect=$('plot').getBoundingClientRect(),pos=(e.clientX-rect.left-geom.m.l)/geom.pw*geom.xmax;hoverIndex=state.points.reduce((best,p,i)=>Math.abs(p.x-pos)<Math.abs(state.points[best].x-pos)?i:best,0);const p=state.points[hoverIndex];$('hover').textContent=`Forwards ${p.first}–${p.last} | `+active.map(g=>p.v[g]===null?labels[g]+': absent':`${labels[g]}: ${p.v[g].toFixed(6)} (N=${p.n[g]}, mean tokens=${p.tokens[g].toFixed(1)})`).join(' | ');draw()};$('plot').onmouseleave=()=>{hoverIndex=null;draw()};window.onresize=draw;
$('cohort').onchange=populate;for(const id of ['example','layers','reference','window','active','matched'])$(id).onchange=update;
function csvText(){return ['reference,first_forward,last_forward,group,mean_cosine_distance,examples,mean_tokens',...state.points.flatMap(p=>keys.map((g,i)=>[state.reference,p.first,p.last,g,p.v[i]??'',p.n[i],p.tokens[i]??''].join(',')))].join('\n')+'\n'}$('download').onclick=()=>{const url=URL.createObjectURL(new Blob([csvText()],{type:'text/csv'}));const a=document.createElement('a');a.href=url;a.download='answer_hidden_drift.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
window.ANSWER_DRIFT={data:DATA,aggregate,update,csvText,get state(){return state}};populate();
</script></body></html>'''


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    result=build(parser.parse_args().out)
    if result is None:
        parser.exit(1,'No completed sample has recorded answer hidden-state drift.\n')
    print(result)
