function audit(){
  function rng(seed){let s=seed>>>0;return ()=>{s=(1664525*s+1013904223)>>>0;return (s+.5)/4294967296}}
  function normal(r){return Math.sqrt(-2*Math.log(r()))*Math.cos(2*Math.PI*r())}
  const d=12,C=8,n=32,nt=128,betas=[0,.1,.3],scales=[0,.5,1.5],seeds=60;
  const zeros=()=>Array(d).fill(0),dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0),add=(a,b)=>a.map((x,i)=>x+b[i]),sub=(a,b)=>a.map((x,i)=>x-b[i]),mul=(a,k)=>a.map(x=>x*k),mean=a=>mul(a.reduce(add,zeros()),1/a.length);
  const sig=x=>1/(1+Math.exp(-x)),ce=(z,t)=>Math.max(z,0)+Math.log1p(Math.exp(-Math.abs(z)))-t*z;
  function stats(data,v){
    let A=0,L=0;
    for(const row of data){
      const p=row.map(o=>sig(o.p+dot(o.s,v))),q=row.map(o=>sig(o.q+dot(o.qs,v)));
      const m=p.map((x,i)=>[Math.min(x,q[i]),Math.min(1-x,1-q[i])]),a=m.map(x=>x[0]+x[1]);
      A+=a[0]+m[0][1]*a[1]+m[0][0]*a[2];
      L+=row.reduce((s,o)=>s+ce(o.p+dot(o.s,v),o.t),0)/3;
    }
    return [A/data.length,L/data.length]
  }
  function data(r,delta,task,count){
    return Array.from({length:count},()=>Array.from({length:3},()=>{
      const s=Array.from({length:d},()=>normal(r)/Math.sqrt(d)),qs=s.map(x=>.3*x+.8*normal(r)/Math.sqrt(d));
      const q=normal(r),p=q+dot(s,delta),t=sig(p+dot(s,task));return {s,qs,q,p,t}
    }))
  }
  function grad(rows,beta){
    let v=zeros(),h=1e-5;
    for(let j=0;j<d;j++){
      const a=zeros(),b=zeros();a[j]=h;b[j]=-h;
      const x=stats(rows,a),y=stats(rows,b);
      v[j]=(beta*(x[0]-y[0])-(x[1]-y[1]))/(2*h)
    }
    return v
  }
  function directions(gs,groups,kind){
    const g0=mean(gs);
    return gs.map((g,c)=>{
      if(kind==="global")return g0;
      if(kind==="local")return g;
      if(kind==="flat")return add(g0,mul(sub(g,g0),.45));
      const ge=mean(gs.filter((_,i)=>groups[i]===groups[c]));
      return add(add(g0,mul(sub(ge,g0),.8)),mul(sub(g,ge),.25))
    })
  }
  const rows=[],agg={};
  for(const scale of scales)for(let seed=0;seed<seeds;seed++){
    const r=rng(123456+seed*12497+Math.round(scale*1000));
    const common=Array.from({length:d},()=>normal(r)*.4),rdelta=Array.from({length:2},()=>Array.from({length:d},()=>normal(r)*scale)),task=Array.from({length:d},()=>normal(r)*.4);
    const train=[],test=[];
    for(let c=0;c<C;c++){
      const delta=add(add(common,rdelta[c>>2]),Array.from({length:d},()=>normal(r)*.35));
      const t=add(task,Array.from({length:d},()=>normal(r)*.2));
      train.push(data(r,delta,t,n));test.push(data(r,delta,t,nt))
    }
    const base=test.map(x=>stats(x,zeros())).reduce((a,b)=>a.map((x,i)=>x+b[i]/C),[0,0]);
    for(const beta of betas){
      const gs=train.map(t=>grad(t,beta)),g0=mean(gs),gg=[0,0,0,0,1,1,1,1],rr=[0,1,0,1,0,1,0,1];
      const ge=gg.map((g)=>mean(gs.filter((_,i)=>gg[i]===g)));
      const variance=[dot(g0,g0),ge.reduce((s,g)=>s+dot(sub(g,g0),sub(g,g0))/C,0),gs.reduce((s,g,i)=>s+dot(sub(g,ge[i]),sub(g,ge[i]))/C,0)];
      for(const kind of ["global","local","flat","aligned","shuffled"]){
        const vs=directions(gs,kind==="shuffled"?rr:gg,kind);
        const norm=Math.sqrt(vs.reduce((s,v)=>s+dot(v,v)/C,0));const fixed=.25;
        const step=vs.map(v=>mul(v,fixed/Math.max(norm,1e-14)));
        const m=test.map((t,c)=>stats(t,step[c])).reduce((a,b)=>a.map((x,i)=>x+b[i]/C),[0,0]);
        const row={scale,seed,beta,kind,dA:m[0]-base[0],dCE:m[1]-base[1],norm:Math.sqrt(step.reduce((s,v)=>s+dot(v,v)/C,0)),variance};
        rows.push(row);const key=[scale,beta,kind].join("|");(agg[key]??=[]).push(row)
      }
    }
  }
  const summary=Object.fromEntries(Object.entries(agg).map(([k,rs])=>[k,{dA:rs.reduce((s,x)=>s+x.dA/rs.length,0),dCE:rs.reduce((s,x)=>s+x.dCE/rs.length,0),norm:rs.reduce((s,x)=>s+x.norm/rs.length,0),jointWin:rs.filter(x=>x.dA>0&&x.dCE<0).length,edgeFrac:rs.reduce((s,x)=>s+x.variance[1]/x.variance.reduce((a,b)=>a+b,0)/rs.length,0)}]));
  const paired={};for(const s of scales)for(const beta of betas){const a=agg[[s,beta,"aligned"].join("|")],b=agg[[s,beta,"shuffled"].join("|")],c=agg[[s,beta,"flat"].join("|")];paired[[s,beta].join("|")]={vsShuffled:a.filter((x,i)=>x.dA>b[i].dA&&x.dCE<b[i].dCE).length,vsFlat:a.filter((x,i)=>x.dA>c[i].dA&&x.dCE<c[i].dCE).length}}
  return {design:{d,C,n,nt,betas,scales,seeds,rmsStep:.25,engine:"JavaScript V8 float64"},summary,paired,rows}
}
console.log(JSON.stringify(audit(),null,2));
