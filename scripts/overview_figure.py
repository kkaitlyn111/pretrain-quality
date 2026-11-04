"""Draw the project overview figure as SVG (figures/overview.svg and a matching .html).

Render to PNG at 2x with headless Chrome, for example:
  chrome --headless=new --force-device-scale-factor=2 --window-size=1400,372 \
         --screenshot=figures/overview.png figures/overview.html
"""
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "figures"
W,H=1400,372
F='font-family="Helvetica Neue, Helvetica, Arial, sans-serif"'
G=dict(fill="#eaf2e6",stroke="#86ad74",title="#3f6b2f",sub="#5d8a4c")      # filters lane (green)
M=dict(fill="#f6e6ec",stroke="#b8687f",title="#7d3049",sub="#a24d67")      # swarm lane (maroon)
N=dict(fill="#f3f4f7",stroke="#a7afbd",title="#2b3442",sub="#5b6575")      # neutral
o=[]; a=o.append
a(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
a('<defs>'+''.join(f'<marker id="m{k}" markerWidth="9" markerHeight="9" refX="7.5" refY="4.5" orient="auto"><path d="M0,0.5 L8,4.5 L0,8.5 z" fill="{c}"/></marker>' for k,c in [("g",G["stroke"]),("m",M["stroke"]),("n","#8a93a3")])+'</defs>')
a(f'<rect width="{W}" height="{H}" fill="white"/>')
def box(x,y,w,h,st,title,sub=None,sub2=None):
    a(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{st["fill"]}" stroke="{st["stroke"]}" stroke-width="1.6"/>')
    lines=[(title,17,600,st["title"])]+([(sub,13.5,400,st["sub"])] if sub else [])+([(sub2,13.5,400,st["sub"])] if sub2 else [])
    total=sum(l[1]*1.45 for l in lines)
    yy=y+h/2-total/2+lines[0][1]
    for t,fs,fw,col in lines:
        a(f'<text x="{x+w/2}" y="{yy:.1f}" {F} font-size="{fs}" font-weight="{fw}" fill="{col}" text-anchor="middle">{t}</text>')
        yy+=fs*1.45
def arrow(pts,k,dash=False):
    d="M"+" L".join(f"{x},{y}" for x,y in pts)
    a(f'<path d="{d}" fill="none" stroke="{dict(g=G,m=M).get(k,{"stroke":"#8a93a3"})["stroke"]}" stroke-width="1.8" marker-end="url(#m{k})"{" stroke-dasharray=\"5 4\"" if dash else ""}/>')
def note(x,y,t,st,anchor="middle",size=13.5):
    a(f'<text x="{x}" y="{y}" {F} font-size="{size}" font-style="italic" fill="{st["sub"]}" text-anchor="{anchor}">{t}</text>')

yA,yB,bh=66,244,80          # lane box tops, box height
cA,cB=yA+bh/2,yB+bh/2
X=[250,546,842,1138]; BW=236
def lane_label(y,num,title,sub,st,sub_x):
    a(f'<text x="250" y="{y-22}" {F} font-size="13" font-weight="700" letter-spacing="1.2" fill="{st["sub"]}">STAGE {num}</text>')
    a(f'<text x="322" y="{y-22}" {F} font-size="20" font-weight="700" fill="{st["title"]}">{title}</text>')
    a(f'<text x="{sub_x}" y="{y-22}" {F} font-size="14.5" fill="{st["sub"]}">{sub}</text>')
lane_label(yA,1,"Filters as the target","why do successful filters rate documents highly?",G,536)
lane_label(yB,2,"A model swarm as the target","which documents actually lower validation loss?",M,628)
box(28,(cA+cB)/2-48,172,96,N,"web documents","2.26M from","Corpus-200B")
sx=200; sy=(cA+cB)/2
arrow([(sx,sy-14),(222,sy-14),(222,cA),(246,cA)],"g")
arrow([(sx,sy+14),(222,sy+14),(222,cB),(246,cB)],"m")
# stage 1
box(X[0],yA,BW,bh,G,"10 filter scores","DCLM, FineWeb-Edu,","QuRater, PC")
arrow([(X[0]+BW,cA),(X[1]-4,cA)],"g")
box(X[1],yA,BW,bh,G,"two-factor latent model","fit by EM, with text","features as covariates")
arrow([(X[1]+BW,cA),(X[2]-4,cA)],"g")
box(X[2],yA,BW,bh,G,"two filter axes","benchmark vs.","generative quality")
note(X[2]+BW/2,yA+bh+24,"top-20% quality 0.680 vs. 0.405 for DCLM",G)
# stage 2
box(X[0],yB,BW,bh,M,"593 small LMs","random subsets,","150M tokens each")
arrow([(X[0]+BW,cB),(X[1]-4,cB)],"m")
note((X[0]+BW+X[1])/2,cB-10,"val losses",M,size=12.5)
box(X[1],yB,BW,bh,M,"target y per document","difference in means")
arrow([(X[1]+BW,cB),(X[2]-4,cB)],"m")
box(X[2],yB,BW,bh,M,"interpretable EBM","recreates y from","readable features")
note(X[2]+BW/2,yB+bh+24,"test R\u00b2 0.656 (Lasso 0.408, topic 0.076)",M)
arrow([(X[2]+BW,cB),(X[3]-4,cB)],"m")
box(X[3],yB,BW,bh,M,"held-out data selection","60M LMs on 150M tokens","from 30 fresh shards")
note(X[3]+BW/2,yB+bh+24,"within-topic EBM beats random",M)
a('</svg>')
s="\n".join(o)
open(OUT / "overview.svg","w").write(s)
open(OUT / "overview.html","w").write('<html><body style="margin:0;background:white">'+s+'</body></html>')
print(f"wrote {OUT / 'overview.svg'}")
