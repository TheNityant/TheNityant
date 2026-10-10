#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path

THEMES={
"dark":{"grid":"#30363d","label":"#c9d1d9","title":"#f0f6fc","fill":"#ffa500","stroke":"#ffa500","vertex":"#f0f6fc"},
"light":{"grid":"#d0d7de","label":"#1f2328","title":"#1f2328","fill":"#c46a00","stroke":"#c46a00","vertex":"#24292f"},
}

def ring(r,n):
    return [(r*math.cos(-math.pi/2+i*2*math.pi/n),r*math.sin(-math.pi/2+i*2*math.pi/n)) for i in range(n)]

def render(data,theme):
    t=THEMES[theme]; axes=data["axes"]; n=len(axes)
    W,H=460,360; cx,cy=230,190; r=112
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
    out.append(f'<text x="{cx}" y="28" text-anchor="middle" fill="{t["title"]}" font-family="ui-sans-serif,Segoe UI,Arial" font-size="16" font-weight="700">{data["title"]}</text>')
    for lvl in range(1,6):
        pts=ring(r*lvl/5,n)
        ps=" ".join(f"{cx+x:.1f},{cy+y:.1f}" for x,y in pts)
        out.append(f'<polygon points="{ps}" fill="none" stroke="{t["grid"]}" stroke-width="1"/>')
    for i,a in enumerate(axes):
        ang=-math.pi/2+i*2*math.pi/n
        x2=cx+r*math.cos(ang); y2=cy+r*math.sin(ang)
        lx=cx+(r+30)*math.cos(ang); ly=cy+(r+30)*math.sin(ang)+4
        out.append(f'<line x1="{cx}" y1="{cy}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{t["grid"]}"/>')
        out.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" fill="{t["label"]}" font-family="ui-sans-serif,Segoe UI,Arial" font-size="12">{a["label"]}</text>')
    pts=[]
    for i,a in enumerate(axes):
        v=max(0,min(100,float(a["value"])))
        ang=-math.pi/2+i*2*math.pi/n
        rr=r*v/100
        pts.append((cx+rr*math.cos(ang),cy+rr*math.sin(ang)))
    ps=" ".join(f"{x:.1f},{y:.1f}" for x,y in pts)
    out.append(f'<polygon points="{ps}" fill="{t["fill"]}" fill-opacity=".20" stroke="{t["stroke"]}" stroke-width="2"/>')
    for x,y in pts: out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{t["vertex"]}"/>')
    out.append('</svg>')
    return "".join(out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    data=json.loads(Path(args.data).read_text())
    for theme in THEMES:
        Path(f"{args.out}-{theme}.svg").write_text(render(data,theme),encoding="utf-8")

if __name__=="__main__":
    main()
