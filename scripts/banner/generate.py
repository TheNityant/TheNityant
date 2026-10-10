#!/usr/bin/env python3
from __future__ import annotations

import html
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "assets" / "source" / "nityant.png"
ASSETS = ROOT / "assets"

W, H = 1180, 610
SEED = 24042026
TRAVELLERS = 520
LOOP = 12.6
INTRO = 2.7

ROWS = [
    ("Subject", "Nityant Tiwari"),
    ("Role", "Software Developer · AI Engineering"),
    ("Location", "Mumbai · India"),
    ("Education", "B.Tech ECS · SFIT"),
    ("Status", "Building + Learning + Shipping"),
    ("ToolChain", "VS Code · Git · Docker · Linux"),
    ("Core.Lang", "Python · Java"),
    ("Core.Backend", "Spring Boot · FastAPI"),
    ("Core.Data", "PostgreSQL · MongoDB"),
    ("Core.AI", "Transformers · LLMs"),
    ("Grid.Portfolio", "thenityant.runs-on.dev"),
    ("Grid.GitHub", "@TheNityant"),
]

THEMES = {
    "dark": {
        "bg": "#0D1117",
        "panel": "#0F141B",
        "panel2": "#111821",
        "line": "#30363D",
        "muted": "#8B949E",
        "text": "#F0F6FC",
        "portrait": "#FFA500",
        "portrait2": "#F0F6FC",
        "chrome": "#FFA500",
        "accent": "#3FB950",
        "shadow": "#000000",
    },
    "light": {
        "bg": "#F6F8FA",
        "panel": "#FFFFFF",
        "panel2": "#F0F3F6",
        "line": "#D0D7DE",
        "muted": "#57606A",
        "text": "#1F2328",
        "portrait": "#C46A00",
        "portrait2": "#24292F",
        "chrome": "#C46A00",
        "accent": "#1A7F37",
        "shadow": "#8C959F",
    },
}

def floyd_steinberg(gray: np.ndarray) -> np.ndarray:
    work = gray.astype(np.float32) / 255.0
    out = np.zeros_like(work, dtype=bool)
    h, w = work.shape
    for y in range(h):
        ltr = (y % 2 == 0)
        xs = range(w) if ltr else range(w - 1, -1, -1)
        direction = 1 if ltr else -1
        for x in xs:
            old = work[y, x]
            new = 1.0 if old >= 0.5 else 0.0
            out[y, x] = bool(new)
            err = old - new
            nx = x + direction
            if 0 <= nx < w:
                work[y, nx] += err * 7 / 16
            if y + 1 < h:
                if 0 <= x - direction < w:
                    work[y + 1, x - direction] += err * 3 / 16
                work[y + 1, x] += err * 5 / 16
                if 0 <= nx < w:
                    work[y + 1, nx] += err * 1 / 16
    return out

def subject_crop(image: Image.Image) -> Image.Image:
    # Wide enough for complete hair, face, collar and shoulders.
    w, h = image.size
    # Source images in this project are usually square/portrait.
    left = int(w * 0.11)
    right = int(w * 0.89)
    top = int(h * 0.04)
    bottom = int(h * 0.97)
    return image.crop((left, top, right, bottom))

def portrait_points(theme: str, rng: np.random.Generator):
    src = Image.open(SOURCE).convert("RGBA")
    crop = subject_crop(src).resize((300, 340), Image.Resampling.LANCZOS)

    rgb = crop.convert("RGB")
    alpha = np.asarray(crop.getchannel("A"), dtype=np.float32) / 255.0

    # If the source has no alpha, infer useful subject signal from darkness/brightness.
    if alpha.mean() > 0.99:
        arr = np.asarray(rgb, dtype=np.float32)
        # estimate dark background from corners
        corners = np.concatenate([
            arr[:20,:20].reshape(-1,3), arr[:20,-20:].reshape(-1,3),
            arr[-20:,:20].reshape(-1,3), arr[-20:,-20:].reshape(-1,3)
        ], axis=0)
        bg = np.median(corners, axis=0)
        delta = np.linalg.norm(arr - bg, axis=2)
        alpha = np.clip((delta - 9) / 48, 0, 1)

    lum = np.asarray(ImageOps.grayscale(rgb), dtype=np.float32)
    prepared = Image.fromarray(np.uint8(np.clip(lum * alpha, 0, 255)), "L")
    mask = Image.fromarray(np.uint8((alpha > 0.06) * 255), "L")
    prepared = ImageOps.equalize(prepared, mask=mask)
    prepared = ImageEnhance.Contrast(prepared).enhance(1.45)
    prepared = prepared.filter(ImageFilter.UnsharpMask(radius=1.8, percent=190, threshold=1))

    bits = floyd_steinberg(np.asarray(prepared))
    active = bits & (alpha > 0.06)
    ys, xs = np.where(active)

    if not len(xs):
        return np.zeros((0,2), np.float32), np.zeros((0,), bool)

    # Banner coordinates inside VISUAL.MAP.
    pts = np.column_stack((76 + xs, 158 + ys)).astype(np.float32)

    # Keep a dense but manageable portrait.
    if len(pts) > 15000:
        pick = rng.choice(len(pts), 15000, replace=False)
        pts = pts[pick]
        xs = xs[pick]
        ys = ys[pick]

    # White accents correspond to bright/neutral source pixels.
    arr = np.asarray(rgb)
    sampled = arr[ys, xs]
    brightness = sampled.mean(axis=1)
    neutrality = sampled.max(axis=1) - sampled.min(axis=1)
    white = (brightness > 145) & (neutrality < 70)
    return pts, white

def point_path(points: np.ndarray) -> str:
    if not len(points):
        return ""
    p = np.rint(points).astype(int)
    unique = sorted({(int(x),int(y)) for x,y in p}, key=lambda q:(q[1],q[0]))
    chunks = []
    i = 0
    while i < len(unique):
        x0, y = unique[i]
        x1 = x0
        i += 1
        while i < len(unique) and unique[i][1] == y and unique[i][0] <= x1 + 1:
            x1 = unique[i][0]
            i += 1
        chunks.append(f"M{x0} {y}h{x1-x0+1}")
    return "".join(chunks)

def sample_polyline(poly, count, rng):
    segs = []
    lengths = []
    for a,b in zip(poly[:-1], poly[1:]):
        a=np.array(a,float); b=np.array(b,float)
        segs.append((a,b))
        lengths.append(np.linalg.norm(b-a))
    lengths=np.array(lengths)
    probs=lengths/lengths.sum()
    choice=rng.choice(len(segs), count, p=probs)
    pts=[]
    for i in choice:
        a,b=segs[i]
        t=rng.random()
        p=a*(1-t)+b*t+rng.normal(0,1.8,2)
        pts.append(p)
    return np.array(pts,np.float32)

def target_code(count, rng):
    # </> symbol inside visual panel
    polylines = [
        [(145,245),(102,328),(145,411)],
        [(265,245),(308,328),(265,411)],
        [(232,232),(184,424)],
    ]
    alloc=[int(count*.34),int(count*.34)]
    alloc.append(count-sum(alloc))
    return np.vstack([sample_polyline(p,n,rng) for p,n in zip(polylines,alloc)])

def target_attention(count, rng):
    # "attention" lattice: 7x7 nodes + linking diagonals.
    grid=[]
    xs=np.linspace(115,300,7)
    ys=np.linspace(236,421,7)
    for y in ys:
        for x in xs:
            grid.append((x,y))
    base=np.array(grid,float)
    pick=base[rng.integers(0,len(base),size=count)]
    return (pick + rng.normal(0,5.5,(count,2))).astype(np.float32)

def target_terminal(count, rng):
    # compact { } + prompt-like baseline
    left=[(137,245),(115,275),(125,322),(105,346),(125,370),(115,414),(137,438)]
    right=[(278,245),(300,275),(290,322),(310,346),(290,370),(300,414),(278,438)]
    prompt=[(158,314),(190,346),(158,378)]
    bar=[(205,382),(270,382)]
    alloc=[int(count*.30),int(count*.30),int(count*.22)]
    alloc.append(count-sum(alloc))
    return np.vstack([
        sample_polyline(left,alloc[0],rng),
        sample_polyline(right,alloc[1],rng),
        sample_polyline(prompt,alloc[2],rng),
        sample_polyline(bar,alloc[3],rng),
    ])

def transport(src, target):
    rows, cols = linear_sum_assignment(cdist(src, target, metric="sqeuclidean"))
    ordered=np.empty_like(target)
    ordered[rows]=target[cols]
    return ordered

def dotted_leader(x1,x2,y):
    if x2 <= x1: return ""
    return "".join(f"M{x:.1f} {y:.1f}h1" for x in np.arange(x1,x2,5))

def text_width(s, size):
    return len(s)*size*.605

def fmt(v):
    return f"{v:.3f}".rstrip("0").rstrip(".")

def render(theme_name, portrait, white_mask, rng):
    t=THEMES[theme_name]
    n=min(TRAVELLERS,len(portrait))
    ids=rng.choice(len(portrait), n, replace=False)
    source=portrait[ids]
    source_white=white_mask[ids]

    code=transport(source, target_code(n,rng))
    attn=transport(code, target_attention(n,rng))
    term=transport(attn, target_terminal(n,rng))

    # 12.6 sec: portrait hold → code → attention → terminal → portrait
    times=[0,2.1,3.2,5.0,6.1,7.9,9.0,10.8,12.6]
    kt=";".join(fmt(x/LOOP) for x in times)
    frames=[source,source,code,code,attn,attn,term,term,source]

    def values_for(i):
        return ";".join(f"{fmt(p[i,0])} {fmt(p[i,1])}" for p in frames)

    parts=[
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">',
        '<title id="title">Nityant live system profile</title>',
        '<desc id="desc">Animated terminal profile with tensor portrait, code, attention-grid and terminal morphs.</desc>',
        '<defs>',
        f'<filter id="shadow" x="-20%" y="-20%" width="140%" height="150%"><feDropShadow dx="0" dy="12" stdDeviation="16" flood-color="{t["shadow"]}" flood-opacity=".26"/></filter>',
        f'<filter id="glow" x="-100%" y="-100%" width="300%" height="300%"><feGaussianBlur stdDeviation="2.4" result="b"/><feFlood flood-color="{t["chrome"]}" flood-opacity=".28"/><feComposite in2="b" operator="in"/><feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge></filter>',
        '<clipPath id="visualClip"><rect x="49" y="124" width="390" height="414" rx="3"/></clipPath>',
        '</defs>',
        f'<rect width="{W}" height="{H}" rx="18" fill="{t["bg"]}"/>',
        f'<rect x="13" y="13" width="1154" height="584" rx="13" fill="{t["panel"]}" stroke="{t["line"]}" filter="url(#shadow)"/>',
        f'<path d="M13 62H1167" stroke="{t["line"]}"/>',
        '<circle cx="38" cy="38" r="6" fill="#FF5F57"/><circle cx="59" cy="38" r="6" fill="#FEBC2E"/><circle cx="80" cy="38" r="6" fill="#28C840"/>',
        f'<text x="590" y="43" text-anchor="middle" fill="{t["muted"]}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" letter-spacing=".4">profile.sh --live</text>',
        # visual panel
        f'<rect x="35" y="88" width="418" height="472" rx="6" fill="{t["panel2"]}" stroke="{t["line"]}"/>',
        f'<path d="M35 124H453" stroke="{t["line"]}"/>',
        f'<text x="49" y="111" fill="{t["chrome"]}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" font-weight="700" letter-spacing="1.2">VISUAL.MAP</text>',
        f'<text x="438" y="111" text-anchor="end" fill="{t["muted"]}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11">300×340 / TENSOR</text>',
        f'<path d="M49 141h12M49 141v12M439 141h-12M439 141v12M49 539h12M49 539v-12M439 539h-12M439 539v-12" fill="none" stroke="{t["chrome"]}" opacity=".55"/>',
        '<g clip-path="url(#visualClip)" shape-rendering="crispEdges">'
    ]

    # Split base portrait into orange / white groups and small intro bands.
    for color_name, mask in [("portrait", ~white_mask), ("portrait2", white_mask)]:
        pts = portrait[mask]
        if not len(pts): continue
        band_ids=rng.integers(0,60,size=len(pts))
        for band in range(60):
            bp=pts[band_ids==band]
            if not len(bp): continue
            start=.04 + (band/59)*1.10
            parts.append(
                f'<path d="{point_path(bp)}" fill="none" stroke="{t[color_name]}" stroke-width="1" opacity="0">'
                f'<animate attributeName="opacity" begin="{start:.2f}s" dur=".72s" values="0;.95" fill="freeze"/>'
                f'<animate attributeName="opacity" begin="{INTRO}s" dur="{LOOP}s" repeatCount="indefinite" keyTimes="{kt}" values=".95;.95;.10;.10;.10;.10;.10;.10;.95"/>'
                '</path>'
            )

    # moving travellers
    for i in range(n):
        color=t["portrait2"] if source_white[i] else t["portrait"]
        parts.append(
            f'<path d="M-.75-.75h1.5v1.5h-1.5z" fill="{color}" opacity="0">'
            f'<animate attributeName="opacity" begin="{INTRO}s" dur="{LOOP}s" repeatCount="indefinite" keyTimes="{kt}" values="0;0;1;1;1;1;1;1;0"/>'
            f'<animateTransform attributeName="transform" type="translate" begin="{INTRO}s" dur="{LOOP}s" repeatCount="indefinite" calcMode="linear" keyTimes="{kt}" values="{values_for(i)}"/>'
            '</path>'
        )

    parts += [
        '</g>',
        f'<text x="58" y="551" fill="{t["muted"]}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="10">PTS {len(portrait):05d} · ORANGE/WHITE · VECTOR</text>',
        # right system panel
        f'<rect x="474" y="88" width="672" height="472" rx="6" fill="{t["panel2"]}" stroke="{t["line"]}"/>',
        f'<path d="M474 124H1146" stroke="{t["line"]}"/>',
        f'<text x="490" y="111" fill="{t["chrome"]}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" font-weight="700" letter-spacing="1.2">SYSTEM.INFO</text>',
        '<g filter="url(#glow)"><circle cx="916" cy="106" r="4" fill="#FF4D5A"><animate attributeName="opacity" values="1;.3;1" dur="1.6s" repeatCount="indefinite"/></circle></g>',
        '<text x="928" y="111" fill="#FF4D5A" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="12" font-weight="700">LIVE</text>',
        f'<rect x="986" y="94" width="142" height="24" rx="12" fill="{t["chrome"]}" opacity=".14" stroke="{t["chrome"]}"/>',
        f'<text x="1057" y="111" text-anchor="middle" fill="{t["chrome"]}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="13" font-weight="700">@TheNityant</text>',
    ]

    value_right=1127.0
    row_y=154.0
    for label,value in ROWS:
        fs=14
        value_len=text_width(value,fs)
        label_len=text_width(label,fs)
        leader_start=491+label_len+12
        leader_end=value_right-value_len-12
        parts += [
            f'<text x="491" y="{row_y:.1f}" fill="{t["muted"]}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="{fs}">{html.escape(label)}</text>',
            f'<path d="{dotted_leader(leader_start,leader_end,row_y-4)}" fill="none" stroke="{t["line"]}" stroke-width="1" shape-rendering="crispEdges"/>',
            f'<text x="{value_right:.1f}" y="{row_y:.1f}" text-anchor="end" fill="{t["text"]}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="{fs}">{html.escape(value)}</text>',
        ]
        row_y += 28.0

    parts += [
        f'<path d="M490 530H1130" stroke="{t["line"]}"/>',
        f'<text x="491" y="548" fill="{t["accent"]}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11">● ALL SYSTEMS NOMINAL</text>',
        f'<text x="1128" y="548" text-anchor="end" fill="{t["muted"]}" font-family="ui-monospace,SFMono-Regular,Consolas,monospace" font-size="11">IST · INDIA NODE</text>',
        '</svg>'
    ]
    return "".join(parts)

def main():
    ASSETS.mkdir(exist_ok=True)
    for idx,theme in enumerate(THEMES):
        rng=np.random.default_rng(SEED+idx)
        pts,white=portrait_points(theme,rng)
        svg=render(theme,pts,white,np.random.default_rng(SEED+100+idx))
        out=ASSETS/f"banner-{theme}.v1.svg"
        out.write_text(svg,encoding="utf-8")
        print(f"{out.name}: {out.stat().st_size/1024:.1f} KiB, {len(pts)} portrait points")

if __name__=="__main__":
    main()
