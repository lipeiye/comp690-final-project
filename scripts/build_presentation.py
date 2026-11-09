"""Package the existing 17 HTML slides with local assets and keyboard navigation."""

from __future__ import annotations

import html
import json
from pathlib import Path
import re
import shutil

ROOT=Path(__file__).resolve().parents[1]
PRESENTATION=ROOT/"presentation"
SOURCE=PRESENTATION/"deck_source"


def main():
    deck=json.loads((SOURCE/"deck.json").read_text())
    assets=PRESENTATION/"assets";assets.mkdir(exist_ok=True)
    slides=[]
    for number,identifier in enumerate(deck["order"],1):
        path=SOURCE/"slides"/f"{identifier}.html"
        source=path.read_text()
        source=re.sub(r'(Peiye Li · COMP 690 · )\d+',lambda m:m.group(1)+str(number),source)
        if "/_blob/" in source:raise RuntimeError(f"Unresolved private asset in {path}")
        def copy_asset(match):
            name,value=match.group(1),html.unescape(match.group(2))
            if value.startswith(("http:","https:","data:")):return match.group(0)
            origin=(path.parent/value).resolve()
            if not origin.is_file():raise FileNotFoundError(origin)
            destination=assets/origin.name
            shutil.copy2(origin,destination)
            return f'{name}="assets/{html.escape(origin.name,quote=True)}"'
        source=re.sub(r'(src|data-video)="([^"]+)"',copy_asset,source)
        slides.append(source.replace("<section ",'<section class="slide" ',1))
    title=html.escape(deck["title"])
    page='''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>'''+title+'''</title><style>
*{box-sizing:border-box}html,body{margin:0;width:100%;height:100%;background:#171a20;color:white;font-family:Arial,sans-serif}
#stage{position:absolute;inset:0;overflow:hidden}.slide{position:absolute!important;width:1920px;height:1080px;transform-origin:top left;overflow:hidden;margin:0}
[hidden]{display:none!important}.slide p,.slide h2,.slide h3{margin:0}.slide ul{margin:0;padding-left:1.1em}.slide th,.slide td{padding:14px 12px}.slide aside{display:none}
#toolbar{position:fixed;bottom:8px;left:50%;transform:translateX(-50%);display:flex;align-items:center;gap:14px;background:#171a20e8;padding:7px 15px;border-radius:7px;font-size:13px;z-index:3}
button{background:#f6f5f1;border:0;border-radius:3px;padding:6px 12px;color:#14213d;cursor:pointer}button:focus-visible{outline:3px solid #f2a65a}
#notes{position:fixed;bottom:62px;left:8%;right:8%;padding:20px;background:#fff;color:#14213d;border:2px solid #b5591a;max-height:35vh;overflow:auto;line-height:1.5;z-index:5}
video{object-fit:contain;background:black}
@media print{@page{size:1920px 1080px;margin:0}html,body{width:auto;height:auto;background:white}#stage{position:static;overflow:visible}.slide,.slide[hidden]{position:relative!important;display:flex!important;transform:none!important;left:0!important;top:0!important;page-break-after:always;break-after:page;width:1920px;height:1080px}#toolbar,#notes{display:none!important}}
</style></head><body><main id="stage" aria-label="Presentation">'''+"\n".join(slides)+'''</main>
<nav id="toolbar" aria-label="Slide controls"><button id="previous" aria-label="Previous slide">←</button><span id="counter"></span><button id="next" aria-label="Next slide">→</button><button id="toggleNotes">Notes (N)</button><button id="fullscreen">Full screen (F)</button><span>Arrow keys · Home/End · click demo to play</span></nav>
<aside id="notes" hidden aria-label="Speaker notes"></aside><script>
const slides=[...document.querySelectorAll('.slide')],notes=document.getElementById('notes');let current=0;
const initial=slides.findIndex(s=>s.id===decodeURIComponent(location.hash.slice(1)));if(initial>=0)current=initial;
function layout(){const scale=Math.min(innerWidth/1920,(innerHeight-45)/1080);for(const s of slides){s.style.transform=`scale(${scale})`;s.style.left=`${(innerWidth-1920*scale)/2}px`;s.style.top=`${Math.max(0,(innerHeight-45-1080*scale)/2)}px`;}}
function show(n){current=Math.max(0,Math.min(slides.length-1,n));slides.forEach((s,i)=>{s.hidden=i!==current;s.setAttribute('aria-hidden',String(i!==current));if(i!==current)s.querySelectorAll('video').forEach(v=>v.pause());});document.getElementById('counter').textContent=`${current+1} / ${slides.length}`;notes.textContent=slides[current].querySelector('aside')?.textContent||'No speaker notes';history.replaceState(null,'','#'+slides[current].id);layout();}
function fullscreen(){if(!document.fullscreenElement)document.documentElement.requestFullscreen?.();else document.exitFullscreen?.();}
document.getElementById('previous').onclick=()=>show(current-1);document.getElementById('next').onclick=()=>show(current+1);
document.getElementById('toggleNotes').onclick=()=>notes.hidden=!notes.hidden;document.getElementById('fullscreen').onclick=fullscreen;
addEventListener('keydown',e=>{if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();show(current+1);}else if(['ArrowLeft','PageUp'].includes(e.key))show(current-1);else if(e.key==='Home')show(0);else if(e.key==='End')show(slides.length-1);else if(e.key.toLowerCase()==='n')notes.hidden=!notes.hidden;else if(e.key.toLowerCase()==='f')fullscreen();});
document.querySelectorAll('[data-video]').forEach(img=>{img.tabIndex=0;img.setAttribute('role','button');img.setAttribute('aria-label','Play recorded EEG replay');function play(){const video=document.createElement('video');video.src=img.dataset.video;video.poster=img.src;video.controls=true;video.style.cssText=img.style.cssText;img.replaceWith(video);video.play();}img.onclick=play;img.onkeydown=e=>{if(e.key==='Enter'){e.stopPropagation();play();}};});
addEventListener('resize',layout);show(current);
</script></body></html>'''
    (PRESENTATION/"index.html").write_text(page)
    print(f"Built {len(slides)} slides with {len(list(assets.iterdir()))} local assets.")


if __name__=="__main__":main()
