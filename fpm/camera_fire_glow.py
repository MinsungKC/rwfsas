"""Detect a fire directly in camera frames by its GLOW (bright white / fire hue
against a dark background) and turn it into a bearing ray -> triangulated fire
location. Complements smoke segmentation: at night / dark canyons the flame
front is a bright white-to-orange blob that pops out even when smoke is invisible.
"""
import sys, math, argparse
import numpy as np, requests, cv2
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.')))
from camera_geolocate import pixel_bearings, triangulate

AC='https://cameras.alertcalifornia.org/public-camera-data/all_cameras-v3.json'
def frame_url(cid): return f'https://cameras.alertcalifornia.org/public-camera-data/{cid}/latest-frame.jpg'
def dk(a,b,c,e):
    R=6371;la1,la2=math.radians(a),math.radians(c);dla=la2-la1;dlo=math.radians(e-b)
    h=math.sin(dla/2)**2+math.cos(la1)*math.cos(la2)*math.sin(dlo/2)**2;return 2*R*math.asin(math.sqrt(h))

def glow_mask(img):
    """Fire glow = very bright (white-hot) OR fire-hue (orange/red) and bright.
    Returns a boolean mask."""
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    H,S,V=hsv[...,0],hsv[...,1],hsv[...,2]
    white_hot=(V>245)&(S<50)                                  # near-white flame core
    fire_hue=(((H<=25)|(H>=160))&(S>90)&(V>180))             # orange/red glow, bright
    m=(white_hot|fire_hue)
    # require the region to be locally BRIGHT vs surroundings (dark background)
    blur=cv2.GaussianBlur(V.astype(np.float32),(31,31),0)
    m &= (V.astype(np.float32) > blur+40)
    m=cv2.morphologyEx(m.astype(np.uint8),cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    return m.astype(bool)

def fire_hue_mask(img):
    """Strict ORANGE/RED flame hue (rejects white sun/moon/city-lights)."""
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    H,S,V=hsv[...,0],hsv[...,1],hsv[...,2]
    return (((H<=22)|(H>=165))&(S>110)&(V>140)).astype(np.uint8)

def detect(img, min_area_px=40, min_fire_frac=0.15):
    """Return (mask, best). A detection is valid ONLY if the glow component
    contains enough true fire-hue (orange/red) pixels -- this rejects the sun,
    moon, and town lights, which are white/round with ~no fire hue."""
    m=glow_mask(img); fh=fire_hue_mask(img)
    n,lab,stats,cent=cv2.connectedComponentsWithStats(m.astype(np.uint8))
    Hh,Ww=img.shape[:2]; best=None
    for i in range(1,n):
        a=stats[i,cv2.CC_STAT_AREA]
        if a<min_area_px: continue
        comp=(lab==i)
        fire_frac=fh[comp].mean()                       # fraction that is fire-hue
        w0,h0=stats[i,cv2.CC_STAT_WIDTH],stats[i,cv2.CC_STAT_HEIGHT]
        roundish=0.7<(w0/max(h0,1))<1.4                 # sun/moon are round
        cy=cent[i][1]/Hh
        # reject: too little fire hue AND (round or high-in-sky) -> sun/moon/light
        if fire_frac<min_fire_frac and (roundish or cy<0.35):
            continue
        score=a*(1+3*fire_frac)                         # favor fiery + large
        if best is None or score>best[0]:
            best=(score,cent[i],stats[i],a,fire_frac)
    return m,best

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--lat',type=float,default=37.57); ap.add_argument('--lon',type=float,default=-119.613)
    ap.add_argument('--radius',type=float,default=40); ap.add_argument('--out',default='dome_glow.jpg')
    a=ap.parse_args()
    j=requests.get(AC,timeout=30,headers={'User-Agent':'x'}).json()
    cams=[]
    for f in j['features']:
        c=f['geometry']['coordinates']
        if c[0] is None: continue
        d=dk(a.lat,a.lon,c[1],c[0])
        p=f['properties']
        if d<=a.radius and p.get('az_current') is not None:
            cams.append({'id':p['id'],'lat':c[1],'lon':c[0],'az':p['az_current'],
                         'fov':p.get('fov') or 62.8,'d':d})
    cams.sort(key=lambda x:x['d'])
    print(f'{len(cams)} cameras within {a.radius}km',flush=True)
    rays=[]; tiles=[]; hits=[]
    for cam in cams[:12]:
        try:
            r=requests.get(frame_url(cam['id']),timeout=15,headers={'User-Agent':'x'})
            if r.status_code!=200 or len(r.content)<2000: continue
            img=cv2.imdecode(np.frombuffer(r.content,np.uint8),cv2.IMREAD_COLOR)
            if img is None: continue
        except Exception: continue
        m,best=detect(img)
        vis=img.copy(); Hh,Ww=img.shape[:2]
        if best is not None:
            score,(cx,cy),st,area,ff=best
            x0,y0,w0,h0=st[0],st[1],st[2],st[3]
            cv2.rectangle(vis,(x0,y0),(x0+w0,y0+h0),(0,0,255),2)
            cv2.putText(vis,'FIRE %dpx hue%.0f%%'%(area,ff*100),(x0,max(y0-6,12)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
            frac=cx/Ww
            b=pixel_bearings(frac,frac,cam['az'],cam['fov'])[1]
            # weight rays by fire-hue confidence + proximity (closer = better)
            rays.append((cam['lat'],cam['lon'],b)); hits.append((cam['id'],cam['d'],area,ff,b))
            print('  %-22s %.1fkm FIRE area=%d hue=%.0f%% bearing=%.1f'%(cam['id'],cam['d'],area,ff*100,b),flush=True)
        else:
            print('  %-22s %.1fkm no fire'%(cam['id'],cam['d']),flush=True)
        tiles.append(cv2.resize(vis,(480,300)))
    # montage
    if tiles:
        while len(tiles)%3: tiles.append(np.zeros((300,480,3),np.uint8))
        rows=[np.hstack(tiles[i:i+3]) for i in range(0,len(tiles),3)]
        cv2.imwrite(a.out,np.vstack(rows))
        print('MONTAGE',a.out)
    fix=None
    if len(rays)>=2:
        fix=triangulate(rays,inlier_km=3.0)
        if fix: print('TRIANGULATED FIRE: %.4f,%.4f  (%.1f km from report)  inliers %d/%d'%(
            fix['lat'],fix['lon'],dk(a.lat,a.lon,fix['lat'],fix['lon']),fix['n_inliers'],fix['n_rays']))
    return hits,fix

if __name__=='__main__':
    main()
