"""fusion_v3 -- GENERAL fire-perimeter model (not per-incident).

run(name, lat, lon) does, for ANY fire:
  1. pull the latest MAPPED perimeter (WFIGS/FIRIS by location) as prior/base
  2. VIIRS 375m + GOES FDCC hotspots -> data belief
  3. CAMERAS near the fire: smoke segmentation (best_seg_v2) for plume-base
     bearings + direct fire-GLOW edges -> angular constraints (saved as a
     labelled montage)
  4. wind + Rothermel (fuel/slope) -> a SPREAD DIRECTION + rate field
  5. fuse everything into a burn-belief grid, then let the belief spread by
     ANISOTROPIC (physics) diffusion along the Rothermel direction -- the shape
     follows the DATA + PHYSICS, no pre-imposed ellipse
  6. Albini ember spotting downwind of the leading edge
Outputs <name>_cameras.jpg, <name>_perimeter.png, <name>_perimeter.geojson.
"""
import os, sys, math, json, time, urllib.request, urllib.parse
os.environ['AWS_NO_SIGN_REQUEST']='YES'
from datetime import date, datetime, timezone, timedelta
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../sim')))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../fire-spread-lab')))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../fire-spread-lab-claude/scripts')))
from shapely.geometry import shape, Point, Polygon, mapping
from shapely.ops import unary_union, transform
from scipy.ndimage import gaussian_filter, rotate as ndrot

def satellite_basemap(bbox, zoom=14):
    """Esri World Imagery XYZ tiles mosaicked over bbox (no auth). Returns
    (rgb array, [W,E,S,N] extent). Web Mercator tiles plotted on a lon/lat axis
    -- fine at fire scale."""
    import io
    from PIL import Image
    W,S,E,N=bbox
    def deg2tile(lat,lon,z):
        n=2**z; xt=(lon+180.0)/360.0*n
        lr=math.radians(lat); yt=(1-math.log(math.tan(lr)+1/math.cos(lr))/math.pi)/2*n
        return xt,yt
    def tile2lon(x,z): return x/2**z*360.0-180.0
    def tile2lat(y,z):
        m=math.pi*(1-2*y/2**z); return math.degrees(math.atan(math.sinh(m)))
    x0f,y0f=deg2tile(N,W,zoom); x1f,y1f=deg2tile(S,E,zoom)
    x0,x1=int(math.floor(x0f)),int(math.floor(x1f)); y0,y1=int(math.floor(y0f)),int(math.floor(y1f))
    nx,ny=x1-x0+1,y1-y0+1
    if nx*ny>64: return None,None
    mos=Image.new('RGB',(nx*256,ny*256))
    for tx in range(x0,x1+1):
        for ty in range(y0,y1+1):
            url=f'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{ty}/{tx}'
            try:
                r=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'x'}),timeout=20).read()
                mos.paste(Image.open(io.BytesIO(r)).convert('RGB'),((tx-x0)*256,(ty-y0)*256))
            except Exception: pass
    ext=[tile2lon(x0,zoom),tile2lon(x1+1,zoom),tile2lat(y1+1,zoom),tile2lat(y0,zoom)]
    return np.asarray(mos),ext


def _get(u,timeout=40):
    try: return json.loads(urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'x'}),timeout=timeout).read())
    except Exception: return None

def mapped_perimeter(lat,lon,pad=0.06):
    """Fetch the current official perimeter (WFIGS, then FIRIS).

    Returns (geom, props, status) where status is 'found', 'not_found' (both
    services were reached and neither has a perimeter here -- a real fresh-
    fire signal), or 'unavailable' (at least one query could not be
    completed). Callers must not treat 'unavailable' the same as 'not_found':
    run() uses 'not_found' to justify the fresh-fire origin-only anchor, and
    doing that on a fetch failure would silently discard a real existing
    perimeter (rebuild-spec Ground Rule #1).
    """
    env=json.dumps({'xmin':lon-pad,'ymin':lat-pad,'xmax':lon+pad,'ymax':lat+pad,'spatialReference':{'wkid':4326}})
    errors=[]
    for base in ['https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Interagency_Perimeters_Current/FeatureServer/0/query',
                 'https://bz1uwwpkuinzbk94.svcs5.arcgis.com/bz1uwWPKUInZBK94/arcgis/rest/services/CA_Perimeters_NIFC_FIRIS_public_view/FeatureServer/0/query']:
        url=base+'?'+urllib.parse.urlencode({'where':'1=1','geometry':env,'geometryType':'esriGeometryEnvelope','inSR':'4326','spatialRel':'esriSpatialRelIntersects','outFields':'*','f':'geojson','geometryPrecision':'6','resultRecordCount':'5'})
        try:
            d=json.loads(urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'x'}),timeout=40).read())
        except Exception as e:
            errors.append(f'{base}: {e!r}'); continue
        if d.get('features'):
            f=max(d['features'],key=lambda x:shape(x['geometry']).area)
            return shape(f['geometry']), f['properties'], 'found'
    return None, None, ('unavailable' if errors else 'not_found')

def cameras_smoke_and_fire(lat,lon,out_jpg,radius_km=40,conf=0.30):
    """Smoke segmentation (best_seg_v2) for plume-base bearings + fire-glow edges.
    Returns (rays, wedges, montage_path). Saves a labelled montage."""
    import requests, cv2
    from camera_geolocate import pixel_bearings
    from camera_fire_glow import detect as glow_detect, frame_url, dk
    from locate_fire_from_cameras import plume_base_extent, upwind_source_frac
    try:
        from ultralytics import YOLO
        seg=YOLO(os.path.join(HERE,'best_seg_v2.pt'))
    except Exception as e:
        seg=None; print('seg load fail',e)
    j=requests.get('https://cameras.alertcalifornia.org/public-camera-data/all_cameras-v3.json',timeout=30,headers={'User-Agent':'x'}).json()
    cams=[]
    for f in j['features']:
        c=f['geometry']['coordinates']
        if c[0] is None: continue
        d=dk(lat,lon,c[1],c[0]); p=f['properties']
        if d<=radius_km and p.get('az_current') is not None:
            cams.append({'id':p['id'],'lat':c[1],'lon':c[0],'az':p['az_current'],'fov':p.get('fov') or 62.8,'d':d})
    cams.sort(key=lambda x:x['d'])
    # wind for upwind plume-base disambiguation
    wj=_get(f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=wind_direction_10m')
    wind_from=float(wj['current']['wind_direction_10m']) if wj else None
    rays=[]; wedges=[]; tiles=[]
    for cam in cams[:9]:
        try:
            r=requests.get(frame_url(cam['id']),timeout=12,headers={'User-Agent':'x'}); img=cv2.imdecode(np.frombuffer(r.content,np.uint8),cv2.IMREAD_COLOR)
        except Exception: img=None
        if img is None: continue
        vis=img.copy(); H,W=img.shape[:2]; got=''
        # SMOKE segmentation (TILED: full frame + crops, so distant/thin plumes
        # are larger relative to the 640px model input) -> plume base -> bearing
        if seg is not None:
            best_sm=None  # (conf, mask_full_bool, poly_full)
            smtiles=[(0,W,0,H),(0,int(W*0.55),0,int(H*0.72)),(int(W*0.22),int(W*0.78),0,int(H*0.72)),(int(W*0.45),W,0,int(H*0.72))]
            for (x0,x1,y0,y1) in smtiles:
                crop=img[y0:y1,x0:x1]
                rc=seg.predict(crop,conf=conf,verbose=False,retina_masks=True,imgsz=640)[0]
                if rc.masks is None or len(rc.masks)==0: continue
                cfs=rc.boxes.conf.cpu().numpy(); k=int(np.argmax(cfs))
                mk=rc.masks.data[k].cpu().numpy()>0.5
                mk=cv2.resize(mk.astype(np.uint8),(x1-x0,y1-y0),interpolation=cv2.INTER_NEAREST).astype(bool)
                full=np.zeros((H,W),bool); full[y0:y1,x0:x1]=mk
                if best_sm is None or cfs[k]>best_sm[0]:
                    cnts,_=cv2.findContours(full.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    poly=max(cnts,key=cv2.contourArea).reshape(-1,2) if cnts else None
                    best_sm=(float(cfs[k]),full,poly)
            if best_sm is not None and best_sm[2] is not None:
                confs=[best_sm[0]]; bi=0; poly=best_sm[2].astype(np.int32); _mfull=best_sm[1]
                overlay=vis.copy(); cv2.fillPoly(overlay,[poly],(0,140,255)); vis=cv2.addWeighted(overlay,0.35,vis,0.65,0)
                cv2.polylines(vis,[poly],True,(0,220,255),2)
                m=_mfull
                ext=plume_base_extent(m,confs[bi])
                if ext is not None:
                    xl,xc,xr=ext
                    if wind_from is not None:
                        srcf,which=upwind_source_frac(xl,xc,xr,cam['az'],wind_from)
                    else: srcf,which=xc,'centre'
                    b=pixel_bearings(srcf,srcf,cam['az'],cam['fov'])[1]
                    rays.append((cam['lat'],cam['lon'],b)); got='smoke %.2f'%confs[bi]
                    cv2.putText(vis,'plume-base src %.0f'%b,(8,H-12),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        # FIRE GLOW edges -> wedge
        m2,best=glow_detect(img)
        if best is not None:
            score,(cx,cy),st,area,ff=best
            if ff>=0.25:
                x0,w0=st[0],st[2]
                bl=pixel_bearings(x0/W,x0/W,cam['az'],cam['fov'])[1]; br=pixel_bearings((x0+w0)/W,(x0+w0)/W,cam['az'],cam['fov'])[1]
                lo_,hi_=sorted([bl,br]); wedges.append((cam,lo_,hi_))
                cv2.rectangle(vis,(x0,st[1]),(x0+w0,st[1]+st[3]),(0,0,255),2); got+=' glow%.0f%%'%(ff*100)
        cv2.putText(vis,'%s %.0fkm %s'%(cam['id'][:16],cam['d'],got),(8,24),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
        tiles.append(cv2.resize(vis,(480,300)))
    import cv2 as _cv
    if tiles:
        while len(tiles)%3: tiles.append(np.zeros((300,480,3),np.uint8))
        rows=[np.hstack(tiles[i:i+3]) for i in range(0,len(tiles),3)]
        _cv.imwrite(out_jpg,np.vstack(rows))
    return rays,wedges,wind_from

def run(name,lat,lon,out_dir):
    _t_start=time.time()
    import fire_fusion as FF
    import data_ingest as DI, rothermel as rm
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from camera_geolocate import _fwd
    os.makedirs(out_dir,exist_ok=True)
    half=0.11; bbox=(lon-half/math.cos(math.radians(lat)),lat-half,lon+half/math.cos(math.radians(lat)),lat+half)
    mx=111320*math.cos(math.radians(lat)); my=111320
    ABI=os.path.join(out_dir,'_abi'); os.makedirs(ABI,exist_ok=True)
    # 1. mapped base + its MAP TIME (the anchor; growth is measured from here)
    import source_health as SH
    base,baseprops,base_status=mapped_perimeter(lat,lon)
    SH.record(out_dir, 'mapped_perimeter',
              'ok' if base_status=='found' else ('no_data' if base_status=='not_found' else 'unavailable'),
              reason='' if base_status!='unavailable' else 'WFIGS and FIRIS both unreachable')
    if base_status=='unavailable':
        print('  ! mapped_perimeter UNAVAILABLE (WFIGS+FIRIS unreachable) -- NOT treating as fresh-fire',flush=True)
    base_ac=(transform(lambda x,y,z=None:((x-lon)*mx,(y-lat)*my),base).area/4046.86) if base else 0
    t_map=None; t_src=None
    if baseprops:
        # poly_PolygonDateTime = the time the mapped polygon actually DEPICTS the
        # fire (flight/observation time) -> the correct anchor for measuring
        # growth. poly_DateCurrent / CreateDate are DB stamps that refresh on ANY
        # metadata edit (that made Dome look ~0h old, starving GOES and forcing
        # VIIRS to fill). FIRIS view exposes bare field names as fallback.
        for k in ('poly_PolygonDateTime','PolygonDateTime','poly_DateCurrent',
                  'DateCurrent','poly_CreateDate','CreateDate'):
            dc=baseprops.get(k)
            if isinstance(dc,(int,float)) and dc>1e11:
                t_map=datetime.fromtimestamp(dc/1000,timezone.utc); t_src=k; break
    if t_map is not None:
        print('  mapped base %.0f ac as-of %s (%s) -> measure GOES growth from here'%(base_ac,t_map.strftime('%m-%d %H:%MZ'),t_src),flush=True)
    # 2/3. cameras (smoke + glow), labelled montage.
    # DISABLE_CAMERAS flag file (fpm/DISABLE_CAMERAS) turns cameras OFF -- the
    # running 5-min loops pick this up next cycle (no restart). Use while the
    # smoke model is being re-labelled/fine-tuned, since unreliable detections
    # + bearing error were degrading the perimeter.
    cam_jpg=os.path.join(out_dir,f'{name}_cameras.jpg')
    if os.path.exists(os.path.join(HERE,'DISABLE_CAMERAS')):
        rays,wedges,wind_from=[],[],None
        print('cameras: DISABLED (flag file present)',flush=True)
    else:
        rays,wedges,wind_from=cameras_smoke_and_fire(lat,lon,cam_jpg)
        print(f'cameras: {len(rays)} smoke-plume rays, {len(wedges)} fire-glow wedges',flush=True)
    # 4. belief grid + data (VIIRS/GOES) + mapped base + camera constraints
    G=FF.BeliefGrid(bbox,res_m=80,prior_p=0.10); burned_geoms=[]; layers={}
    # --- 5-MINUTE TEMPORAL DESIGN ---
    # This runs every 5 min. VIIRS is the best SHAPE source but only passes every
    # few hours, so its weight DECAYS with overpass age; between passes the
    # always-on 5-min sources (GOES C07 raw thermal, FDCC, ADP smoke) + cameras
    # + Rothermel physics propagate the last perimeter forward.
    import firms_fixed as _F
    vage_h=99.0
    try:
        _v=_F.fetch(bbox,date.today()-timedelta(days=2),date.today())
        if _v:
            def _pa(s):
                s=s.replace('Z',''); dt=datetime.fromisoformat(s)
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            _last=max(_pa(d['acq']) for d in _v)
            vage_h=max((datetime.now(timezone.utc)-_last).total_seconds()/3600,0.0)
    except Exception: pass
    vwt=0.92*FF.cond_weight('viirs',overpass_age_h=vage_h)   # fresh~0.9, 6h~0.45, 12h~0.18
    print(f'  VIIRS overpass age {vage_h:.1f}h -> weight {vwt:.2f} (decays so 5-min sources carry between passes)',flush=True)
    # (canonical source key -- must match the adapter's own Observation.source
    # / Unavailable.source so the health log doesn't fragment one source
    # across two different keys, display label, adapter call)
    for src,nm,fn in [('viirs','VIIRS(%.1fh)'%vage_h,lambda:FF.obs_viirs(bbox,days=2,weight=vwt)),
                  ('goes','GOES-FDCC(5min)',lambda:FF.obs_goes(bbox,ABI,weight=0.4)),
                  ('goes_c07','GOES-C07-hot(5min)',lambda:FF.obs_goes_c07_hot(bbox,ABI,weight=0.5)),
                  ('sentinel2_swir','S2-SWIR-fire',lambda:FF.obs_sentinel2_swir_fire(bbox,weight=0.85)),
                  ('goes_adp_smoke','GOES-ADP-smoke(5min)',lambda:FF.obs_goes_adp_smoke(bbox,ABI,weight=0.4))]:
        t0=time.time()
        try:
            o=fn()
            dt=time.time()-t0
            if isinstance(o,FF.Unavailable):
                # Fetch failed -- do NOT add to the belief grid (that would be
                # a silent "nothing here" vote) and mark it distinctly from a
                # real negative so a run of these is visible as a health
                # problem, not a quiet fire.
                print('  !',nm,'UNAVAILABLE:',o.reason[:120],flush=True)
                SH.record(out_dir,src,'unavailable',reason=o.reason,latency_s=dt)
            elif o:
                G.add(o); print('  +',nm,o.note,flush=True); layers[o.source]=o
                if o.kind=='burned' and o.geom is not None: burned_geoms.append(o.geom)
                SH.record(out_dir,src,'ok',reason=o.note,latency_s=dt,record_count=1)
            else:
                print('  ~',nm,'none',flush=True)
                SH.record(out_dir,src,'no_data',latency_s=dt)
        except Exception as e:
            print('  !',nm,repr(e)[:70])
            SH.record(out_dir,src,'unavailable',reason=repr(e)[:200],latency_s=time.time()-t0)
    if base is not None:
        G.add(FF.Observation('mapped','burned',geom=base,sigma_m=60,weight=0.95,note='mapped base')); burned_geoms.append(base)
    elif vage_h>=12:
        # fresh fire, no VIIRS, no mapped perimeter: ANCHOR on the reported
        # origin so the perimeter stays on the fire and can't drift to an
        # off-location camera-bearing crossing. Origin is the one reliable point.
        oseed=Point(lon,lat).buffer(500/111320.0)
        G.add(FF.Observation('origin','burned',geom=oseed,sigma_m=300,weight=0.8,note='reported-origin anchor')); burned_geoms.append(oseed)
        print('  + ORIGIN anchor (fresh fire, no VIIRS/mapped)',flush=True)
    # camera fire-edge wedges as constraints (range-band near fire)
    wedge_polys=[]
    for cam,lo_,hi_ in wedges:
        r0=max(cam['d']-8,1); r1=cam['d']+8
        ring=[_fwd(cam['lat'],cam['lon'],b,r1) for b in np.linspace(lo_,hi_,12)]+[_fwd(cam['lat'],cam['lon'],b,r0) for b in np.linspace(hi_,lo_,12)]
        wed=Polygon([(p[1],p[0]) for p in ring]).intersection(Point(lon,lat).buffer(0.06))
        if not wed.is_empty and wed.area>0:
            wedge_polys.append(wed)
            G.add(FF.Observation('camera','burned',geom=wed,sigma_m=150,weight=0.6,note='cam wedge')); burned_geoms.append(wed)
    # AIRCRAFT: use the TRACKER's recent DROP PATHS (retardant/water), not the
    # live position. Drop runs trace the worked/defended perimeter. The tracker
    # (aircraft_tracker.py) runs continuously; here we read its recent paths and
    # backfill from OpenSky history so a cold start still has data.
    try:
        from shapely.geometry import LineString
        import aircraft_tracker as ATR
        t0=time.time()
        try: ATR.poll_once(name, lat, lon)   # log one fresh sample too
        except Exception as e:
            print('  ! AIRCRAFT poll_once failed (using stored history only):',repr(e)[:100],flush=True)
            SH.record(out_dir,'aircraft_poll','unavailable',reason=repr(e)[:200],latency_s=time.time()-t0)
        paths=ATR.recent_paths(name, hours=2, seed_from_history=(lat,lon))
        drop_paths=[p for p in paths if p['drop'] and len(p['coords'])>=2]
        layers['_aircraft_paths']=paths
        if drop_paths:
            dg=unary_union([LineString(p['coords']).buffer(400/111320.0) for p in drop_paths])
            _ao=FF.Observation('aircraft','burned',geom=dg,sigma_m=500,weight=0.55,note=f'{len(drop_paths)} recent drop runs')
            G.add(_ao); burned_geoms.append(dg); layers['aircraft']=_ao
            print('  + AIRCRAFT %d recent DROP paths (worked edge)'%len(drop_paths),flush=True)
            SH.record(out_dir,'aircraft',           'ok',latency_s=time.time()-t0,record_count=len(drop_paths))
        else:
            print('  ~ AIRCRAFT %d paths, no drop runs yet'%len(paths),flush=True)
            SH.record(out_dir,'aircraft','no_data',latency_s=time.time()-t0)
    except Exception as e:
        print('  ! AIRCRAFT',repr(e)[:60])
        SH.record(out_dir,'aircraft','unavailable',reason=repr(e)[:200])
    # AIRCRAFT SHUTTLE: a CONFIRMED repeating water/base<->fire round trip
    # (aircraft_tracker.detect_shuttles) is much higher-precision than any
    # single drop path -- see fire_fusion.obs_aircraft_shuttle.
    t0=time.time()
    try:
        sh_obs=FF.obs_aircraft_shuttle(name, hours=6)
        dt=time.time()-t0
        if isinstance(sh_obs,FF.Unavailable):
            print('  !','AIRCRAFT-SHUTTLE','UNAVAILABLE:',sh_obs.reason[:120],flush=True)
            SH.record(out_dir,'aircraft_shuttle','unavailable',reason=sh_obs.reason,latency_s=dt)
        elif sh_obs:
            G.add(sh_obs); burned_geoms.append(sh_obs.geom); layers['aircraft_shuttle']=sh_obs
            print('  +','AIRCRAFT-SHUTTLE',sh_obs.note,flush=True)
            SH.record(out_dir,'aircraft_shuttle','ok',reason=sh_obs.note,latency_s=dt,record_count=1)
        else:
            print('  ~','AIRCRAFT-SHUTTLE','none (no confirmed multi-trip shuttle)',flush=True)
            SH.record(out_dir,'aircraft_shuttle','no_data',latency_s=dt)
    except Exception as e:
        print('  ! AIRCRAFT-SHUTTLE',repr(e)[:70])
        SH.record(out_dir,'aircraft_shuttle','unavailable',reason=repr(e)[:200],latency_s=time.time()-t0)
    # RADIO: wired hook -- contributes spread-direction/callout constraints when a
    # live scanner audio feed is supplied (fpm/fire_radio.py). No free historical
    # audio, so it's None here unless an audio path is passed in.
    print('  ~ RADIO none (needs live Broadcastify audio feed; hook in fire_radio.py)',flush=True)

    # 5. PHYSICS shaping: anisotropic diffusion of the DATA belief along Rothermel dir
    P=G.prob()
    try:
        fuel=int(np.median(DI.get_fbfm40(bbox,G.lats,G.lons)))
    except Exception: fuel=165
    wj=_get(f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=wind_speed_10m,wind_direction_10m,relative_humidity_2m,temperature_2m&wind_speed_unit=mph&temperature_unit=fahrenheit')
    if wj is None:
        # _get() swallows the real failure reason; the fallback below is
        # fabricated weather (7mph/270deg/70F/30%RH), not an observation, so
        # this must be visible rather than indistinguishable from real data.
        print('  ! open-meteo (Rothermel wind/RH) UNAVAILABLE -- using fallback defaults',flush=True)
        SH.record(out_dir,'open_meteo_wind','unavailable',reason='open-meteo forecast fetch failed')
    else:
        SH.record(out_dir,'open_meteo_wind','ok',record_count=1)
    cur=wj['current'] if wj else {}
    wspd=cur.get('wind_speed_10m',7); wto=(cur.get('wind_direction_10m',270)+180)%360
    fm=float(DI.dead_fuel_moisture_1h(cur.get('temperature_2m',70),cur.get('relative_humidity_2m',30)))
    rr=rm.rothermel_point(fuel,fm,wspd,wto,6,200) or {'lb':1.5,'intensity_kw_m':1000}
    lb=rr['lb']
    # elongate belief along wto by lb (physics), keep data-driven detail
    Pr=ndrot(P,-wto,reshape=False,order=1,mode='nearest')
    Pr=gaussian_filter(Pr,(1.0,1.0+1.2*min(lb-1,3)))     # more elongation along-wind for higher lb
    Pe=ndrot(Pr,wto,reshape=False,order=1,mode='nearest'); Pe=gaussian_filter(Pe,0.7)
    Pe=Pe/max(Pe.max(),1e-6)*max(P.max(),1e-6)
    # ---- PERIMETER DECISION -------------------------------------------------
    # The design (user-directed): GOES is the 5-min growth DRIVER, Rothermel is the
    # SHAPE, the mapped perimeter is the ANCHOR, VIIRS is only CONFIRMATION. We do
    # NOT contour a VIIRS belief blob when a mapped base exists.
    def ac(g): return transform(lambda x,y,z=None:((x-lon)*mx,(y-lat)*my),g).area/4046.86 if g else 0
    from shapely.geometry import Point as _Pt
    from shapely import affinity
    _origin=_Pt(lon,lat)
    def _pick(g):   # component NEAREST THE ORIGIN (the reported fire), not the largest
        if g is None or g.is_empty: return None
        parts=[g] if g.geom_type=='Polygon' else list(g.geoms)
        return min(parts,key=lambda p:p.distance(_origin)) if parts else None
    # belief-contour perimeter -- kept ONLY as the no-base / no-fresh fallback
    pk=float(Pe.max()); lvl=0.5 if pk>=0.55 else max(0.5*pk,0.12)
    cs=plt.contour(G.LON,G.LAT,Pe,levels=[lvl]);plt.close()
    polys=[Polygon(s) for s in cs.allsegs[0] if len(s)>=4]; polys=[p for p in polys if p.is_valid and p.area>0]
    perim_belief=_pick(unary_union(polys)) if polys else None
    perim=None; base_driven=False

    # PRIMARY PATH -- a mapped perimeter exists: anchor on it (at t_map), then GROW
    # it using GOES hotspots that lit SINCE t_map (5-min cadence = the driver),
    # shaped by Rothermel spread physics. GOES sets where/how far it grew; VIIRS
    # only confirms. This replaces the old VIIRS-driven belief contour.
    if base is not None:
        now=datetime.now(timezone.utc)
        dt_h=min(max((now-t_map).total_seconds()/3600.0,0.0),24.0) if t_map is not None else 3.0
        # 1. GOES crossed cells detected AFTER the map time = the growth signal
        gg=None; gg_src='since'
        if t_map is not None:
            try:
                og=FF.obs_goes(bbox,ABI,since_dt=t_map,weight=0.4)
                if og and og.geom is not None and not og.geom.is_empty: gg=og.geom
            except Exception as e: print('  ! GOES-growth',repr(e)[:60],flush=True)
        # fallback: if the precise since-query found nothing (cold cache / tiny
        # interval) but the belief-grid GOES layer (last 10h crossed cells, already
        # fetched) exists, use THAT -- still GOES-driven, just a wider window. The
        # reach intersection below clips it to physics-feasible growth. Prevents a
        # pure-Rothermel lobe from taking over whenever GOES has any cells.
        if (gg is None or gg.is_empty) and 'goes' in layers and layers['goes'].geom is not None and not layers['goes'].geom.is_empty:
            gg=layers['goes'].geom; gg_src='10h-layer'
        # 2. Rothermel head advance over the elapsed interval, DUTY-CYCLED (free
        #    ROS is an instantaneous peak; observed front advance ~0.22x once
        #    lulls/duty cycle are accounted for -- see ml_correction gotcha).
        ros=rr.get('ros_head_m_min',6.0)
        adv_km=min(max(0.15, ros*dt_h*60*0.22/1000.0), 8.0)
        # GOES admission radius: at least ONE GOES cell (~2km) so dual-confirmed
        # adjacent growth is never clipped by physics (GOES is the driver); the
        # Rothermel lobe below still only PROJECTS as far as adv_km allows.
        reach=base.buffer(max(adv_km,2.0)/111.0)            # GOES-reachable growth this interval
        grown=[base]; goes_present=False
        # 3. GOES growth that is physically REACHABLE from the mapped edge
        if gg is not None:
            reachable=gg.intersection(reach)
            if not reachable.is_empty and reachable.area>0:
                grown.append(reachable); goes_present=True
                print('  GOES growth [%s]: +%.0f ac reachable (%.1fh, adv cap %.1fkm)'%(gg_src,ac(reachable.difference(base)),dt_h,adv_km),flush=True)
            else:
                print('  GOES growth since map: none reachable yet (%.1fh)'%dt_h,flush=True)
        else:
            print('  GOES growth since map: no GOES cells since %s'%(t_map.strftime('%H:%MZ') if t_map else 'n/a'),flush=True)
        # 4. Rothermel directional HEAD LOBE off the base's downwind edge -- a
        #    realistic tapered ellipse (elongated by lb along wto). When GOES growth
        #    is present it already sets the EXTENT, so the lobe is only a MODEST
        #    taper (<=1.5km) to smooth the coarse 2km cells + extend the head a
        #    little. Only when GOES is ABSENT does the lobe carry the growth (still
        #    capped -- free Rothermel overshoots on long intervals; flagged).
        lobe_adv=min(adv_km,1.5) if goes_present else min(adv_km,4.0)
        if not goes_present and adv_km>4.0:
            print('  ! no GOES growth: pure-Rothermel head capped 4km (low confidence, %.1fh)'%dt_h,flush=True)
        r_base_km=(ac(base)*4046.86/math.pi)**0.5/1000.0
        edge=_fwd(base.centroid.y,base.centroid.x,wto,r_base_km)      # downwind edge of base
        semilong=lobe_adv*max(lb,1.0)/2 + r_base_km*0.3
        semishort=max(lobe_adv/2, semilong/max(lb,1.2))
        lobe=affinity.scale(_Pt(edge[1],edge[0]).buffer(1.0,resolution=48),semishort/111.0,semilong/111.0)
        lobe=affinity.rotate(lobe,-wto,origin=(edge[1],edge[0]))      # long axis downwind
        lobe=affinity.translate(lobe,(semilong*0.5/111.0)*math.sin(math.radians(wto)),(semilong*0.5/111.0)*math.cos(math.radians(wto)))
        grown.append(lobe.intersection(reach))
        perim=unary_union([g for g in grown if g is not None and not g.is_empty])
        sm=max(lobe_adv/111.0*0.15, 0.002)
        perim=perim.buffer(sm).buffer(-sm*0.8)                        # smooth the union of coarse cells
        # VIIRS CONFIRMATION ONLY -- bounded by the ROTHERMEL physics advance
        # (adv_km, NOT the 2km GOES-cell floor), so VIIRS can nudge the head out
        # to where physics allows but can NEVER balloon the base on its own. When
        # the interval is short adv_km->0.15km, so VIIRS moves the edge <=150m.
        if 'viirs' in layers and layers['viirs'].geom is not None:
            vg=layers['viirs'].geom.intersection(base.buffer(adv_km/111.0))
            if not vg.is_empty and vg.area>0:
                perim=unary_union([perim,vg]).buffer(sm*0.3).buffer(-sm*0.2)
        perim=_pick(perim); base_driven=True
        print('  PRIMARY base+GOES+Rothermel: %.0f ac (base %.0f -> +%.0f growth)'%(ac(perim),base_ac,ac(perim)-base_ac),flush=True)

    # FRESH FIRE (no mapped base, no fresh VIIRS): Rothermel ellipse FROM ORIGIN.
    # Gated on base_status=='not_found', not just base is None: if WFIGS/FIRIS
    # were unreachable this cycle, a real perimeter may exist that we simply
    # failed to fetch, and anchoring on the origin alone would silently
    # discard it (rebuild-spec Ground Rule #1). An unreachable fetch degrades
    # to the belief-contour fallback below instead of asserting "fresh fire".
    elif base is None and vage_h>=12 and base_status!='unavailable':
        near=[g for g in burned_geoms if g.distance(_origin)<3000/111320.0 and g.geom_type!='Point']
        near_ac=ac(unary_union(near)) if near else 0
        tgt=max(300.0,near_ac)
        semilong=math.sqrt(tgt*4046.86/math.pi*rr.get('lb',1.4))/1000.0
        semishort=semilong/rr.get('lb',1.4)
        ell=affinity.scale(_origin.buffer(1.0,resolution=48),semishort/111.0,semilong/111.0)
        ell=affinity.rotate(ell,-wto,origin=(lon,lat))
        ell=affinity.translate(ell,(semilong*0.4/111.0)*math.sin(math.radians(wto)),(semilong*0.4/111.0)*math.cos(math.radians(wto)))
        perim=(unary_union([ell]+near) if near else ell).buffer(0.02).buffer(-0.01)
        print('  FRESH-FIRE origin-grown ellipse -> %.0f ac (downwind %.0f, lb %.2f)'%(ac(perim),wto,rr.get('lb',1.4)),flush=True)

    # FALLBACK (base None but recent VIIRS present): belief contour, clipped to
    # detection support + near-origin VIIRS. GOES/VIIRS both feed the belief here.
    else:
        perim=perim_belief
        if perim is not None and burned_geoms:
            grow_km=max(0.6,(rr.get('ros_head_m_min',6)/1000.0)*60*0.15)
            support=unary_union([g.buffer(grow_km/111.0) for g in burned_geoms])
            clipped=perim.intersection(support)
            if not clipped.is_empty and clipped.area>0: perim=_pick(clipped)
            must=[]
            if 'viirs' in layers and layers['viirs'].geom is not None:
                vg=layers['viirs'].geom.intersection(_origin.buffer(0.06))
                if not vg.is_empty and vg.area>0: must.append(vg)
            if perim is not None and must:
                perim=_pick(unary_union([perim]+must).buffer(grow_km/111.0*0.4).buffer(-grow_km/111.0*0.3))
    # 6. ember spotting downwind of leading edge
    spot_m=rm.spotting_distance_m(rr.get('intensity_kw_m',1000),wspd); spots=[]
    if perim is not None:
        cen=perim.centroid; lead=_fwd(cen.y,cen.x,wto,ac(perim)**0.5*0.05+0.5)  # rough leading point
        rng=np.random.default_rng(1)
        for _ in range(5):
            p=_fwd(lead[0],lead[1],wto+rng.uniform(-25,25),spot_m/1000*rng.uniform(0.4,1.2))
            spots.append(Point(p[1],p[0]).buffer(rng.uniform(0.0004,0.0009)))
    spot_u=unary_union(spots) if spots else None
    print(f'PERIMETER {ac(perim):.0f} ac | lb={lb:.2f} spotting {spot_m:.0f}m | mapped base {base_ac:.0f} ac',flush=True)
    # 7. map -- SATELLITE BASEMAP + every data source overlaid
    fig,ax=plt.subplots(figsize=(11,10)); asp=1/math.cos(math.radians(lat))
    # size the view to the fire but at least the perimeter extent
    if perim is not None:
        b=perim.bounds; pad=max(0.02,(b[2]-b[0]),(b[3]-b[1]))*0.6
        vx=(min(b[0],lon)-pad,max(b[2],lon)+pad); vy=(min(b[1],lat)-pad,max(b[3],lat)+pad)
    else: vx=(lon-0.05,lon+0.05); vy=(lat-0.045,lat+0.05)
    vbbox=(vx[0],vy[0],vx[1],vy[1])
    # satellite imagery background
    try:
        z=15 if (vx[1]-vx[0])<0.06 else (14 if (vx[1]-vx[0])<0.12 else 13)
        img,ext=satellite_basemap(vbbox,zoom=z)
        if img is not None: ax.imshow(img,extent=ext,origin='upper',aspect=asp,zorder=0)
    except Exception as e: print('  basemap fail',repr(e)[:60])
    def draw_poly(g,**kw):
        if g is None or g.is_empty: return
        first=True
        for gg in ([g] if g.geom_type=='Polygon' else list(g.geoms)):
            if gg.is_empty: continue
            xs,ys=gg.exterior.xy; ax.plot(xs,ys,label=kw.pop('label',None) if first else None,**{k:v for k,v in kw.items() if k!='label' or first}); first=False
    # --- overlays: every source ---
    if 'goes' in layers: draw_poly(layers['goes'].geom,color='#ff6666',lw=1.2,alpha=0.9,label='GOES crossed cells')
    if 'sentinel2_swir' in layers: draw_poly(layers['sentinel2_swir'].geom,color='#ff00ff',lw=1.6,label='S2 SWIR fire (20m)')
    v=None
    try:
        import firms_fixed as F2; v=F2.fetch(bbox,date.today()-timedelta(days=2),date.today())
    except Exception: pass
    if v: ax.scatter([d['lon'] for d in v],[d['lat'] for d in v],s=16,c='red',edgecolor='k',linewidth=0.2,zorder=5,label='latest VIIRS (%d)'%len(v))
    if 'goes_c07' in layers: draw_poly(layers['goes_c07'].geom,color='#ffff00',lw=1.0,alpha=0.7,label='GOES C07 hot (5min)')
    # aircraft DROP PATHS (retardant/water) with timestamps -> the worked edge
    if layers.get('_aircraft_paths'):
        import time as _t
        dl=True; tl=True
        for pp in layers['_aircraft_paths']:
            xs=[c[0] for c in pp['coords']]; ys=[c[1] for c in pp['coords']]
            if pp['drop']:
                ax.plot(xs,ys,'-',color='#ff1493',lw=3,zorder=7,label='aircraft DROP path' if dl else None); dl=False
                mins=(_t.time()-pp['t1'])/60
                ax.text(xs[-1],ys[-1],'%s %.0fm ago'%(pp.get('kind','drop')[:4],mins),fontsize=6,color='white',zorder=8)
            else:
                ax.plot(xs,ys,':',color='#dddddd',lw=1,alpha=0.6,zorder=4,label='aircraft transit' if tl else None); tl=False
    # camera wedges as LOCAL regions (not long sightlines) -- the angular bound near the fire
    if wedge_polys:
        first=True
        for wp in wedge_polys:
            for gg in ([wp] if wp.geom_type=='Polygon' else list(wp.geoms)):
                if gg.is_empty: continue
                xs,ys=gg.exterior.xy; ax.plot(xs,ys,'-',color='orange',lw=1.2,alpha=0.7,zorder=2,label='camera fire-edge bound' if first else None); first=False
    if 'goes_adp_smoke' in layers:   # smoke plume DIRECTION arrow
        br=math.radians(layers['goes_adp_smoke'].bearing)
        ax.annotate('',xy=(lon+0.02*math.sin(br),lat+0.02*math.cos(br)),xytext=(lon,lat),arrowprops=dict(fc='#bbbbbb',ec='k',width=1.5,headwidth=8),zorder=7)
        ax.plot([],[],color='#bbbbbb',lw=2,label='GOES ADP smoke dir')
    if base is not None:
        lbl='PREVIOUS mapped %.0f ac'%base_ac
        for g in ([base] if base.geom_type=='Polygon' else base.geoms):
            xs,ys=g.exterior.xy; ax.plot(xs,ys,'--',color='#00e5ff',lw=2,zorder=6,label=lbl); lbl=None
    if perim is not None:
        for g in ([perim] if perim.geom_type=='Polygon' else perim.geoms):
            xs,ys=g.exterior.xy; ax.plot(xs,ys,'-',color='#39ff14',lw=3.2,zorder=8,label='PREDICTED %.0f ac'%ac(perim)); ax.fill(xs,ys,color='#39ff14',alpha=0.10,zorder=3)
    if spot_u is not None:
        lbl='ember spots (~%.0fm)'%spot_m
        for g in ([spot_u] if spot_u.geom_type=='Polygon' else spot_u.geoms):
            xs,ys=g.exterior.xy; ax.fill(xs,ys,color='#ff4500',alpha=0.7,zorder=9,label=lbl); lbl=None
    ax.annotate('',xy=(lon+0.012*math.sin(math.radians(wto)),lat+0.012*math.cos(math.radians(wto))),xytext=(lon,lat),arrowprops=dict(fc='cyan',ec='k',width=2.2,headwidth=10),zorder=10)
    ax.plot(lon,lat,'*',color='yellow',ms=15,markeredgecolor='k',zorder=10,label='origin')
    ax.set_xlim(*vx); ax.set_ylim(*vy)
    ax.set_title('%s fire — fusion_v3 on satellite imagery | %.0f ac\nVIIRS(%.0fh)+GOES(FDCC/C07/ADP)+S2-SWIR+cameras+aircraft | Rothermel lb=%.2f, spotting %.0fm'%(name,ac(perim),vage_h,lb,spot_m))
    ax.set_xlabel('lon'); ax.set_ylabel('lat'); ax.set_aspect(asp); ax.legend(loc='lower left',fontsize=8,framealpha=0.92)
    perim_png=os.path.join(out_dir,f'{name}_perimeter.png'); plt.savefig(perim_png,dpi=120,bbox_inches='tight')
    if perim is not None:
        json.dump({'type':'FeatureCollection','features':[{'type':'Feature','properties':{'fire':name,'acres':round(ac(perim)),'method':'fusion_v3'},'geometry':mapping(perim)}]},open(os.path.join(out_dir,f'{name}_perimeter.geojson'),'w'))
    health_html=SH.write_html(out_dir,fire=name)
    # Phase 9 item 3: snapshot every input used this cycle (reproducibility).
    # Phase 9 item 4: structured cycle metrics + physics-feasibility alert.
    try:
        import cycle_provenance as CP
        scalars={'lat':lat,'lon':lon,'bbox':list(bbox),
                 'wind_to_bearing':wto,'wind_speed_mph':wspd,'fuel_model':fuel,
                 'fuel_moisture_1h':fm,'rothermel_lb':lb,'base_acres':base_ac,
                 'viirs_overpass_age_h':vage_h,'t_map':str(t_map) if t_map is not None else None,
                 'mapped_base_status':base_status,'spotting_m':spot_m}
        CP.snapshot_inputs(out_dir,name,layers,scalars)
        CP.record_cycle_metrics(out_dir,name,pred_acres=ac(perim),base_acres=base_ac,
                                runtime_s=time.time()-_t_start,
                                source_health_summary=SH.summarize(out_dir),
                                extra={'perim_present':perim is not None,
                                       'n_sources_used':len([k for k in layers if not k.startswith('_')])})
    except Exception as e:
        print('  ! cycle_provenance failed:',repr(e)[:100],flush=True)
    return dict(perimeter_png=perim_png,cameras_jpg=cam_jpg,acres=ac(perim),cam_rays=len(rays),cam_wedges=len(wedges),source_health_html=health_html)

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('name'); ap.add_argument('lat',type=float); ap.add_argument('lon',type=float); ap.add_argument('--out',default='.')
    a=ap.parse_args(); print(json.dumps(run(a.name,a.lat,a.lon,a.out),indent=2))
