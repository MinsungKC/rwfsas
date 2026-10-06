"""5-minute loop for the deployable learned-field v3 live perimeter (the lab's
validated best model). One process per fire. Writes a timestamped dir + latest_*
copies + history.jsonl, same convention as the fusion loop."""
import sys, os, time, json, importlib
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../fpm')))   # satellite_basemap reuse in the renderer
import deploy_live_field as DLF


def main():
    name, lat, lon, outbase = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    period = float(sys.argv[5]) if len(sys.argv) > 5 else 300.0
    k = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
    hist = os.path.join(outbase, name, 'history.jsonl')
    os.makedirs(os.path.dirname(hist), exist_ok=True)
    print(f'field loop: {name} @ ({lat},{lon}) every {period:.0f}s k={k} -> {outbase}', flush=True)
    import shutil
    while True:
        t0 = time.time(); ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        odir = os.path.join(outbase, name, ts)
        try:
            importlib.reload(DLF)                 # pick up model/runner edits each cycle
            r = DLF.run_field(name, lat, lon, odir, k=k)
            if r is not None:
                with open(hist, 'a', encoding='utf-8') as fh: fh.write(json.dumps(r) + '\n')
                for suf in ('_field.png', '_field.geojson', '_field.json'):
                    src = os.path.join(odir, name + suf)
                    if os.path.exists(src): shutil.copy2(src, os.path.join(outbase, name, 'latest' + suf))
                print(f'{ts} {name}: {r["pred_acres"]} ac [{r["lo_acres"]},{r["hi_acres"]}] (cycle {time.time()-t0:.0f}s)', flush=True)
        except Exception as e:
            print(f'{ts} {name}: ERROR {repr(e)[:120]}', flush=True)
        dt = time.time() - t0
        if dt < period: time.sleep(period - dt)


if __name__ == '__main__':
    main()
