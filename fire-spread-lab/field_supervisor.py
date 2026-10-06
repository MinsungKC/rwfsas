"""Supervisor for the live field perimeter product (rebuild spec Phase 9).

Fixes the operational-hygiene gaps:
  * ONE writer per output base -- lockfile + PID guard; a second instance refuses
    to start (stops the duplicate-process races on shared dirs / ABI cache).
  * Each 5-min cycle runs deploy_live_field.py in a FRESH SUBPROCESS -- no
    importlib.reload (which silently mixes old/new class defs); clean state every
    cycle; code edits are picked up automatically on the next cycle.
  * run_field snapshots its exact input bundle (<name>_inputs.json) each cycle, so
    ANY perimeter is reproducible offline with `deploy_live_field.py --replay`.
  * Structured per-cycle log + latest_* copies + history.jsonl; flags a cycle
    whose growth exceeds a physics-feasible bound (a bug detector).

Usage:
  python field_supervisor.py <outbase> <period_s> <name lat lon k> [<name lat lon k> ...]
"""
import os, sys, json, time, socket, shutil, subprocess
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
DLF = os.path.join(HERE, 'deploy_live_field.py')
MAX_GROWTH_RATIO = 8.0        # pred/base above this in one cycle -> flag (feasibility bug detector)


def _alive(pid):
    try:
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, 0, int(pid))
        if not h:
            return False
        ctypes.windll.kernel32.CloseHandle(h); return True
    except Exception:
        try:
            os.kill(int(pid), 0); return True
        except Exception:
            return False


def acquire_lock(path):
    if os.path.exists(path):
        try:
            info = json.load(open(path))
            if info.get('pid') and _alive(info['pid']):
                print(f'REFUSING: supervisor pid {info["pid"]} already holds {path}', flush=True)
                sys.exit(1)
            print(f'stale lock (pid {info.get("pid")} dead) -> reclaiming', flush=True)
        except Exception:
            pass
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({'pid': os.getpid(), 'host': socket.gethostname(),
               'started': datetime.now(timezone.utc).isoformat()}, open(path, 'w'))


def main():
    outbase = sys.argv[1]; period = float(sys.argv[2]); rest = sys.argv[3:]
    fires = [tuple(rest[i:i+4]) for i in range(0, len(rest), 4)]
    lock = os.path.join(outbase, 'supervisor.lock'); acquire_lock(lock)
    print(f'field supervisor pid {os.getpid()}: {len(fires)} fires every {period:.0f}s -> {outbase}', flush=True)
    try:
        while True:
            t0 = time.time()
            for name, lat, lon, k in fires:
                ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                odir = os.path.join(outbase, name, ts); os.makedirs(odir, exist_ok=True)
                c0 = time.time()
                try:
                    r = subprocess.run([sys.executable, '-u', DLF, name, lat, lon, odir, k],
                                       capture_output=True, text=True, timeout=285)
                except subprocess.TimeoutExpired:
                    print(f'{ts} {name}: TIMEOUT (>285s)', flush=True); continue
                fj = os.path.join(odir, f'{name}_field.json')
                if os.path.exists(fj):
                    rec = json.load(open(fj)); rec['t'] = ts; rec['cycle_s'] = round(time.time()-c0, 1)
                    ratio = rec['pred_acres']/max(rec['base_acres'], 1)
                    flag = '  !FEASIBILITY' if ratio > MAX_GROWTH_RATIO else ''
                    with open(os.path.join(outbase, name, 'history.jsonl'), 'a') as fh:
                        fh.write(json.dumps(rec)+'\n')
                    for suf in ('_field.png', '_field.geojson', '_field.json', '_inputs.json'):
                        s = os.path.join(odir, name+suf)
                        if os.path.exists(s): shutil.copy2(s, os.path.join(outbase, name, 'latest'+suf))
                    print(f'{ts} {name}: {rec["pred_acres"]} ac [{rec["lo_acres"]},{rec["hi_acres"]}] '
                          f'x{ratio:.1f} ({rec["cycle_s"]}s){flag}', flush=True)
                else:
                    tail = (r.stderr or r.stdout or '')[-240:]
                    print(f'{ts} {name}: FAILED rc={r.returncode} {tail}', flush=True)
            dt = time.time()-t0
            if dt < period:
                time.sleep(period-dt)
    finally:
        try:
            os.remove(lock)
        except Exception:
            pass


if __name__ == '__main__':
    main()
