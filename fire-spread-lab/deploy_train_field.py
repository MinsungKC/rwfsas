"""Train a DEPLOYABLE learned-field v3 model on ALL frozen cohort fires (no
holdout) so it can run live. Eval used LOFO; deployment trains on everything.
Live fires (Dome/Floriston, Sept 2026) are not in this older cohort, so live
inference is genuinely out-of-sample.

Persists: _frozen/deployable_field_v3.pkl = {model, retained_negative_fraction,
feature_names, classifier_kwargs, n_steps, trained_utc, git_note}.
"""
import sys, os, json, pickle, importlib.util
from datetime import datetime, timezone
import numpy as np
# model code lives in the NON-claude lab tree; import it by explicit path so the
# local `models/` package here does not shadow it.
_LF_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../fire-spread-lab/models/learned_field_v3.py'))
_spec = importlib.util.spec_from_file_location('learned_field_v3', _LF_PATH)
LF = importlib.util.module_from_spec(_spec)
sys.modules['learned_field_v3'] = LF   # register before exec so @dataclass can introspect (py3.14)
_spec.loader.exec_module(LF)

STEPS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_frozen', 'steps.jsonl')
OUT   = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_frozen', 'deployable_field_v3.pkl')


def main():
    dem = LF.ZeroDem()   # terrain-free first cut; matches the v2 feature regime (exp_hi gIoU ~0.43)
    # STREAM: build one block, subsample immediately, drop the full grid (holding
    # all 564 grids at once is ~3GB and OOMs). Keep only the small selected arrays.
    from sklearn.ensemble import HistGradientBoostingClassifier
    rng = np.random.default_rng(0)
    xs, ys = [], []
    all_neg = kept_neg = 0
    nbad = nblk = 0
    for i, ln in enumerate(open(STEPS, encoding='utf-8')):
        try:
            step = json.loads(ln)
            tb = LF.make_training_block(step, step['real_wkb'], step['end'], dem=dem)
            if tb is None or len(np.unique(tb.labels)) < 1:
                continue
            pos = np.flatnonzero(tb.labels == 1); neg = np.flatnonzero(tb.labels == 0)
            all_neg += len(neg)
            count = min(len(neg), max(int(len(pos) * LF.NEGATIVE_RATIO), 200))
            if count < len(neg): neg = rng.choice(neg, count, replace=False)
            kept_neg += len(neg)
            sel = np.concatenate((pos, neg))
            xs.append(np.asarray(tb.grid.features[sel], dtype=np.float32))
            ys.append(np.asarray(tb.labels[sel]))
            nblk += 1
        except Exception:
            nbad += 1
        if (i + 1) % 100 == 0:
            print(f'  ...{i+1} steps ({nblk} used, {nbad} skipped)', flush=True)
    print(f'built {nblk} training blocks ({nbad} skipped)', flush=True)
    x, y = np.vstack(xs), np.concatenate(ys)
    del xs, ys
    print(f'train matrix {x.shape}, pos rate {y.mean():.3f}', flush=True)
    if len(np.unique(y)) != 2:
        raise SystemExit('need both classes')
    model = HistGradientBoostingClassifier(**LF.CLASSIFIER_KWARGS).fit(x, y)
    rnf = kept_neg / max(all_neg, 1)
    with open(OUT, 'wb') as fh:
        pickle.dump({'model': model, 'retained_negative_fraction': rnf,
                     'feature_names': LF.FEATURE_NAMES, 'classifier_kwargs': LF.CLASSIFIER_KWARGS,
                     'n_steps': nblk, 'dem': 'ZeroDem',
                     'trained_utc': datetime.now(timezone.utc).isoformat()}, fh)
    print(f'SAVED {OUT}  rnf={rnf:.4f}  n_steps={nblk}  train_rows={len(y)}  pos_rate={y.mean():.3f}', flush=True)


if __name__ == '__main__':
    main()
