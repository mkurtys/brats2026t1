"""
Local web dashboard to review BraTS validation NIfTIs with segmentation overlays.

No extra deps: stdlib http.server + nibabel + matplotlib (already installed).
Slices are rendered server-side to PNG; the browser is a thin scrubber.

Usage:
    python -m mbrats.review_dashboard              # serve on http://127.0.0.1:8757
    python -m mbrats.review_dashboard --port 9000
    python -m mbrats.review_dashboard --images /some/dir --seg predictions/foo

Image roots and overlay (seg) sources are auto-discovered; the flags above just
add extra ones. Open the URL, pick a case on the left, flip modality/overlay/axis.

Label mapping (from mbrats/met_labels.py, authoritative):
    1=NETC  2=SNFH  3=ET  4=RC
"""

import argparse
import concurrent.futures
import csv
import io
import json
import re
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import matplotlib
matplotlib.use('Agg')
from matplotlib import image as mpimg
import nibabel as nib
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent

LABEL_NAMES  = {1: 'NETC', 2: 'SNFH', 3: 'ET', 4: 'RC'}
LABEL_COLORS = {1: [1.0, 0.4, 0.0], 2: [0.2, 0.6, 1.0], 3: [1.0, 0.9, 0.0], 4: [0.9, 0.2, 0.9]}
DIFF_COLORS  = {'tp': [0.1, 0.9, 0.2], 'fp': [1.0, 0.15, 0.15], 'fn': [0.2, 0.5, 1.0]}  # agree / over / miss
AXIS_NAMES   = {0: 'Sagittal', 1: 'Coronal', 2: 'Axial'}

# Validation images: per-case dir, files like BraTS-MET-00833-000-t1c.nii.gz
VAL_DIR = Path('/media/mkurtys/T7/datasets/brats2026/Validation')
# Training images (nnU-Net raw): flat, BraTS-MET-XXXX_0001.nii.gz
IMAGESTR = PROJECT / 'nnunet_raw/Dataset001_BraTSMETS/imagesTr'
LABELSTR = PROJECT / 'nnunet_raw/Dataset001_BraTSMETS/labelsTr'
PRED_ROOT = PROJECT / 'predictions'
RESULTS_DIR = PROJECT / 'results'                       # *_cv_eval.csv per-case DSC
NNUNET_RESULTS = PROJECT / 'nnunet_results/Dataset001_BraTSMETS'
METRICS = ['mean', 'et', 'tc', 'wt', 'rc']              # 'mean' = nanmean(et,tc,wt)

MOD_SUFFIX = {'t1n': '_0000', 't1c': '_0001', 't2w': '_0002', 't2f': '_0003'}
MODALITIES = ['t1c', 't1n', 't2w', 't2f']


# --------------------------------------------------------------------------- #
# Source model: an image root knows how to find a modality file for a case;
# a seg source knows how to find a single-file segmentation for a case.
# --------------------------------------------------------------------------- #
class ImageRoot:
    def __init__(self, name, path, style):
        self.name = name
        self.path = Path(path)
        self.style = style  # 'val' (per-case dir, -mod suffix) or 'nnunet' (_000x)

    def case_path(self, case, modality):
        if self.style == 'val':
            return self.path / case / f'{case}-{modality}.nii.gz'
        return self.path / f'{case}{MOD_SUFFIX[modality]}.nii.gz'

    def list_cases(self):
        cases = set()
        if not self.path.exists():
            return []
        if self.style == 'val':
            for p in self.path.iterdir():
                if p.is_dir() and p.name.startswith('BraTS') and not p.name.startswith('._'):
                    cases.add(p.name)
        else:
            for p in self.path.glob('*_0000.nii.gz'):
                if not p.name.startswith('._'):
                    cases.add(p.name[:-len('_0000.nii.gz')])
        return sorted(cases)


class SegSource:
    def __init__(self, name, path):
        self.name = name
        self.path = Path(path)

    def seg_path(self, case):
        return self.path / f'{case}.nii.gz'


IMAGE_ROOTS = OrderedDict()
SEG_SOURCES = OrderedDict()
SCORE_TABLES = OrderedDict()   # name -> {case: {'et','tc','wt','rc','mean'}}
SCORE_ROWS = OrderedDict()      # name -> {case: full csv row dict}


def _f(row, key):
    try:
        return float(row.get(key, ''))
    except (TypeError, ValueError):
        return None


def _load_score_table(csv_path):
    """Parse a *_cv_eval.csv into ({case: {metric: dsc}}, {case: full_row})."""
    table, rows = {}, {}
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            case = row.get('subject_id')
            if not case:
                continue
            vals = {m: _f(row, f'lesionwise_dsc_mean_{m}') for m in ('et', 'tc', 'wt', 'rc')}
            main = [vals[m] for m in ('et', 'tc', 'wt') if vals[m] is not None]
            vals['mean'] = sum(main) / len(main) if main else None
            table[case] = vals
            rows[case] = row
    return table, rows


def case_metrics(table_name, case):
    """Curated per-case metrics (DSC/NSD/HD95 + lesion TP/FP/FN/F1 per region)."""
    row = SCORE_ROWS.get(table_name, {}).get(case)
    if row is None:
        return None
    out = {}
    for r in ('et', 'tc', 'wt', 'rc'):
        out[r] = {
            'dsc': _f(row, f'lesionwise_dsc_mean_{r}'),
            'nsd': _f(row, f'lesionwise_nsd_mean_{r}'),
            'hd95': _f(row, f'lesionwise_hd95_mean_{r}'),
            'tp': _f(row, f'all_instance_tp_{r}'),
            'fp': _f(row, f'all_instance_fp_{r}'),
            'fn': _f(row, f'all_instance_fn_{r}'),
            'f1': _f(row, f'all_instance_f1_{r}'),
        }
    return out


def discover_score_tables():
    if not RESULTS_DIR.exists():
        return
    for p in sorted(RESULTS_DIR.glob('*_cv_eval.csv')):
        try:
            t, rows = _load_score_table(p)
        except Exception:  # noqa - skip malformed csv
            continue
        if t:
            name = p.name[:-len('_cv_eval.csv')]
            SCORE_TABLES[name] = t
            SCORE_ROWS[name] = rows


def discover_cv_preds():
    """nnU-Net fold_*/validation dirs = predictions on held-out train cases (have GT)."""
    if not NNUNET_RESULTS.exists():
        return
    for cfg in sorted(NNUNET_RESULTS.iterdir()):
        if not cfg.is_dir():
            continue
        for fold in sorted(cfg.glob('fold_*')):
            val = fold / 'validation'
            if val.is_dir() and next(val.glob('*.nii.gz'), None) is not None:
                trainer = cfg.name.split('__')[0].replace('nnUNetTrainer', '')
                conf = cfg.name.split('__')[-1]
                name = f'CV {trainer}/{conf}/{fold.name}'
                SEG_SOURCES[name] = SegSource(name, val)


def discover_sources(extra_images, extra_segs):
    if VAL_DIR.exists():
        IMAGE_ROOTS['Validation'] = ImageRoot('Validation', VAL_DIR, 'val')
    if IMAGESTR.exists():
        IMAGE_ROOTS['Train (imagesTr)'] = ImageRoot('Train (imagesTr)', IMAGESTR, 'nnunet')
    for d in extra_images:
        d = Path(d)
        style = 'nnunet' if list(d.glob('*_0000.nii.gz')) else 'val'
        IMAGE_ROOTS[d.name] = ImageRoot(d.name, d, style)

    if LABELSTR.exists():
        SEG_SOURCES['GT (labelsTr)'] = SegSource('GT (labelsTr)', LABELSTR)
    if PRED_ROOT.exists():
        for d in sorted(PRED_ROOT.iterdir()):
            if d.is_dir():
                SEG_SOURCES[d.name] = SegSource(d.name, d)
    discover_cv_preds()
    for d in extra_segs:
        d = Path(d)
        SEG_SOURCES[d.name] = SegSource(d.name, d)
    discover_score_tables()


def cases_for(root, sort, metric):
    """Return [{case, score}] for a root, optionally sorted by a score table (asc)."""
    cases = root.list_cases()
    table = SCORE_TABLES.get(sort)
    if table is None or metric not in METRICS:
        return [{'case': c, 'score': None} for c in cases]
    out = []
    for c in cases:
        v = table.get(c, {}).get(metric)
        out.append({'case': c, 'score': v})
    # cases with a score first (worst -> best), then the rest alphabetically
    scored = sorted((o for o in out if o['score'] is not None), key=lambda o: o['score'])
    unscored = [o for o in out if o['score'] is None]
    return scored + unscored


# --------------------------------------------------------------------------- #
# Volume loading with a small LRU cache (volumes are ~35 MB each).
# --------------------------------------------------------------------------- #
_cache = OrderedDict()
_cache_lock = threading.Lock()
_CACHE_MAX = 40                        # ~holds current + neighbour cases (≈7 vols each)
_load_locks = {}                       # per-key lock: avoids N threads decompressing
_load_locks_guard = threading.Lock()   # the same 35 MB volume at once (thundering herd)
_prefetch_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)


def _key_lock(key):
    with _load_locks_guard:
        lk = _load_locks.get(key)
        if lk is None:
            lk = _load_locks[key] = threading.Lock()
        return lk


def _load(path, kind):
    """kind='img' -> normalised float32; kind='seg' -> int16 labels."""
    key = (str(path), kind)
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    with _key_lock(key):
        with _cache_lock:                      # recheck: another thread may have loaded it
            if key in _cache:
                _cache.move_to_end(key)
                return _cache[key]
        img = nib.load(str(path))
        data = img.get_fdata(dtype=np.float32)
        data = np.rint(data).astype(np.int16) if kind == 'seg' else _normalise(data)
        with _cache_lock:
            _cache[key] = data
            _cache.move_to_end(key)
            while len(_cache) > _CACHE_MAX:
                _cache.popitem(last=False)
        return data


def _normalise(bg):
    pos = bg[bg > 0]
    lo, hi = np.percentile(pos, [1, 99]) if pos.size else (0.0, 1.0)
    return np.clip((bg - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)


def _prefetch_one(path, kind):
    try:
        _load(path, kind)
    except Exception:  # noqa - prefetch is best-effort
        pass


def prefetch_case(root, case, seg_names, modalities):
    """Warm the cache for a whole case (given modalities + segs) in the background."""
    for m in modalities:
        p = root.case_path(case, m)
        if p.exists():
            _prefetch_pool.submit(_prefetch_one, p, 'img')
    for name in seg_names:
        src = SEG_SOURCES.get(name)
        if src is not None and src.seg_path(case).exists():
            _prefetch_pool.submit(_prefetch_one, src.seg_path(case), 'seg')


def _get_slice(vol, idx, axis):
    sl = np.take(vol, idx, axis=axis)
    if axis < 2:
        sl = np.rot90(sl)
    return sl


def _seg_slice(seg_src, case, idx, axis):
    if seg_src is None:
        return None
    sp = seg_src.seg_path(case)
    if not sp.exists():
        return None
    return _get_slice(_load(sp, 'seg'), idx, axis)


def render_png(root, case, modality, axis, idx, seg_src, labels, alpha,
               mode='labels', ref_src=None):
    bg_path = root.case_path(case, modality)
    bg = _load(bg_path, 'img')
    idx = max(0, min(bg.shape[axis] - 1, idx))
    bg_sl = _get_slice(bg, idx, axis)
    rgb = np.stack([bg_sl] * 3, axis=-1)

    seg_sl = _seg_slice(seg_src, case, idx, axis)
    if mode == 'diff':
        # pred (seg) vs reference (ref, usually GT), on union of selected labels
        ref_sl = _seg_slice(ref_src, case, idx, axis)
        pred_fg = np.isin(seg_sl, labels) if seg_sl is not None else np.zeros(bg_sl.shape, bool)
        ref_fg = np.isin(ref_sl, labels) if ref_sl is not None else np.zeros(bg_sl.shape, bool)
        for key, m in (('fn', ref_fg & ~pred_fg), ('fp', pred_fg & ~ref_fg),
                       ('tp', pred_fg & ref_fg)):
            if m.any():
                rgb[m] = rgb[m] * (1 - alpha) + np.array(DIFF_COLORS[key]) * alpha
    elif seg_sl is not None:
        for lbl in labels:
            m = seg_sl == lbl
            if m.any():
                rgb[m] = rgb[m] * (1 - alpha) + np.array(LABEL_COLORS[lbl]) * alpha
    buf = io.BytesIO()
    mpimg.imsave(buf, np.clip(rgb, 0, 1), format='png')
    return buf.getvalue()


def seg_meta(root, case, modality, seg_src):
    bg_path = root.case_path(case, modality)
    if not bg_path.exists():
        return {'exists': False}
    bg = _load(bg_path, 'img')
    out = {'exists': True, 'shape': list(bg.shape), 'counts': {}, 'best': {}}
    if seg_src is None:
        out['best'] = {a: bg.shape[a] // 2 for a in (0, 1, 2)}
        return out
    sp = seg_src.seg_path(case)
    if not sp.exists():
        out['seg_missing'] = True
        out['best'] = {a: bg.shape[a] // 2 for a in (0, 1, 2)}
        return out
    seg = _load(sp, 'seg')
    for lbl, name in LABEL_NAMES.items():
        out['counts'][name] = int((seg == lbl).sum())
    fg = seg > 0
    for a in (0, 1, 2):
        if fg.any():
            counts = fg.sum(axis=tuple(i for i in range(3) if i != a))
            out['best'][a] = int(np.argmax(counts))
        else:
            out['best'][a] = bg.shape[a] // 2
    return out


# --------------------------------------------------------------------------- #
# HTTP handler
# --------------------------------------------------------------------------- #
_case_re = re.compile(r'^BraTS-MET-[0-9]+-[0-9]+$')


def _valid_case(c):
    return bool(_case_re.match(c or ''))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # quiet

    def _write(self, ctype, body, code=200, cache=None):
        # A client that scrubbed on to the next slice cancels the request, so
        # writing to the closed socket raises — that's expected, swallow it.
        try:
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            if cache:
                self.send_header('Cache-Control', cache)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self._dead = True

    def _json(self, obj, code=200):
        self._write('application/json', json.dumps(obj).encode(), code)

    def _png(self, data):
        self._write('image/png', data, cache='no-store')

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == '/':
                self._write('text/html; charset=utf-8', INDEX_HTML.encode())
            elif u.path == '/api/config':
                self._json({
                    'roots': list(IMAGE_ROOTS.keys()),
                    'segs': list(SEG_SOURCES.keys()),
                    'score_tables': list(SCORE_TABLES.keys()),
                    'metrics': METRICS,
                    'modalities': MODALITIES,
                    'labels': [{'id': k, 'name': v, 'color': LABEL_COLORS[k]}
                               for k, v in LABEL_NAMES.items()],
                    'axes': AXIS_NAMES,
                })
            elif u.path == '/api/cases':
                root = IMAGE_ROOTS.get(q.get('root'))
                if root is None:
                    self._json({'cases': []})
                else:
                    self._json({'cases': cases_for(root, q.get('sort', ''),
                                                    q.get('metric', 'mean'))})
            elif u.path == '/api/meta':
                root = IMAGE_ROOTS.get(q.get('root'))
                seg = SEG_SOURCES.get(q.get('seg'))
                case = q.get('case')
                if root is None or not _valid_case(case):
                    self._json({'exists': False}, 400)
                    return
                self._json(seg_meta(root, case, q.get('modality', 't1c'), seg))
            elif u.path == '/api/casemetrics':
                self._json({'metrics': case_metrics(q.get('table'), q.get('case'))})
            elif u.path == '/api/prefetch':
                root = IMAGE_ROOTS.get(q.get('root'))
                case = q.get('case')
                if root is not None and _valid_case(case):
                    mods = [m for m in q.get('mods', '').split(',') if m in MOD_SUFFIX] or MODALITIES
                    segs = [s for s in q.get('segs', '').split(',') if s]
                    prefetch_case(root, case, segs, mods)
                self._json({'ok': True})
            elif u.path == '/slice':
                root = IMAGE_ROOTS.get(q.get('root'))
                seg = SEG_SOURCES.get(q.get('seg'))
                case = q.get('case')
                if root is None or not _valid_case(case):
                    self.send_error(400)
                    return
                axis = int(q.get('axis', 2))
                idx = int(q.get('idx', 0))
                modality = q.get('modality', 't1c')
                alpha = float(q.get('alpha', 0.5))
                labels = [int(x) for x in q.get('labels', '1,2,3,4').split(',') if x]
                mode = q.get('mode', 'labels')
                ref = SEG_SOURCES.get(q.get('ref'))
                self._png(render_png(root, case, modality, axis, idx, seg, labels,
                                     alpha, mode=mode, ref_src=ref))
            else:
                self.send_error(404)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # client scrubbed away / closed the tab
        except FileNotFoundError as e:
            if not getattr(self, '_dead', False):
                self._json({'error': str(e)}, 404)
        except Exception as e:  # noqa
            if not getattr(self, '_dead', False):
                self._json({'error': repr(e)}, 500)


INDEX_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>BraTS Review</title>
<style>
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;font:13px/1.4 system-ui,sans-serif;background:#111;color:#ddd;display:flex;height:100vh}
#side{width:280px;flex:0 0 280px;background:#1a1a1a;border-right:1px solid #333;display:flex;flex-direction:column;overflow:hidden}
#side h1{font-size:14px;margin:10px 12px;color:#fff}
#casewrap{flex:1;overflow-y:auto;border-top:1px solid #333;border-bottom:1px solid #333}
.case{padding:4px 12px;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.case:hover{background:#252525}
.case.sel{background:#2b4a6b;color:#fff}
#filter{margin:6px 12px;padding:5px;width:calc(100% - 24px);background:#000;border:1px solid #444;color:#ddd;border-radius:4px}
.grp{padding:8px 12px;border-bottom:1px solid #262626}
.grp label.hdr{display:block;color:#888;font-size:11px;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px}
select{width:100%;background:#000;color:#ddd;border:1px solid #444;padding:4px;border-radius:4px}
.row{display:flex;gap:4px;flex-wrap:wrap}
.btn{padding:4px 8px;background:#222;border:1px solid #444;border-radius:4px;cursor:pointer;color:#ccc}
.btn.on{background:#2b4a6b;color:#fff;border-color:#3a6ea5}
.lbl{display:inline-flex;align-items:center;gap:5px;padding:3px 7px;border:1px solid #444;border-radius:4px;cursor:pointer;margin:2px 0}
.sw{width:12px;height:12px;border-radius:2px;display:inline-block}
#main{flex:1;display:flex;flex-direction:column;overflow:hidden}
#top{padding:8px 14px;border-bottom:1px solid #333;display:flex;align-items:center;gap:14px;flex-wrap:wrap}
#counts span{margin-right:12px}
#metrics{padding:0 14px}
#metrics table{border-collapse:collapse;font-size:12px}
#metrics td,#metrics th{padding:2px 10px;text-align:right;border-bottom:1px solid #262626}
#metrics th{color:#888;font-weight:normal}
#metrics td:first-child,#metrics th:first-child{text-align:left;color:#9cf}
#metrics .miss{color:#e77}
#view{flex:1;display:flex;align-items:center;justify-content:center;gap:6px;overflow:hidden;background:#000}
.panel{position:relative;display:flex;align-items:center;justify-content:center;height:100%;min-width:0;flex:1}
.panel img{max-width:100%;max-height:100%;image-rendering:pixelated}
.ptitle{position:absolute;top:6px;left:6px;background:#000a;padding:2px 7px;border-radius:4px;font-size:11px;color:#9cf;z-index:1}
#ctl{padding:8px 14px;border-top:1px solid #333;display:flex;align-items:center;gap:10px}
#slice{flex:1}
.mono{font-variant-numeric:tabular-nums}
kbd{background:#333;border-radius:3px;padding:1px 5px;font-size:11px}
</style></head>
<body>
<div id="side">
  <h1>BraTS Case Review</h1>
  <div class="grp">
    <label class="hdr">Image root</label>
    <select id="root"></select>
  </div>
  <div class="grp">
    <label class="hdr">Sort cases by (worst→best)</label>
    <select id="sort"></select>
    <div class="row" style="margin-top:4px"><select id="metric" style="flex:1"></select></div>
  </div>
  <input id="filter" placeholder="filter cases…">
  <div id="casewrap"></div>
  <div class="grp">
    <label class="hdr">Overlay (segmentation)</label>
    <select id="seg"></select>
    <label class="lbl" style="margin-top:6px"><input type=checkbox id="ovon" checked> show overlay <span style="color:#666">(o)</span></label>
    <label class="lbl"><input type=checkbox id="diff"> diff vs GT <span style="color:#666">(d)</span></label>
    <div id="difflegend" style="display:none;font-size:11px;margin-top:4px">
      <span class=sw style="background:#1ce633"></span> agree
      <span class=sw style="background:#ff2626;margin-left:8px"></span> pred-only (FP)
      <span class=sw style="background:#3380ff;margin-left:8px"></span> GT-only (miss)
    </div>
  </div>
  <div class="grp">
    <label class="hdr">Compare panel</label>
    <select id="cmp"></select>
  </div>
  <div class="grp">
    <label class="hdr">Modality</label>
    <div class="row" id="mods"></div>
  </div>
  <div class="grp">
    <label class="hdr">Axis</label>
    <div class="row" id="axes"></div>
  </div>
  <div class="grp">
    <label class="hdr">Labels</label>
    <div id="labels"></div>
  </div>
  <div class="grp">
    <label class="hdr">Overlay opacity <span id="av" class="mono"></span></label>
    <input id="alpha" type="range" min="0" max="1" step="0.05" value="0.5" style="width:100%">
  </div>
</div>
<div id="main">
  <div id="top">
    <b id="title">—</b>
    <button class="btn" id="jump" title="jump to largest-lesion slice">⤓ lesion</button>
    <span style="color:#666">keys: <kbd>↑</kbd><kbd>↓</kbd> slice · <kbd>←</kbd><kbd>→</kbd> case · <kbd>o</kbd> overlay · <kbd>d</kbd> diff</span>
  </div>
  <div id="metrics"></div>
  <div id="counts" class="mono" style="padding:4px 14px"></div>
  <div id="view">
    <div class="panel"><div class="ptitle" id="ptitle1"></div><img id="img" alt="pick a case"></div>
    <div class="panel" id="panel2" style="display:none"><div class="ptitle" id="ptitle2"></div><img id="img2"></div>
  </div>
  <div id="ctl">
    <button class="btn" id="prev">◀</button>
    <input id="slice" type="range" min="0" max="154" value="77">
    <button class="btn" id="next">▶</button>
    <span class="mono" id="sinfo" style="width:120px;text-align:right">—</span>
  </div>
</div>
<script>
const S={root:null,case:null,seg:null,cmp:'',sort:'',metric:'mean',overlayOn:true,diff:false,ref:'',
         modality:'t1c',axis:2,idx:77,alpha:0.5,labels:new Set([1,2,3,4]),cfg:null,meta:null,cases:[]};
const $=id=>document.getElementById(id);
async function j(u){const r=await fetch(u);return r.json();}
const enc=encodeURIComponent;
function fmt(s){return s==null?'':s.toFixed(3);}

async function init(){
  S.cfg=await j('/api/config');
  const rootSel=$('root');
  S.cfg.roots.forEach(r=>rootSel.add(new Option(r,r)));
  // sort + metric
  const sortSel=$('sort');sortSel.add(new Option('Name (A→Z)',''));
  S.cfg.score_tables.forEach(t=>sortSel.add(new Option(t,t)));
  const metSel=$('metric');S.cfg.metrics.forEach(m=>metSel.add(new Option(m.toUpperCase(),m)));
  metSel.value='mean';
  // overlay + compare selects
  const segSel=$('seg');segSel.add(new Option('(none)',''));
  const cmpSel=$('cmp');cmpSel.add(new Option('(none)',''));
  S.cfg.segs.forEach(s=>{segSel.add(new Option(s,s));cmpSel.add(new Option(s,s));});
  S.ref=S.cfg.segs.find(s=>s.startsWith('GT'))||'';   // reference for diff mode
  // --- defaults: best fold-0 CV model (copypaste_warmstart) vs GT ---
  const bestSort=S.cfg.score_tables.find(t=>t.includes('copypaste_warmstart'));
  const bestSeg=S.cfg.segs.find(s=>s.includes('CopyPasteWarmStart')&&s.includes('fold_0'));
  const trainRoot=S.cfg.roots.find(r=>r.includes('imagesTr'));
  if(trainRoot)rootSel.value=trainRoot;
  if(bestSort){sortSel.value=bestSort;S.sort=bestSort;}
  const defSeg=bestSeg||S.cfg.segs.find(s=>s.startsWith('CV'))||S.cfg.segs.find(s=>!s.startsWith('GT'))||S.cfg.segs[0]||'';
  segSel.value=defSeg;S.seg=defSeg;
  cmpSel.value=S.ref;S.cmp=S.ref;
  rootSel.onchange=async()=>{S.root=rootSel.value;await loadCases();};
  sortSel.onchange=async()=>{S.sort=sortSel.value;await loadCases();};
  metSel.onchange=async()=>{S.metric=metSel.value;if(S.sort)await loadCases();};
  segSel.onchange=()=>{S.seg=segSel.value;refreshMeta();};
  cmpSel.onchange=()=>{S.cmp=cmpSel.value;draw();};
  $('ovon').onchange=e=>{S.overlayOn=e.target.checked;draw();};
  $('diff').onchange=e=>{S.diff=e.target.checked;$('difflegend').style.display=S.diff?'':'none';draw();};
  S.cfg.modalities.forEach(m=>{const b=document.createElement('div');b.className='btn'+(m===S.modality?' on':'');b.textContent=m;b.onclick=()=>{S.modality=m;[...$('mods').children].forEach(c=>c.classList.toggle('on',c===b));refreshMeta();};$('mods').appendChild(b);});
  Object.entries(S.cfg.axes).forEach(([a,name])=>{const b=document.createElement('div');b.className='btn'+(+a===S.axis?' on':'');b.textContent=name;b.dataset.a=a;b.onclick=()=>{S.axis=+a;[...$('axes').children].forEach(c=>c.classList.toggle('on',c===b));applyMeta(true);};$('axes').appendChild(b);});
  S.cfg.labels.forEach(l=>{const el=document.createElement('label');el.className='lbl';const c='rgb('+l.color.map(x=>Math.round(x*255)).join(',')+')';el.innerHTML=`<input type=checkbox checked> <span class=sw style="background:${c}"></span>${l.name} (${l.id})`;el.querySelector('input').onchange=e=>{e.target.checked?S.labels.add(l.id):S.labels.delete(l.id);draw();};$('labels').appendChild(el);});
  $('alpha').oninput=e=>{S.alpha=+e.target.value;$('av').textContent=S.alpha.toFixed(2);draw();};$('av').textContent='0.50';
  $('slice').oninput=e=>{S.idx=+e.target.value;draw();};
  $('prev').onclick=()=>stepCase(-1);$('next').onclick=()=>stepCase(1);
  $('jump').onclick=()=>{if(S.meta&&S.meta.best){S.idx=S.meta.best[S.axis];$('slice').value=S.idx;draw();}};
  $('filter').oninput=renderCases;
  document.onkeydown=e=>{
    if(e.target.tagName==='INPUT'&&e.target.type==='text')return;
    if(e.key==='ArrowUp'){S.idx=Math.min(+$('slice').max,S.idx+1);$('slice').value=S.idx;draw();e.preventDefault();}
    if(e.key==='ArrowDown'){S.idx=Math.max(0,S.idx-1);$('slice').value=S.idx;draw();e.preventDefault();}
    if(e.key==='ArrowLeft'){stepCase(-1);e.preventDefault();}
    if(e.key==='ArrowRight'){stepCase(1);e.preventDefault();}
    if(e.key==='o'){S.overlayOn=!S.overlayOn;$('ovon').checked=S.overlayOn;draw();e.preventDefault();}
    if(e.key==='d'){S.diff=!S.diff;$('diff').checked=S.diff;$('difflegend').style.display=S.diff?'':'none';draw();e.preventDefault();}
  };
  S.root=rootSel.value;await loadCases();
}
async function loadCases(){
  const d=await j(`/api/cases?root=${enc(S.root)}&sort=${enc(S.sort)}&metric=${enc(S.metric)}`);
  S.cases=d.cases;renderCases();
  if(S.cases.length){selectCase(S.cases[0].case);}else{S.case=null;$('img').removeAttribute('src');}
}
function curIndex(){return S.cases.findIndex(o=>o.case===S.case);}
let _caseEls={};   // case id -> its list <div>, so selection just moves a class (no full rebuild)
function renderCases(){const f=$('filter').value.toLowerCase();const w=$('casewrap');w.innerHTML='';_caseEls={};
  const frag=document.createDocumentFragment();
  S.cases.filter(o=>o.case.toLowerCase().includes(f)).forEach(o=>{
    const el=document.createElement('div');el.className='case'+(o.case===S.case?' sel':'');
    const sc=o.score!=null?`<span class=mono style="color:#e88;margin-right:6px">${fmt(o.score)}</span>`:'';
    el.innerHTML=sc+o.case;el.onclick=()=>selectCase(o.case);frag.appendChild(el);_caseEls[o.case]=el;});
  w.appendChild(frag);}
function highlight(){const prev=$('casewrap').querySelector('.case.sel');if(prev)prev.classList.remove('sel');
  const el=_caseEls[S.case];if(el){el.classList.add('sel');el.scrollIntoView({block:'nearest'});}}
function stepCase(d){const i=curIndex();const n=i+d;if(n>=0&&n<S.cases.length)selectCase(S.cases[n].case);}
function prefetch(caseId,mods){if(!caseId)return;
  const segs=[...new Set([S.seg,S.cmp,S.ref].filter(Boolean))].join(',');
  fetch(`/api/prefetch?root=${enc(S.root)}&case=${caseId}&segs=${enc(segs)}&mods=${enc(mods)}`).catch(()=>{});}
async function selectCase(c){S.case=c;highlight();
  loadMetrics();
  prefetch(c,'');                             // whole current case: all modalities
  const nx=S.cases[curIndex()+1];if(nx)prefetch(nx.case,S.modality);  // next case: current modality only
  await refreshMeta(true);}
async function loadMetrics(){
  if(!S.sort){$('metrics').innerHTML='';return;}
  const d=await j(`/api/casemetrics?table=${enc(S.sort)}&case=${S.case}`);
  const m=d.metrics;
  if(!m){$('metrics').innerHTML=`<span style="color:#666;font-size:12px">no metrics for this case in ${S.sort}</span>`;return;}
  const f2=x=>x==null?'–':x.toFixed(3),f0=x=>x==null?'–':Math.round(x);
  let h=`<table><tr><th>${S.sort}</th><th>DSC</th><th>NSD</th><th>HD95</th><th>TP</th><th>FP</th><th>FN</th><th>F1</th></tr>`;
  for(const r of ['et','tc','wt','rc']){const x=m[r];const miss=(x.dsc!=null&&x.dsc<0.3)?' class=miss':'';
    h+=`<tr${miss}><td>${r.toUpperCase()}</td><td>${f2(x.dsc)}</td><td>${f2(x.nsd)}</td><td>${f2(x.hd95)}</td><td>${f0(x.tp)}</td><td>${f0(x.fp)}</td><td>${f0(x.fn)}</td><td>${f2(x.f1)}</td></tr>`;}
  $('metrics').innerHTML=h+'</table>';}
async function refreshMeta(jump){S.meta=await j(`/api/meta?root=${enc(S.root)}&case=${S.case}&modality=${S.modality}&seg=${enc(S.seg)}`);applyMeta(jump);}
function applyMeta(jump){if(!S.meta||!S.meta.exists){$('title').textContent=S.case+' (image missing)';$('img').removeAttribute('src');return;}
  const sh=S.meta.shape;$('slice').max=sh[S.axis]-1;
  if(jump){S.idx=(S.meta.best&&S.meta.best[S.axis]!=null)?S.meta.best[S.axis]:Math.floor(sh[S.axis]/2);}
  S.idx=Math.min(S.idx,sh[S.axis]-1);$('slice').value=S.idx;
  let ct='';if(S.meta.counts)for(const[k,v]of Object.entries(S.meta.counts))if(v>0)ct+=`<span>${k}: ${v}</span>`;
  if(S.meta.seg_missing)ct='<span style="color:#c66">no seg for this case</span>';
  const o=S.cases[curIndex()];if(o&&o.score!=null)ct=`<span style="color:#e88">${S.sort} ${S.metric.toUpperCase()} DSC=${fmt(o.score)}</span>`+ct;
  $('counts').innerHTML=ct;draw();}
function sliceUrl(seg,diff){const lbls=(S.overlayOn?[...S.labels]:[]).join(',')||'0';
  let u=`/slice?root=${enc(S.root)}&case=${S.case}&modality=${S.modality}&axis=${S.axis}&idx=${S.idx}&seg=${enc(seg)}&labels=${lbls}&alpha=${S.alpha}`;
  if(diff&&S.overlayOn)u+=`&mode=diff&ref=${enc(S.ref)}`;
  return u;}
// coalesce image loads: at most one request per panel in flight; while it
// loads, later draws only update the target url, then load once when free.
const _want={img:null,img2:null},_cur={img:null,img2:null},_busy={img:false,img2:false};
function setImg(id,url){_want[id]=url;pump(id);}
function pump(id){if(_busy[id])return;const url=_want[id];if(url==null||url===_cur[id])return;
  _busy[id]=true;_cur[id]=url;const el=$(id);
  el.onload=el.onerror=()=>{_busy[id]=false;pump(id);};el.src=url;}
function draw(){if(!S.meta||!S.meta.exists)return;
  $('title').textContent=`${S.case}  ·  ${S.modality.toUpperCase()}  ·  ${S.cfg.axes[S.axis]}`;
  $('sinfo').textContent=`${S.cfg.axes[S.axis][0]} ${S.idx}/${S.meta.shape[S.axis]-1}`;
  const t1=!S.overlayOn?'overlay off':(S.diff?`${S.seg||'?'} △ vs GT`:(S.seg||'no overlay'));
  $('ptitle1').textContent=t1;
  setImg('img',sliceUrl(S.seg,S.diff));
  if(S.cmp){$('panel2').style.display='';$('ptitle2').textContent=(S.overlayOn?S.cmp:'overlay off');setImg('img2',sliceUrl(S.cmp,false));}
  else{$('panel2').style.display='none';}}
init();
</script>
</body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', type=int, default=8757)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--images', action='append', default=[], help='extra image root dir')
    ap.add_argument('--seg', action='append', default=[], help='extra seg source dir')
    args = ap.parse_args()

    discover_sources(args.images, args.seg)
    print('Image roots :', ', '.join(IMAGE_ROOTS) or '(none)')
    print('Seg sources :', ', '.join(SEG_SOURCES) or '(none)')
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'\n  →  http://{args.host}:{args.port}\n')
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print('\nbye')


if __name__ == '__main__':
    main()
