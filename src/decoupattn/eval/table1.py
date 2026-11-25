from __future__ import annotations

import csv
import json
import math
import os

BASELINES = ("calibatt", "svoo", "svg2", "svgear", "sol", "pisa", "sparsepr")
COLUMNS = ("floor", *BASELINES, "ours")
LABELS = {"floor": "Floor", "calibatt": "CalibAtt", "svoo": "SVOO", "svg2": "SVG2", "svgear": "SVG-EAR",
          "sol": "Sol-Attn", "pisa": "PISA", "sparsepr": "SparsePR", "ours": "Ours"}
MODELS = (("cp1", "CP1", 35, "video"), ("cp25", "CP2.5", 4, "video"), ("wan21", "Wan2.1", 4, "video"),
          ("wan22", "Wan2.2", 4, "video"), ("hy", "HY", 50, "video"), ("nano", "Nano", 35, "video"),
          ("t25", "T2.5", 4, "controlled"), ("dr", "DR", 4, "renderer"), ("alaya", "Alaya", 50, "renderer"),
          ("unirelight", "UniRel.", 35, "renderer"))
MODEL_KEYS = tuple(m[0] for m in MODELS)
METRICS = ("psnr", "ssim", "lpips")
FIELDS = ("psnr", "psnr_sd", "speedup", "ssim", "ssim_sd", "lpips", "lpips_sd")
DIGITS = {"psnr": 2, "psnr_sd": 2, "speedup": 2, "ssim": 3, "ssim_sd": 3, "lpips": 3, "lpips_sd": 3}
HIGHER_IS_BETTER = {"psnr": True, "ssim": True, "lpips": False, "speedup": True}
ANCHORS = ("process", "common")
OURS_TAG = "decoupattn"

REFERENCE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))))), "data", "table1")


def printed(x: float | None, field: str) -> float | None:
    return None if x is None or math.isnan(x) else float(f"{x:.{DIGITS[field]}f}")


def fmt(x: float | None, field: str, sign: bool = False) -> str:
    if x is None:
        return "--"
    return f"{x:+.{DIGITS[field]}f}" if sign else f"{x:.{DIGITS[field]}f}"


def _read(path: str):
    with open(path) as f:
        return json.load(f)


def floor_tag(cfg: dict) -> str:
    return f"dense_{cfg['model']['floor_backend']}"


def reference_backend(cfg: dict) -> str:
    return cfg["model"].get("reference_backend", cfg["model"]["dense_backend"])


def collect(run_dir: str, model: str, cfg: dict) -> dict:
    tags = {"floor": floor_tag(cfg), **{b: b for b in BASELINES}, "ours": OURS_TAG}
    rec = {"model": model, "methods": {}, "timing": {}}
    for col, tag in tags.items():
        p = os.path.join(run_dir, "metrics", f"{tag}.json")
        if os.path.exists(p):
            m = _read(p)
            rec["seed"] = m.get("seed", rec.get("seed"))
            rec["methods"][col] = {"clips": {r["clip"]: {k: r[k] for k in METRICS if k in r} for r in m["clips"]}}
        p = os.path.join(run_dir, "timing", f"{tag}.json")
        if col != "floor" and os.path.exists(p):
            t = _read(p)
            rec["timing"][col] = {"dense_ms": t["dense_ms"], "method_ms": t["method_ms"]}
            if t.get("own_pipeline"):
                rec["timing"][col]["own_dense"] = True
    return rec


def load_reference(model: str, root: str | None = None) -> dict:
    return _read(os.path.join(root or REFERENCE_DIR, "reference", f"{model}.json"))


def load_paper(root: str | None = None) -> dict:
    return _read(os.path.join(root or REFERENCE_DIR, "paper.json"))


def clip_stats(clips: dict, expected: list[str] | None = None) -> dict:
    if expected is not None and set(clips) != set(expected):
        missing = sorted(set(expected) - set(clips))
        raise ValueError(f"clips do not match the split: {len(clips)} scored, {len(expected)} expected, missing {missing[:5]}")
    out = {"n": len(clips)}
    for k in METRICS:
        v = [c[k] for c in clips.values() if k in c]
        if not v:
            continue
        mean = sum(v) / len(v)
        out[k] = mean
        out[k + "_sd"] = math.sqrt(sum((x - mean) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else None
    return out


def speedup(record: dict, col: str, anchor: str = "common") -> float | None:
    if col == "floor":
        return 1.0
    t = record.get("timing", {}).get(col)
    if not t or not t.get("method_ms"):
        return None
    if anchor == "process" or t.get("own_dense"):
        dense = t.get("dense_ms")
    elif anchor == "common":
        dense = record.get("dense_ms") or record.get("timing", {}).get("ours", {}).get("dense_ms")
    else:
        raise ValueError(f"unknown anchor {anchor}")
    return dense / t["method_ms"] if dense else None


def model_row(record: dict, anchor: str = "common", expected: list[str] | None = None) -> dict:
    row = {}
    for col in COLUMNS:
        m = record["methods"].get(col)
        if m is None:
            continue
        s = clip_stats(m["clips"], expected)
        cell = {f: printed(s.get(f), f) for f in FIELDS if f != "speedup"}
        cell["speedup"] = printed(speedup(record, col, anchor), "speedup")
        cell["n"] = s["n"]
        row[col] = cell
    d = delta(row)
    if d:
        row["delta"] = d
    return row


def delta(row: dict) -> dict:
    out = {}
    for f in ("psnr", "speedup", "ssim", "lpips"):
        vals = [row.get(b, {}).get(f) for b in BASELINES]
        ours = row.get("ours", {}).get(f)
        if ours is not None and all(v is not None for v in vals):
            out[f] = printed(ours - sum(vals) / len(vals), f)
    return out


def average_row(rows: dict) -> dict:
    avg = {}
    for col in COLUMNS:
        cells = [r.get(col) for r in rows.values()]
        if not cells or any(c is None for c in cells):
            continue
        avg[col] = {}
        for f in ("psnr", "speedup", "ssim", "lpips"):
            vals = [c.get(f) for c in cells]
            if all(v is not None for v in vals):
                avg[col][f] = printed(sum(vals) / len(vals), f)
    d = delta(avg)
    if d:
        avg["delta"] = d
    return avg


def assemble(records: dict, anchor: str = "common", splits: dict | None = None) -> dict:
    rows = {m: model_row(records[m], anchor, (splits or {}).get(m)) for m in MODEL_KEYS if m in records}
    table = dict(rows)
    if len(rows) == len(MODEL_KEYS):
        table["average"] = average_row(rows)
    return table


def best(row: dict, field: str) -> str | None:
    vals = {c: row[c][field] for c in COLUMNS[1:] if c in row and row[c].get(field) is not None}
    if not vals:
        return None
    return (max if HIGHER_IS_BETTER[field] else min)(vals, key=vals.get)


def compare(table: dict, paper: dict) -> list[dict]:
    out = []
    for model, prow in paper.items():
        for col, pcell in prow.items():
            for f, pv in pcell.items():
                got = table.get(model, {}).get(col, {}).get(f)
                status = "missing" if got is None else ("matched" if abs(got - pv) < 0.5 * 10 ** -DIGITS[f] else "mismatch")
                out.append({"model": model, "column": col, "field": f, "paper": pv, "got": got, "status": status})
    return out


def compare_summary(cmp: list[dict]) -> dict:
    out: dict = {}
    for c in cmp:
        cell = out.setdefault(c["model"], {}).setdefault(c["column"], {"matched": 0, "mismatch": [], "missing": []})
        if c["status"] == "matched":
            cell["matched"] += 1
        elif c["status"] == "mismatch":
            cell["mismatch"].append((c["field"], c["got"], c["paper"]))
        else:
            cell["missing"].append(c["field"])
    return out


def _label(model: str) -> tuple[str, str]:
    for key, name, steps, _ in MODELS:
        if key == model:
            return name, f"({steps})"
    return "Avg.", ""


def to_csv(table: dict, path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "column", *FIELDS, "n"])
        for model, row in table.items():
            for col in (*COLUMNS, "delta"):
                if col in row:
                    w.writerow([model, col, *[row[col].get(k, "") if row[col].get(k) is not None else "" for k in FIELDS],
                                row[col].get("n", "")])


def _md_cell(cell: dict, a: str, b: str, bold_a: bool, bold_b: bool, sign: bool = False) -> str:
    def one(f, bold):
        if cell.get(f) is None:
            return "--"
        s = fmt(cell[f], f, sign)
        sd = cell.get(f + "_sd")
        s = f"**{s}**" if bold else s
        return f"{s} ± {fmt(sd, f + '_sd')}" if sd is not None else s
    x = one(a, bold_a)
    y = one(b, bold_b)
    if b == "speedup" and cell.get(b) is not None:
        y = y + "×"
    return f"{x} / {y}"


def to_markdown(table: dict) -> str:
    head = ["Model", *[LABELS[c] for c in COLUMNS], "Δ"]
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    for model, row in table.items():
        name, steps = _label(model)
        bests = {f: best(row, f) for f in ("psnr", "ssim", "lpips")}
        for (a, b), first in (((("psnr", "speedup")), name), (("ssim", "lpips"), steps)):
            cells = []
            for col in COLUMNS:
                cells.append(_md_cell(row[col], a, b, bests[a] == col, b != "speedup" and bests[b] == col)
                             if col in row else "--")
            cells.append(_md_cell(row["delta"], a, b, False, False, sign=True) if "delta" in row else "--")
            lines.append("| " + " | ".join([first, *cells]) + " |")
    return "\n".join(lines) + "\n"


def to_latex(table: dict) -> str:
    def val(cell, f, bold, floor):
        if cell.get(f) is None:
            return "--"
        s = fmt(cell[f], f)
        s = f"\\textbf{{{s}}}" if bold else s
        s = f"\\textcolor{{floorgray}}{{{s}}}" if floor else s
        return f"\\vsd{{{s}}}{{{fmt(cell[f + '_sd'], f + '_sd')}}}" if cell.get(f + "_sd") is not None else s

    def signed(x, f):
        return "--" if x is None else fmt(x, f, sign=True).replace("-", "$-$")

    lines = []
    for model, row in table.items():
        name, steps = _label(model)
        bests = {f: best(row, f) for f in ("psnr", "ssim", "lpips")}
        top, bot = [name], [f"\\kk{{{steps}}}" if steps else ""]
        for col in COLUMNS:
            c = row.get(col, {})
            sp = c.get("speedup")
            top += [val(c, "psnr", bests["psnr"] == col, col == "floor"), "--" if sp is None else f"\\spd{{{fmt(sp, 'speedup')}}}"]
            bot += [val(c, "ssim", bests["ssim"] == col, col == "floor"), val(c, "lpips", bests["lpips"] == col, col == "floor")]
        d = row.get("delta", {})
        top += [f"\\pos{{{signed(d.get('psnr'), 'psnr')}}}", f"\\dspd{{{fmt(d['speedup'], 'speedup', True)}}}"
                if d.get("speedup") is not None else "--"]
        bot += [f"\\pos{{{signed(d.get('ssim'), 'ssim')}}}", f"\\pos{{{signed(d.get('lpips'), 'lpips')}}}"]
        lines += [" & ".join(top) + "\\\\", " & ".join(bot) + "\\\\"]
    return "\n".join(lines) + "\n"


def write_outputs(table: dict, out_dir: str, stem: str = "table1") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    paths = {ext: os.path.join(out_dir, f"{stem}.{ext}") for ext in ("json", "csv", "md", "tex")}
    with open(paths["json"], "w") as f:
        json.dump(table, f, indent=1)
    to_csv(table, paths["csv"])
    with open(paths["md"], "w") as f:
        f.write(to_markdown(table))
    with open(paths["tex"], "w") as f:
        f.write(to_latex(table))
    return paths
