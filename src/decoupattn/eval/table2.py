from __future__ import annotations

import csv
import json
import math
import os

from .table1 import MODEL_KEYS, MODELS, compare_summary

VARIANTS = ("Tied", "Tied_dagger", "Centroid", "U", "D", "Frozen")
ROWS = ("Full", *VARIANTS)
LABELS = {"Full": "Full method", "Tied": "Tied", "Tied_dagger": "Tied†", "Centroid": "Centroid", "U": "U", "D": "D",
          "Frozen": "Frozen"}
TEX_LABELS = {"Full": "Full method", "Tied": "Tied\\vt{blocks as groups}", "Tied_dagger": "Tied$^\\dagger$\\vt{same, our plan}",
              "Centroid": "Centroid\\vt{no corrections}", "U": "U\\vt{every $J_u{=}1$}",
              "D": "D\\vt{uncompensated $\\varepsilon_u$}", "Frozen": "Frozen\\vt{no reassignment}"}
DIGITS = {"psnr": 2, "psnr_sd": 2, "dpsnr": 2, "dpsnr_sd": 2, "ms": 0, "dms": 0, "wins": 0, "supp": 0, "of": 0}
SUPPORT_DB = 0.3
SUPPORT_CLIPS = 6
TAG_PREFIX = "abl_"

REFERENCE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))))), "data", "table2")


def tag(variant: str) -> str:
    return TAG_PREFIX + variant


def printed(x: float | None, field: str) -> float | None:
    if x is None or math.isnan(x):
        return None
    return float(f"{x:.2f}") if DIGITS[field] else int(math.copysign(math.floor(abs(x) + 0.5), x))


def fmt(x: float | None, field: str) -> str:
    if x is None:
        return "--"
    if field in ("psnr", "psnr_sd", "dpsnr_sd"):
        return f"{x:.2f}"
    if field == "dpsnr":
        return f"{x:+.2f}"
    return f"{int(x):+,d}" if field == "dms" else f"{int(x):,d}"


def _read(path: str):
    with open(path) as f:
        return json.load(f)


def collect(run_dir: str, model: str) -> dict:
    rec = {"model": model, "variants": {}, "timing": {}}
    for v in ROWS:
        p = os.path.join(run_dir, "metrics", f"{tag(v)}.json")
        if os.path.exists(p):
            m = _read(p)
            rec["seed"] = m.get("seed", rec.get("seed"))
            rec["variants"][v] = {"clips": {r["clip"]: {"psnr": r["psnr"]} for r in m["clips"]}}
        p = os.path.join(run_dir, "timing", f"{tag(v)}.json")
        if os.path.exists(p):
            rec["timing"][v] = {"method_ms": _read(p)["method_ms"]}
    return rec


def load_reference(model: str, root: str | None = None) -> dict:
    return _read(os.path.join(root or REFERENCE_DIR, "reference", f"{model}.json"))


def load_paper(root: str | None = None) -> dict:
    return _read(os.path.join(root or REFERENCE_DIR, "paper.json"))


def _sd(v: list[float]) -> float | None:
    if len(v) < 2:
        return None
    mean = sum(v) / len(v)
    return math.sqrt(sum((x - mean) ** 2 for x in v) / (len(v) - 1))


def _psnr(clips: dict, expected: list[str] | None) -> dict:
    if expected is not None and set(clips) != set(expected):
        missing = sorted(set(expected) - set(clips))
        raise ValueError(f"clips do not match the split: {len(clips)} scored, {len(expected)} expected, missing {missing[:5]}")
    return {c: x["psnr"] for c, x in clips.items()}


def delta_ms(record: dict, variant: str) -> float | None:
    t = record.get("timing", {}).get(variant)
    if not t:
        return None
    if t.get("delta_ms") is not None:
        return t["delta_ms"]
    full = t.get("full_ms") or record.get("timing", {}).get("Full", {}).get("method_ms")
    return t["method_ms"] - full if full and t.get("method_ms") else None


def model_row(record: dict, expected: list[str] | None = None) -> dict:
    row = {}
    if "Full" not in record["variants"]:
        return row
    full = _psnr(record["variants"]["Full"]["clips"], expected)
    ms = record.get("timing", {}).get("Full", {}).get("method_ms")
    row["Full"] = {"psnr": printed(sum(full.values()) / len(full), "psnr"), "psnr_sd": printed(_sd(list(full.values())), "psnr_sd"),
                   "ms": printed(ms, "ms"), "n": len(full)}
    for v in VARIANTS:
        if v not in record["variants"]:
            continue
        var = _psnr(record["variants"][v]["clips"], expected)
        if set(var) != set(full):
            raise ValueError(f"{record.get('model')} {v}: its clips differ from the full method's")
        diff = [var[c] - full[c] for c in full]
        row[v] = {"dpsnr": printed(sum(diff) / len(diff), "dpsnr"), "dpsnr_sd": printed(_sd(diff), "dpsnr_sd"),
                  "wins": sum(d < 0 for d in diff), "dms": printed(delta_ms(record, v), "dms"), "n": len(diff)}
    return row


def supports(cell: dict) -> bool:
    return cell["dpsnr"] <= -SUPPORT_DB and cell["wins"] >= SUPPORT_CLIPS


def average_row(rows: dict) -> dict:
    def mean(cells, f):
        vals = [c.get(f) for c in cells]
        return printed(sum(vals) / len(vals), f) if vals and all(x is not None for x in vals) else None

    avg = {}
    full = [r["Full"] for r in rows.values() if "Full" in r]
    if len(full) == len(rows):
        avg["Full"] = {"psnr": mean(full, "psnr"), "ms": mean(full, "ms")}
    for v in VARIANTS:
        cells = [r[v] for r in rows.values() if v in r]
        if cells:
            avg[v] = {"dpsnr": mean(cells, "dpsnr"), "dms": mean(cells, "dms"),
                      "supp": sum(supports(c) for c in cells), "of": len(cells)}
    return avg


def assemble(records: dict, splits: dict | None = None) -> dict:
    rows = {m: model_row(records[m], (splits or {}).get(m)) for m in MODEL_KEYS if m in records}
    rows = {m: r for m, r in rows.items() if r}
    table = dict(rows)
    if len(rows) == len(MODEL_KEYS):
        table["average"] = average_row(rows)
    return table


def compare(table: dict, paper: dict) -> list[dict]:
    out = []
    for model, prow in paper.items():
        for row, pcell in prow.items():
            for f, pv in pcell.items():
                got = table.get(model, {}).get(row, {}).get(f)
                status = "missing" if got is None else ("matched" if abs(got - pv) < 0.5 * 10 ** -DIGITS[f] else "mismatch")
                out.append({"model": model, "column": row, "field": f, "paper": pv, "got": got, "status": status})
    return out


def _columns(table: dict) -> list[tuple[str, str]]:
    cols = [(k, name) for k, name, _, _ in MODELS if k in table]
    return [*cols, ("average", "Avg.")] if "average" in table else cols


def to_csv(table: dict, path: str) -> None:
    fields = ("psnr", "psnr_sd", "ms", "dpsnr", "dpsnr_sd", "wins", "dms", "supp", "of", "n")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "variant", *fields])
        for model, row in table.items():
            for v in ROWS:
                if v in row:
                    w.writerow([model, v, *["" if row[v].get(k) is None else row[v][k] for k in fields]])


def _quality(cell: dict | None, full: bool) -> str:
    if not cell:
        return "--"
    a, b = ("psnr", "psnr_sd") if full else ("dpsnr", "dpsnr_sd")
    s = fmt(cell.get(a), a)
    if cell.get(b) is not None:
        s += f" ± {fmt(cell[b], b)}"
    return s + (f" ({cell['wins']})" if cell.get("wins") is not None else "")


def to_markdown(table: dict) -> str:
    cols = _columns(table)
    whole = "average" in table
    head = ["Variant", *[h for _, h in cols], *(["Supp."] if whole else [])]
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    for v in ROWS:
        if not any(v in table[k] for k, _ in cols):
            continue
        f = "ms" if v == "Full" else "dms"
        avg = table.get("average", {}).get(v, {})
        supp = [f"{avg['supp']}/{avg['of']}" if avg.get("supp") is not None else ""] if whole else []
        blank = [""] if whole else []
        lines.append("| " + " | ".join([LABELS[v], *[_quality(table[k].get(v), v == "Full") for k, _ in cols], *supp]) + " |")
        lines.append("| " + " | ".join(["ms", *[fmt(table[k].get(v, {}).get(f), f) for k, _ in cols], *blank]) + " |")
    return "\n".join(lines) + "\n"


def to_latex(table: dict) -> str:
    def num(x, f):
        return "--" if x is None else fmt(x, f).replace("-", "$-$").replace(",", "{,}")

    cols = _columns(table)
    lines = []
    for v in ROWS:
        if not any(v in table[k] for k, _ in cols):
            continue
        a, b = ("psnr", "ms") if v == "Full" else ("dpsnr", "dms")
        cells = [table[k].get(v, {}) for k, _ in cols]
        avg = table.get("average", {}).get(v, {})
        supp = f"{avg['supp']}/{avg['of']}" if avg.get("supp") is not None else ""
        top = [num(c.get(a), a).replace("+", "") for c in cells]
        bot = ["--" if c.get(b) is None else f"\\tms{{{num(c[b], b)}}}" for c in cells]
        lines.append(" & ".join(["", TEX_LABELS[v], *top, supp]) + "\\\\[-0.9pt]")
        lines.append(" & ".join(["", "\\quad \\textit{\\scriptsize ms}", *bot, ""]) + "\\\\")
    return "\n".join(lines) + "\n"


def write_outputs(table: dict, out_dir: str, stem: str = "table2") -> dict:
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


__all__ = ["LABELS", "ROWS", "VARIANTS", "assemble", "average_row", "collect", "compare", "compare_summary", "delta_ms",
           "load_paper", "load_reference", "model_row", "supports", "tag", "to_csv", "to_latex", "to_markdown",
           "write_outputs"]
