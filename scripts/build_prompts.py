from __future__ import annotations

import argparse
import json
import os
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(REPO, "data", "prompts", "moviegenbench.json")
RAW = "https://raw.githubusercontent.com/facebookresearch/MovieGenBench/{commit}/{file}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="a local copy of MovieGenVideoBench.txt (default: download the pinned commit)")
    ap.add_argument("--out", help="output file (default: <DECOUPATTN_DATA_ROOT>/prompts/moviegenbench.json)")
    a = ap.parse_args()
    with open(INDEX) as f:
        idx = json.load(f)
    if a.src:
        with open(a.src, encoding="utf-8") as f:
            text = f.read()
    else:
        with urllib.request.urlopen(RAW.format(commit=idx["commit"], file=idx["file"])) as r:
            text = r.read().decode("utf-8")
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    prompts = {clip: lines[n] for split in idx["splits"].values() for clip, n in split.items()}
    out = a.out or os.path.join(os.environ["DECOUPATTN_DATA_ROOT"], "prompts", "moviegenbench.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".tmp", "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=1, ensure_ascii=False)
    os.replace(out + ".tmp", out)
    print(f"{len(prompts)} prompts ({len(lines)} lines of {idx['file']} @ {idx['commit'][:7]}) -> {out}")


if __name__ == "__main__":
    main()
