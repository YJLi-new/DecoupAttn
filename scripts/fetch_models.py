from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = os.path.join(REPO, "configs", "downloads.toml")
MARKER = ".decoupattn_fetch.json"
ROOT_VAR = "$DECOUPATTN_DATA_ROOT"
WEIGHT_KEYS = ("repo", "revision", "include", "exclude")


def load_spec(path: str = SPEC) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def data_root(dry: bool) -> str:
    root = os.environ.get("DECOUPATTN_DATA_ROOT")
    if root:
        return root
    if dry:
        return ROOT_VAR
    sys.exit("set DECOUPATTN_DATA_ROOT to the data root (see README.md)")


def shell(cmd: list[str]) -> str:
    return " ".join(f'"{a}"' if a.startswith(ROOT_VAR) else shlex.quote(a) for a in cmd)


def clone_cmds(r: dict, dest: str) -> list[list[str]]:
    cmds = [["git", "clone", r["url"], dest]]
    if r.get("commit"):
        cmds.append(["git", "-C", dest, "checkout", r["commit"]])
    return cmds


def apply_patches(r: dict, dest: str) -> None:
    for p in r.get("patches", []):
        path = os.path.join(dest, p["file"])
        with open(path, encoding="utf-8") as f:
            text = f.read()
        if p["new"] in text:
            continue
        if p["old"] not in text:
            raise SystemExit(f"cannot patch {path}: the text to replace is not there (other commit?)")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text.replace(p["old"], p["new"], 1))
        print(f"  patched {p['file']}")


def fetch_repo(r: dict, root: str, dry: bool) -> bool:
    dest = os.path.join(root, r["dest"])
    if dry:
        print("\n".join(shell(c) for c in clone_cmds(r, dest)))
        for file in dict.fromkeys(p["file"] for p in r.get("patches", [])):
            print(f"# then patch {r['dest']}/{file} as listed in configs/downloads.toml")
        return True
    if os.path.isdir(os.path.join(dest, ".git")):
        git = subprocess.run(["git", "-C", dest, "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        head = git.stdout.strip()
        if r.get("commit") and not head.startswith(r["commit"]):
            print(f"  {r['dest']}: present at {head[:10]}, not {r['commit']}; left unchanged", file=sys.stderr)
        else:
            print(f"  {r['dest']}: present")
    else:
        tmp = dest + ".partial"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        for c in clone_cmds(r, tmp):
            subprocess.run(c, check=True)
        os.replace(tmp, dest)
    apply_patches(r, dest)
    return True


def hf_cmd(w: dict, root: str) -> list[str]:
    cmd = ["hf", "download", w["repo"]]
    if w.get("revision"):
        cmd += ["--revision", w["revision"]]
    for p in w.get("include", []):
        cmd += ["--include", p]
    for p in w.get("exclude", []):
        cmd += ["--exclude", p]
    if w["dest"]:
        cmd += ["--local-dir", os.path.join(root, w["dest"])]
    return cmd


def entry(w: dict) -> dict:
    return {k: w.get(k) for k in WEIGHT_KEYS}


def completed(dest: str) -> list:
    try:
        with open(os.path.join(dest, MARKER)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


def fetch_weights(w: dict, root: str, dry: bool) -> bool:
    if dry:
        print(shell(hf_cmd(w, root)))
        return True
    dest = os.path.join(root, w["dest"]) if w["dest"] else None
    if dest and entry(w) in completed(dest):
        print(f"  {w['repo']} -> {w['dest']}: complete")
        return True
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import GatedRepoError

    print(f"  {w['repo']} -> {w['dest'] or 'Hugging Face cache'} (about {w.get('gib', '?')} GiB)", flush=True)
    try:
        snapshot_download(w["repo"], revision=w.get("revision"), allow_patterns=w.get("include"),
                          ignore_patterns=w.get("exclude"), local_dir=dest)
    except GatedRepoError:
        print(f"  {w['repo']} is gated: accept its license on https://huggingface.co/{w['repo']}, run `hf auth login`, "
              "then run this command again", file=sys.stderr)
        return False
    if dest:
        done = [*completed(dest), entry(w)]
        with open(os.path.join(dest, MARKER), "w") as f:
            json.dump(done, f, indent=1)
    return True


def total_gib(m: dict, optional: bool = False) -> float:
    return sum(w.get("gib", 0.0) for w in m.get("weights", []) if optional or not w.get("optional"))


def list_models(spec: dict) -> None:
    print(f"{'model':<11} {'repositories':<44} {'GiB':>7}  gated")
    for key, m in spec.items():
        repos = ", ".join(f"{r['url'].rsplit('/', 1)[-1]}@{r.get('commit', 'HEAD')}" for r in m.get("repos", [])) or "-"
        gated = ", ".join(w["repo"] for w in m.get("weights", []) if w.get("gated")) or "-"
        extra = total_gib(m, True) - total_gib(m)
        size = f"{total_gib(m):7.1f}" + (f" (+{extra:.1f} optional)" if extra else "")
        print(f"{key:<11} {repos:<44} {size}  {gated}")


def fetch_model(key: str, m: dict, root: str, repos: bool, weights: bool, optional: bool, dry: bool) -> bool:
    print(f"# {key}: {m.get('license', '')}")
    ok = True
    if repos:
        for r in m.get("repos", []):
            ok &= fetch_repo(r, root, dry)
    if weights:
        for w in m.get("weights", []):
            if w.get("optional") and not optional:
                print(f"# optional (--optional): {shell(hf_cmd(w, root))}")
                continue
            ok &= fetch_weights(w, root, dry)
    for cmd in m.get("post", []) if weights else []:
        print(f"# then run: {cmd}" if dry else f"  next: {cmd}")
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="a model key of configs/downloads.toml, or 'all'")
    ap.add_argument("--repos", action="store_true", help="only the official code repositories")
    ap.add_argument("--weights", action="store_true", help="only the weights")
    ap.add_argument("--optional", action="store_true", help="also the entries marked optional")
    ap.add_argument("--dry-run", action="store_true", help="print the equivalent git / hf commands, fetch nothing")
    ap.add_argument("--list", action="store_true", help="list the models, their repositories and download sizes")
    ap.add_argument("--spec", default=SPEC, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    spec = load_spec(a.spec)
    if a.list:
        list_models(spec)
        return 0
    if not a.model:
        ap.error("--model or --list is required")
    keys = list(spec) if a.model == "all" else [a.model]
    unknown = [k for k in keys if k not in spec]
    if unknown:
        ap.error(f"unknown model {unknown[0]!r}; known: {', '.join(spec)}")
    root = data_root(a.dry_run)
    repos, weights = (a.repos or not a.weights), (a.weights or not a.repos)
    ok = True
    for k in keys:
        ok &= fetch_model(k, spec[k], root, repos, weights, a.optional, a.dry_run)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
