from __future__ import annotations

import argparse
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
SOURCES = os.path.join(REPO, "third_party", "baselines")
ROOT_VAR = "$DECOUPATTN_DATA_ROOT"


def load_sources(directory: str = SOURCES) -> dict[str, dict]:
    out = {}
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name, "source.toml")
        if os.path.isfile(path):
            with open(path, "rb") as f:
                out[name] = {**tomllib.load(f), "dir": os.path.join(directory, name)}
    return out


def data_root(dry: bool) -> str:
    root = os.environ.get("DECOUPATTN_DATA_ROOT")
    if root:
        return root
    if dry:
        return ROOT_VAR
    sys.exit("set DECOUPATTN_DATA_ROOT to the data root (see README.md)")


def destination(src: dict, root: str) -> str:
    return os.path.join(root, "baselines", src.get("dest") or src["name"])


def shell(cmd: list[str]) -> str:
    return " ".join(f'"{a}"' if a.startswith(ROOT_VAR) else shlex.quote(a) for a in cmd)


def clone_cmds(src: dict, dest: str) -> list[list[str]]:
    cmds = [["git", "init", "--quiet", dest],
            ["git", "-C", dest, "remote", "add", "origin", src["url"]],
            ["git", "-C", dest, "fetch", "--depth", "1", "origin", src["commit"]],
            ["git", "-C", dest, "checkout", "--quiet", "--detach", "FETCH_HEAD"]]
    sub = src.get("submodules")
    if sub:
        paths = [] if sub is True else ["--", *sub]
        cmds.append(["git", "-C", dest, "submodule", "update", "--init", "--recursive", "--depth", "1", *paths])
    return cmds


def required(src: dict, dest: str) -> list[tuple[dict, str]]:
    return [(r, os.path.join(dest, r["path"])) for r in src.get("requires", [])]


def patch_files(src: dict) -> list[str]:
    return [os.path.join(src["dir"], p) for p in src.get("patches", [])]


def patch_cmds(src: dict, dest: str) -> list[list[str]]:
    return [["git", "-C", dest, "apply", p] for p in patch_files(src)]


def apply_patches(src: dict, dest: str) -> None:
    for patch in patch_files(src):
        applied = subprocess.run(["git", "-C", dest, "apply", "--reverse", "--check", patch], capture_output=True,
                                 check=False).returncode == 0
        if applied:
            continue
        subprocess.run(["git", "-C", dest, "apply", patch], check=True)
        print(f"  applied {os.path.basename(patch)}")


def state(src: dict, root: str) -> str:
    dest = destination(src, root)
    if not os.path.isdir(os.path.join(dest, ".git")):
        return "missing"
    head = subprocess.run(["git", "-C", dest, "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    head = head.stdout.strip()
    return "present" if head == src["commit"] else head


def fetch(src: dict, root: str, dry: bool) -> bool:
    dest = destination(src, root)
    if dry:
        cmds = clone_cmds(src, dest)
        for r, path in required(src, dest):
            cmds += clone_cmds(r, path)
        print("\n".join(shell(c) for c in cmds + patch_cmds(src, dest)))
        return True
    st = state(src, root)
    if st == "missing":
        tmp = dest + ".partial"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        for c in clone_cmds(src, tmp):
            subprocess.run(c, check=True)
        os.replace(tmp, dest)
        print(f"  {src['name']}: fetched at {src['commit'][:10]}")
    elif st == "present":
        print(f"  {src['name']}: present")
    else:
        print(f"  {src['name']}: present at {st[:10]}, not {src['commit'][:10]}; left unchanged", file=sys.stderr)
        return False
    for r, path in required(src, dest):
        if not os.path.isdir(os.path.join(path, ".git")):
            if os.path.isdir(path) and not os.listdir(path):
                os.rmdir(path)
            for c in clone_cmds(r, path):
                subprocess.run(c, check=True)
            print(f"  {src['name']}: fetched {r['path']} at {r['commit'][:10]}")
    apply_patches(src, dest)
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", action="append", help="directory of third_party/baselines/ (repeatable; default: all)")
    ap.add_argument("--list", action="store_true", help="list the repositories and exit")
    ap.add_argument("--dry-run", action="store_true", help="print the git commands instead of running them")
    a = ap.parse_args(argv)
    sources = load_sources()
    if a.list:
        root = os.environ.get("DECOUPATTN_DATA_ROOT")
        for name, s in sources.items():
            where = f"  [{state(s, root)}]" if root else ""
            print(f"{name:28s} {s['commit'][:10]}  {s['license']:12s} {', '.join(s.get('used_by', [])):18s} "
                  f"{len(s.get('patches', []))} patch(es){where}")
        return 0
    unknown = [n for n in (a.name or []) if n not in sources]
    if unknown:
        ap.error(f"unknown repository {', '.join(unknown)}; known: {', '.join(sources)}")
    root = data_root(a.dry_run)
    ok = True
    for name in a.name or list(sources):
        if a.dry_run:
            print(f"# {name}")
        ok = fetch(sources[name], root, a.dry_run) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
