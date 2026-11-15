import argparse
import json
import runpy
import shutil
import sys
import traceback
from pathlib import Path

import kalfa
from kalfa import summary
from kalfa.commands import report as reporting
from kalfa.commands import sweep as sweeping
from kalfa.core import run as running
from kalfa.records import reader


HERE = Path(__file__).resolve().parent
RECIPE = "example.json"
MARKER = ".kalfa-demo"
KINDS = ("run", "resume", "sweep", "predict", "generate", "export", "summarize")


def folders(root=HERE):
    return sorted(path for path in Path(root).iterdir() if (path / RECIPE).is_file())


def recipe_of(folder):
    return json.loads((Path(folder) / RECIPE).read_text(encoding="utf-8"))


def prepare(folder, root=None):
    folder = Path(folder)
    root = Path(root) if root is not None else folder.parent
    for name in [*recipe_of(folder).get("needs", []), folder.name]:
        script = root / name / "make_data.py"
        if script.is_file():
            runpy.run_path(str(script), run_name="__main__")


def filled(value, records, folder):
    if isinstance(value, str):
        return value.replace("{records}", str(records)).replace("{here}", str(folder))
    if isinstance(value, dict):
        return {key: filled(item, records, folder) for key, item in value.items()}
    if isinstance(value, list):
        return [filled(item, records, folder) for item in value]
    return value


def layers(step, records, folder, device):
    sets = filled(dict(step.get("sets") or {}), records, folder)
    chosen = device or step.get("device")
    if chosen:
        sets["device"] = chosen
    params = filled(dict(step.get("params") or {}), records, folder)
    return list(sets.items()), list(params.items())


def kind_of(step):
    found = [kind for kind in KINDS if kind in step]
    if len(found) != 1:
        raise ValueError(f"a recipe step names exactly one of {KINDS}, got {sorted(step)}")
    return found[0]


def finished(record):
    manifest = reader.read_manifest(record) or {}
    return manifest.get("status") in ("done", "stopped"), manifest.get("status")


def play_step(step, folder, records, device, log):
    kind = kind_of(step)
    sets, params = layers(step, records, folder, device)
    if kind == "run":
        target = records / step["record"]
        kalfa.run(str(folder / step["run"]), sets=sets, params=params, record=str(target))
        ok, status = finished(target)
        return {"kind": kind, "record": str(target), "ok": ok, "status": status}
    if kind == "resume":
        target = records / step["record"]
        running.resume(str(records / step["resume"]), sets=sets, params=params, target=str(target))
        ok, status = finished(target)
        return {"kind": kind, "record": str(target), "ok": ok, "status": status}
    if kind == "sweep":
        root = records / step["record"]
        plan = sweeping.plan_of(paths=[str(folder / step["sweep"])], sets=sets, params=params, record=str(root))
        results = sweeping.run_local(plan, log=log)
        return {"kind": kind, "record": str(root), "ok": bool(results) and all(item is not None for item in results),
                "status": f"{sum(item is not None for item in results)}/{len(results)} points"}
    if kind == "predict":
        data = step.get("data")
        found = kalfa.api.predict(records / step["predict"], data=str(folder / data) if data else None,
                                  which=step.get("which"), device=device or step.get("device"))
        return {"kind": kind, "record": str(records / step["predict"]), "ok": len(found.table) > 0,
                "status": found.path}
    if kind == "generate":
        found = kalfa.api.generate(records / step["generate"], n=step.get("n"), device=device or step.get("device"))
        return {"kind": kind, "record": str(records / step["generate"]), "ok": bool(found.paths),
                "status": ", ".join(Path(path).name for path in found.paths)}
    if kind == "export":
        found = kalfa.api.export(records / step["export"], format=step.get("format", "state_dict"),
                                 device=device or step.get("device"))
        return {"kind": kind, "record": str(records / step["export"]), "ok": Path(found.path).is_file(),
                "status": Path(found.path).name}
    paths = [str(records / name) for name in step["summarize"]]
    written = []
    if step.get("out"):
        written.extend(str(path) for path in summary.write(summary.summarize(paths), records / step["out"]))
    if step.get("html"):
        written.append(str(reporting.write_report(paths, out=str(records / step["html"]))))
    if not written:
        summary.summarize(paths)
    return {"kind": kind, "record": ", ".join(step["summarize"]), "ok": True, "status": ", ".join(written)}


def play(folder, records, device=None, log=print):
    folder = Path(folder).resolve()
    records = Path(records).resolve()
    records.mkdir(parents=True, exist_ok=True)
    results = []
    for step in recipe_of(folder)["steps"]:
        try:
            result = play_step(step, folder, records, device, log)
        except Exception as failure:
            result = {"kind": kind_of(step), "record": None, "ok": False,
                      "status": f"{type(failure).__name__}: {failure}", "traceback": traceback.format_exc()}
        results.append(result)
        log(f"{folder.name}: {result['kind']} {result['status']}" + ("" if result["ok"] else "  FAILED"))
        if not result["ok"]:
            break
    return results


def fresh(directory):
    directory = Path(directory)
    if directory.exists():
        if not (directory / MARKER).exists():
            raise SystemExit(f"{directory} exists and was not made by demo.py; remove it or choose --out")
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    (directory / MARKER).write_text("made by examples/demo.py\n", encoding="utf-8")
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description="train every example briefly into a demo root for kalfa board")
    parser.add_argument("--out", default="runs/demo", help="the demo root, runs/demo by default")
    parser.add_argument("--only", nargs="*", help="example folder names to play, all by default")
    parser.add_argument("--device", help="cpu, mps, cuda or auto, over each config's device")
    args = parser.parse_args(argv)
    chosen = [folder for folder in folders() if not args.only or folder.name in args.only]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    failed = []
    for folder in chosen:
        prepare(folder)
        results = play(folder, fresh(out / folder.name), device=args.device)
        if not all(result["ok"] for result in results):
            failed.append(folder.name)
            print(results[-1].get("traceback", ""), file=sys.stderr)
    print(f"\n{len(chosen) - len(failed)} of {len(chosen)} examples played into {out}")
    if failed:
        print("failed: " + ", ".join(failed), file=sys.stderr)
    print(f"open them with: uv run kalfa board {out} --open")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
