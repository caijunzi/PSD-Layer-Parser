"""Run every input against an isolated adaptive store; retain all evidence."""
import os
import sys
import json
import hashlib
import sqlite3
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get("ULS_E2E_OUT", str(ROOT / "outputs" / "adaptive-e2e-20260915-final")))
CASES = [
    ("damask_sample.png", "textile_damask", "plate", "壁布"),
    ("金地屏风_江户芦雁寒林六曲_绫边装裱.jpg", "japanese_screen_gold", "plate", "金地原图"),
    ("金地屏风_钓舟芦雁仿琳派_深框商用图.png", "japanese_screen_gold", "plate", "金地商用图"),
    ("青绿山水_松亭瀑布渔舟_纸本设色.jpeg", "chinese_ink_landscape_ai", "design", "宋代水墨"),
    ("青绿山水_草亭雅集秋景_纸本设色.jpeg", "chinese_ink_landscape_ai", "design", "明代水墨"),
    ("纸本水墨_平远枯树亭阁_淡设色.png", "chinese_ink_landscape_ai", "design", "元代水墨"),
    ("西洋壁画_乔托风圣母圣人群像_湿壁画金底.jpeg", "western_oil_painting", "design", "油画"),
    # 绢本工笔（花鸟题材：牡丹/枝叶/禽鸟/山石/水面）与壁布 preset 的类目
    # （巴洛克团花/金箔卷草纹样）完全不匹配 —— 2026-09-16 实测：用 textile_damask
    # 跑绢本时 AI 检测对壁布类目零产出、规则引擎退回屏风系，
    # ④内容承载丢 8.712%/20.004%、⑤合成 RMSE 26.7/44.6 均失败。
    # 改用 chinese_ink_landscape_ai（含 trees_vegetation / fauna_geese / plum_blossom /
    # mountains_cliffs / water_ripples 等花鸟+山水类目）后 8 维全过：
    # ④ lost_ratio 8.712% → 0.004%，⑤ rmse_lowfreq 26.72 → 1.86。
    ("绢本工笔_牡丹双雀_仿古绢底.jpeg", "chinese_ink_landscape_ai", "design", "绢本工笔真值 1"),
    ("绢本工笔_锦鸡鸳鸯松石_仿古绢底.jpeg", "chinese_ink_landscape_ai", "design", "绢本工笔真值 2"),
]

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    production = ROOT / "webui/data/adaptive_semantics.db"
    production_hash = sha(production)
    results = []
    from PIL import Image
    for i, (name, preset, mode, kind) in enumerate(CASES, 1):
        case = OUT / f"case-{i:02d}"
        case.mkdir(exist_ok=True)
        db = case / "semantics.db"
        if not db.exists():
            with sqlite3.connect(f"file:{production.as_posix()}?mode=ro", uri=True) as src:
                with sqlite3.connect(db) as dst:
                    src.backup(dst)
        env = dict(os.environ, PYTHONIOENCODING="utf-8", ULS_AUDIT_AFTER_RUN="1", ULS_AUDIT_STRICT="1",
                   ADAPTIVE_DB_PATH=str(db), ADAPTIVE_EPISODE_PATH=str(case / "episodes.jsonl"),
                   ADAPTIVE_INDEX_PATH=str(case / "index.pkl"), ADAPTIVE_PENDING_PATH=str(case / "pending.jsonl"),
                   ADAPTIVE_LEARNING_TASKS_PATH=str(case / "learning.jsonl"), TASK_ID=f"e2e-case-{i:02d}")
        source = ROOT / "inputs" / name
        rec = {"case": i, "sample": name, "type": kind, "preset": preset, "product_line": mode,
               "input_sha256": sha(source), "input_size": Image.open(source).size, "adaptive_mode": "hybrid",
               "scale": 1, "runs": []}
        for phase in ("cold", "repeat"):
            psb = case / f"{phase}.psb"
            log = case / f"{phase}.log"
            cmd = [sys.executable, "-u", "run_universal_engine.py", "--input", str(source), "--preset", preset,
                   "--mode", mode, "--scale", "1", "--adaptive-mode", "hybrid", "--output", str(psb)]
            started = time.monotonic()
            print(f"START {i}/{len(CASES)} {name} {phase}", flush=True)
            with log.open("wb") as stream:
                p = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            audit_file = psb.with_suffix(".audit.json")
            report = json.loads(audit_file.read_text(encoding="utf-8")) if audit_file.exists() else None
            episodes_file = case / "episodes.jsonl"
            episodes = [json.loads(s) for s in episodes_file.read_text(encoding="utf-8").splitlines() if s.strip()] if episodes_file.exists() else []
            ep = episodes[-1] if episodes else {}
            log_text = log.read_text(encoding="utf-8", errors="replace")
            cbr_hit_line = next((line for line in log_text.splitlines() if "CBR 命中（白名单通过）" in line), None)
            cbr_apply_line = next((line for line in log_text.splitlines() if "CBR 已应用：" in line), None)
            run = {"phase": phase, "exit_code": p.returncode, "seconds": round(time.monotonic() - started, 2),
                   "psb_exists": psb.exists(), "psb_sha256": sha(psb) if psb.exists() else None,
                   "audit_passed": report.get("passed") if report else None,
                   "audit_issues": report.get("issues") if report else ["未生成审计报告"],
                   "episode_id": ep.get("episode_id"), "detections": len(ep.get("dino_detections", [])),
                   "cbr": {"hit": cbr_hit_line is not None, "hit_line": cbr_hit_line,
                           "applied": cbr_apply_line is not None, "apply_line": cbr_apply_line,
                           "episode_reused": "cbr_reused=1" in json.dumps(ep, ensure_ascii=False)},
                   "artifact": str(psb), "audit": str(audit_file), "log": str(log)}
            saved_env = {k: os.environ.get(k) for k in env if k.startswith("ADAPTIVE_")}
            os.environ.update({k:v for k,v in env.items() if k.startswith("ADAPTIVE_")})
            try:
                from webui.backend.core.background_learner import BackgroundLearner
                worker = BackgroundLearner()
                if ep:
                    worker.enqueue_learning_task(ep["episode_id"], trigger="e2e")
                    worker.drain_once()
                    status = worker.get_task_status(ep["episode_id"])
                    run["learning_status"] = status
                worker.stop_worker()
            except Exception as exc:
                run["learning_error"] = repr(exc)
            finally:
                for k,v in saved_env.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
            rec["runs"].append(run)
            (case / "result.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"END {i}/{len(CASES)} {phase} exit={p.returncode} audit={run['audit_passed']} detections={run['detections']} cbr_hit={run['cbr']['hit']} cbr_applied={run['cbr']['applied']}", flush=True)
        rec["byte_reproducible"] = rec["runs"][0]["psb_sha256"] is not None and rec["runs"][0]["psb_sha256"] == rec["runs"][1]["psb_sha256"]
        results.append(rec)
        (OUT / "results.json").write_text(json.dumps({"production_db_unchanged": production_hash == sha(production), "cases": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("ALL_CASES_FINISHED", flush=True)

if __name__ == "__main__":
    main()
