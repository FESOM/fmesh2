"""M4 acceptance driver: the full editing workflow on core2.

Orchestrates the plan's Task 26 scenario end-to-end against a live server:

1. serve core2 through the real `fmesh2` CLI (built frontend);
2. locally refine the Denmark Strait via the API (RTopo coastal land cut);
3. drive the real browser UI (`frontend/e2e/m4_driver.mjs`, Playwright
   chromium): brush-select coastal nodes in the refined strait, drag them
   (tangent-plane move commit), smooth, undo/redo, save-as — screenshots
   land in docs/acceptance/figures/;
4. validate the saved artifacts: FESOM triplet readable, journal replay on
   the pristine refined mesh reproduces the saved mesh hash bit-exact.

Run from the repo root:
    micromamba run -n meshgen python experiments/m4_acceptance/runner.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
CORE2 = REPO.parent / "core2"
RTOPO = REPO.parent / "fmesh" / "topo" / "RTopo-2.0.1_30sec_bos_fix_lowres_D3.nc"
PORT = int(os.environ.get("FMESH2_M4_PORT", 8323))
BASE = f"http://127.0.0.1:{PORT}"

# Denmark Strait box (Greenland/Iceland coasts included -> coastal cavity)
STRAIT = [[-32.0, 64.5], [-20.0, 64.5], [-20.0, 69.5], [-32.0, 69.5]]
TARGET_KM = 55.0


def wait_server(timeout: float = 60.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            httpx.get(f"{BASE}/api/meshes", timeout=2.0).raise_for_status()
            return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError("server did not come up")


def refine_strait(client: httpx.Client, out_dir: Path) -> tuple[str, dict]:
    mesh_id = client.get("/api/meshes").json()[0]["id"]
    r = client.post(
        f"/api/mesh/{mesh_id}/refine",
        json={
            "polygon": STRAIT,
            "target_km": TARGET_KM,
            "topo": str(RTOPO),
            "out_dir": str(out_dir),
        },
    )
    r.raise_for_status()
    job = r.json()["job_id"]
    while True:
        snap = client.get(f"/api/jobs/{job}").json()
        if snap["status"] != "running":
            break
        time.sleep(0.5)
    assert snap["status"] == "done", snap
    return snap["result"]["mesh_id"], snap["result"]["report"]


def validate_saved(refined_dir: Path, saved_dir: Path) -> dict:
    sys.path.insert(0, str(REPO / "src"))
    from fmesh2.core.io_fesom import read_fesom
    from fmesh2.edit.journal import Journal, apply_journal, mesh_hash

    saved = read_fesom(saved_dir)
    journal = Journal(saved_dir / "edits.jsonl")
    replayed = apply_journal(read_fesom(refined_dir), saved_dir / "edits.jsonl")
    return {
        "saved_nodes": saved.n_nodes,
        "saved_tri": saved.n_tri,
        "journal_active_ops": len(journal.ops),
        "replay_hash_matches": mesh_hash(replayed) == mesh_hash(saved),
    }


def main() -> int:
    assert (CORE2 / "nod2d.out").exists(), f"core2 not found at {CORE2}"
    assert RTOPO.exists(), f"RTopo not found at {RTOPO}"
    work = Path(tempfile.mkdtemp(prefix="fmesh2_m4_"))
    refined_dir = work / "core2_strait55"
    saved_dir = work / "core2_strait55_edited"
    figures = REPO / "docs" / "acceptance" / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    server = subprocess.Popen(
        ["micromamba", "run", "-n", "meshgen", "fmesh2", str(CORE2),
         "--port", str(PORT), "--no-browser"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_server()
        with httpx.Client(base_url=BASE, timeout=600.0) as client:
            t0 = time.time()
            refined_id, report = refine_strait(client, refined_dir)
            print(f"REFINE ok in {time.time() - t0:.1f}s -> {refined_id}")
            print("REFINE_REPORT " + json.dumps(report))

        env = os.environ | {
            "FMESH2_M4_BASE": BASE,
            "FMESH2_M4_MESH": refined_id,
            "FMESH2_M4_OUT": str(saved_dir),
            "FMESH2_M4_FIGDIR": str(figures),
        }
        driver = subprocess.run(
            ["node", "e2e/m4_driver.mjs"],
            cwd=REPO / "frontend",
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        print(driver.stdout)
        if driver.returncode != 0:
            print(driver.stderr, file=sys.stderr)
            return 1

        checks = validate_saved(refined_dir, saved_dir)
        print("VALIDATE " + json.dumps(checks))
        assert checks["replay_hash_matches"], "journal replay hash mismatch!"
        assert checks["journal_active_ops"] >= 2, "expected move+smooth in journal"
        print(f"M4 ACCEPTANCE PASSED (artifacts in {work})")
        return 0
    finally:
        server.terminate()
        server.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
