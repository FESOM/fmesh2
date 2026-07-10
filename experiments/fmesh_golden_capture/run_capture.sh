#!/bin/bash
# One-time capture of legacy fmesh golden fixtures (Task 19 prerequisite).
#
# Prerequisites:
#   - micromamba env `fmesh_legacy` from ../fmesh/requirements.yml, plus two
#     fixes the solver gets wrong: `openssl>=3` (netCDF4 runtime) and
#     `fastkml=0.12` (fmesh uses the old callable `.features()` API)
#   - ../fmesh/topo and ../fmesh/coastlines populated via ../fmesh/download_data.sh
#   - shipped coastline cache pkls copied to the underscore names the code
#     actually looks up: "min_length=100km_averaging=50km.pkl"
#
# Usage: ./run_capture.sh <workdir>
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
FMESH2_ROOT="$(cd "$HERE/../.." && pwd)"
FMESH="$(cd "$FMESH2_ROOT/../fmesh" && pwd)"
WORK="${1:?usage: run_capture.sh <workdir>}"

mkdir -p "$WORK/jigsaw"
cd "$WORK"
ln -sfn "$FMESH/topo" topo
ln -sfn "$FMESH/coastlines" coastlines
ln -sfn "$FMESH/kml" kml
cp "$FMESH/fmesh.py" "$FMESH/utils.py" .
cp "$HERE/configure.yaml" .

micromamba run -n fmesh_legacy python fmesh.py 2>&1 | tee run.log

micromamba run -n fmesh_legacy python "$HERE/convert_outputs.py" \
    "$WORK" "$FMESH2_ROOT/tests/fixtures/fmesh_golden"
cp run.log "$FMESH2_ROOT/tests/fixtures/fmesh_golden/run.log"
cp "$HERE/configure.yaml" "$FMESH2_ROOT/tests/fixtures/fmesh_golden/configure.yaml"
echo CAPTURE_DONE
