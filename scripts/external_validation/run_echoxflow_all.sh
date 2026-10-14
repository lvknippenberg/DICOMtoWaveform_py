#!/usr/bin/env bash
# Primary EchoXFlow scoring run + sensitivity runs, then the markdown summary.
set -e
cd /c/Claude/Doppler/validation
PY=/c/Users/p70096100/AppData/Local/miniconda3/envs/echoxflow/python.exe
$PY score_echoxflow.py > echoxflow_out/log_primary.txt 2>&1
$PY score_echoxflow.py --black 60 > echoxflow_out/log_black60.txt 2>&1
$PY score_echoxflow.py --black 80 > echoxflow_out/log_black80.txt 2>&1
$PY score_echoxflow.py --black 0 > echoxflow_out/log_black0.txt 2>&1
$PY score_echoxflow.py --native > echoxflow_out/log_native.txt 2>&1
$PY score_echoxflow.py --no-offset-fix > echoxflow_out/log_nooffsetfix.txt 2>&1
$PY summarize_echoxflow.py > /dev/null
echo "all runs done"
