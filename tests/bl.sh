#!/usr/bin/env bash
# Blender 内テストを実行：tests/bl.sh <module> [test ...]
B="${BLENDER:-/c/Program Files/Blender Foundation/Blender 5.0/blender.exe}"
cd "$(dirname "$0")/.."
mkdir -p tests/out
"$B" -b --factory-startup --python tests/blender/run.py -- "$@" > "tests/out/$1.log" 2>&1
code=$?
grep -E "^(PASS|FAIL|RESULT)|Error|assert" "tests/out/$1.log" | tail -40
exit $code
