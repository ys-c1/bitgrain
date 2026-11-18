#!/bin/bash
# Quadtree partition codec acceptance test.
# Builds and runs the self-contained demo in c/quadtree.c (no new
# third-party dependencies, no network access). Run from repo root:
#   bash tests/quadtree_test.sh
set -e
cd "$(dirname "$0")/.."

OUT_DIR="tests/out"
mkdir -p "$OUT_DIR"
BIN="$OUT_DIR/quadtree_demo"

echo "== compiling c/quadtree.c (single library file, libc+libm only) =="
cc -std=c11 -Wall -Wextra -Wpedantic -O2 -DBITGRAIN_QT_DEMO \
   c/quadtree.c -lm -o "$BIN"

echo "== running acceptance harness =="
"$BIN"
