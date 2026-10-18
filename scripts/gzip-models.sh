#!/bin/sh
# Trunk post-build hook: gzips the OBJ models the scenes use into the web build
# as models/<file name>.gz, where the web build downloads them from
# (src/engine/embedded_assets.rs).
set -eu

out="$TRUNK_STAGING_DIR/models"
mkdir -p "$out"
grep -h '^[[:space:]]*file:' assets/scenes/*.yaml |
    sed 's/^[[:space:]]*file:[[:space:]]*//' |
    while read -r file; do basename "$file"; done |
    sort -u |
    while read -r name; do
        gzip -c "assets/models/$name" > "$out/$name.gz"
    done
