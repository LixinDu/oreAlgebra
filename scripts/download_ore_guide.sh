#!/usr/bin/env bash
set -euo pipefail

URL="${1:-https://www3.risc.jku.at/research/combinat/software/ore_algebra/main.pdf}"
OUT="${2:-data/ore_algebra_guide.pdf}"

mkdir -p "$(dirname "$OUT")"

if command -v curl >/dev/null 2>&1; then
  curl -fL "$URL" -o "$OUT"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$OUT" "$URL"
else
  echo "ERROR: neither curl nor wget is installed." >&2
  exit 1
fi

echo "Downloaded: $OUT"
