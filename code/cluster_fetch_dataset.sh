#!/bin/bash
# Pull the full PushT-FR3 dataset onto cluster storage.
#
# A single curl gets ~0.45 MB/s to the HF LFS CDN from inside the pod (~6.5 h for
# 10.4 GB), but 8-16 concurrent range requests aggregate to ~9 MB/s. Chunks are
# fetched independently with size verification and retries, then concatenated,
# so a dropped connection costs one 64 MB chunk instead of the whole transfer.
set -u

URL="https://huggingface.co/datasets/Rongxuan-Zhou/pusht_lewm_fr3/resolve/main/pusht_lewm_fr3.h5"
ROOT=/mnt/group/jxdong/wm_exp/data
OUT=$ROOT/pusht_lewm_fr3.h5
TMP=$ROOT/dl
SIZE=10442118656
CHUNK=67108864
PAR=16

N=$(( (SIZE + CHUNK - 1) / CHUNK ))
mkdir -p "$TMP"
rm -f "$TMP/failures.txt"
echo "size=$SIZE chunks=$N parallel=$PAR"

fetch() {
  local i=$1
  local A=$((i * CHUNK)) B
  B=$((A + CHUNK - 1))
  [ $B -ge $SIZE ] && B=$((SIZE - 1))
  local EXP=$((B - A + 1))
  local f
  f="$TMP/c_$(printf %05d "$i")"
  for try in 1 2 3 4 5; do
    if [ -f "$f" ] && [ "$(stat -c%s "$f")" -eq "$EXP" ]; then
      return 0
    fi
    curl -sS -L -r "$A-$B" -o "$f" --max-time 600 "$URL" 2>/dev/null
    if [ -f "$f" ] && [ "$(stat -c%s "$f")" -eq "$EXP" ]; then
      return 0
    fi
    sleep $((try * 3))
  done
  echo "FAIL $i" >> "$TMP/failures.txt"
  return 1
}
export -f fetch
export URL TMP SIZE CHUNK

START=$(date +%s)
seq 0 $((N - 1)) | xargs -P $PAR -I{} bash -c 'fetch {}'
ELAPSED=$(( $(date +%s) - START ))
echo "fetch done in ${ELAPSED}s"

if [ -f "$TMP/failures.txt" ]; then
  echo "FAILED CHUNKS:"; cat "$TMP/failures.txt"; exit 1
fi

# verify every chunk before spending time on concatenation
BAD=0
for i in $(seq 0 $((N - 1))); do
  A=$((i * CHUNK)); B=$((A + CHUNK - 1))
  [ $B -ge $SIZE ] && B=$((SIZE - 1))
  EXP=$((B - A + 1))
  f="$TMP/c_$(printf %05d "$i")"
  if [ ! -f "$f" ] || [ "$(stat -c%s "$f")" -ne "$EXP" ]; then
    echo "BAD $i"; BAD=1
  fi
done
[ $BAD -eq 1 ] && exit 1

cat "$TMP"/c_* > "$OUT"
GOT=$(stat -c%s "$OUT")
echo "concat: $GOT / $SIZE"
[ "$GOT" -ne "$SIZE" ] && { echo "SIZE MISMATCH"; exit 1; }

python3 - <<'PY'
import h5py
p = "/mnt/group/jxdong/wm_exp/data/pusht_lewm_fr3.h5"
with h5py.File(p, "r") as f:
    def show(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  {name}: {obj.shape} {obj.dtype}")
    f.visititems(show)
    print("  attrs:", dict(f.attrs))
print("HDF5 OK")
PY

rm -rf "$TMP"
echo "ALL DONE -> $OUT"
