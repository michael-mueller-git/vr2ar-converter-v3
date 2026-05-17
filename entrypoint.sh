#!/usr/bin/env sh

for d in /nix/store/*ffmpeg-[0-9]*-bin/; do
    [ -x "$d/bin/ffmpeg" ] || continue
    ln -sf "$d/bin/ffmpeg" /usr/local/bin/ffmpeg
    ln -sf "$d/bin/ffprobe" /usr/local/bin/ffprobe
    break
done

python3 -u /app/main.py
