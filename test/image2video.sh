#!/usr/bin/env bash

ffmpeg -loop 1 -i "$1" -t 5 -r 1 -c:v libx265 -crf 16 -pix_fmt yuv420p "$1-video.mp4"
