#!/usr/bin/env bash
# Description Remove the background for an given AR video with existing alpha mask

w=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 "$1")
h=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$1")
fps=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$1")

# Calculate exact intermediate dimensions to avoid rounding mismatches
# 0.4x of original dimensions
sw=$((w * 4 / 10))
sh=$((h * 4 / 10))

echo "Width: $w, Height: $h, fps: $fps (Target Small: ${sw}x${sh})"

ffmpeg -i "$1" -i mask_eroded.png -i mask.png -filter_complex \
"[1:v]scale=$w:$h[mask]; \
[2:v]scale=$sw:$sh[mask_outher]; \
[0:v]split=8[base0][base1][base2][base3][base4][base5][size_ref][scaled_src]; \
[size_ref]geq=r=128:g=128:b=128,scale=$w:$h[black_scaled]; \
[scaled_src]copy[scaled]; \
[base0]crop=w=iw*0.2:h=ih*0.2:x=iw/2-iw*0.1:y=ih-ih*0.2[lt]; \
[base1]crop=w=iw*0.2:h=ih*0.2:x=iw/2-iw*0.1:y=0[lb]; \
[base2]crop=w=iw*0.1:h=ih*0.2:x=iw-iw*0.1:y=ih-ih*0.2[rtl]; \
[base3]crop=w=iw*0.1:h=ih*0.2:x=iw-iw*0.1:y=0[rbl]; \
[base4]crop=w=iw*0.1:h=ih*0.2:x=0:y=ih-ih*0.2[rtr]; \
[base5]crop=w=iw*0.1:h=ih*0.2:x=0:y=0[rbr]; \
[lt][lb]vstack=inputs=2[lhalf]; \
[rtl][rtr]hstack=inputs=2[rtop]; \
[rbl][rbr]hstack=inputs=2[rbottom]; \
[rtop][rbottom]vstack=inputs=2[rhalf]; \
[lhalf][rhalf]hstack=inputs=2,scale=$sw:$sh[s_mask]; \
[s_mask]split=2[s_mask1][s_mask2]; \
[s_mask2][mask_outher]alphamerge,split=2[masked_alpha1][masked_alpha2]; \
[s_mask1]scale=$w:$h:flags=neighbor,format=gray8[fullmask]; \
[mask][fullmask]blend=all_mode=and[fullmask2]; \
[scaled][fullmask2]alphamerge[vamasked]; \
[black_scaled][vamasked]overlay=format=auto[outv]; \
[masked_alpha1]crop=iw/2:ih:0:0,split=2[masked_alpha_l1][masked_alpha_l2]; \
[masked_alpha2]crop=iw/2:ih:iw/2:0,split=4[masked_alpha_r1][masked_alpha_r2][masked_alpha_r3][masked_alpha_r4]; \
[outv][masked_alpha_l1]overlay=W*0.5-w*0.5:-0.5*h[out_lt]; \
[out_lt][masked_alpha_l2]overlay=W*0.5-w*0.5:H-0.5*h[out_tb]; \
[out_tb][masked_alpha_r1]overlay=0-w*0.5:-0.5*h[out_l_lt]; \
[out_l_lt][masked_alpha_r2]overlay=0-w*0.5:H-0.5*h[out_tb_ltb]; \
[out_tb_ltb][masked_alpha_r3]overlay=W-w*0.5:-0.5*h[out_r_lt]; \
[out_r_lt][masked_alpha_r4]overlay=W-w*0.5:H-0.5*h[outv2]" \
-map "[outv2]" -map "0:a?" -c:a copy "$1-nogb.mp4"
