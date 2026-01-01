#!/usr/bin/env bash

ffmpeg -i "$1" -filter_complex "[0:v]split=6[base0][base1][base2][base3][base4][base5];[base0]crop=w=iw*0.2:h=ih*0.2:x=iw/2-iw*0.1:y=ih-ih*0.2[lt];[base1]crop=w=iw*0.2:h=ih*0.2:x=iw/2-iw*0.1:y=0[lb];[base2]crop=w=iw*0.1:h=ih*0.2:x=iw-iw*0.1:y=ih-ih*0.2[rtl];[base3]crop=w=iw*0.1:h=ih*0.2:x=iw-iw*0.1:y=0[rbl];[base4]crop=w=iw*0.1:h=ih*0.2:x=0:y=ih-ih*0.2[rtr];[base5]crop=w=iw*0.1:h=ih*0.2:x=0:y=0[rbr];[lt][lb]vstack=inputs=2[lhalf];[rtl][rtr]hstack=inputs=2[rtop];[rbl][rbr]hstack=inputs=2[rbottom];[rtop][rbottom]vstack=inputs=2[rhalf];[lhalf][rhalf]hstack=inputs=2[s_mask];[s_mask]scale=iw*2.5:ih*2.5,format=gray8[fullmask]" -map "[fullmask]:v" -map "0:a?" -c:a copy "$1-mask.mp4"


