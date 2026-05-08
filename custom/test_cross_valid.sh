#!/usr/bin/env bash
for i in 1 2 3 4 5 6 7 8 9 10 11 12
do
    python tools/test.py /srv/fenster/people/Ningtao/Project/USVideo/tracking/mmtracking/work_dirs/mask_selsa_faster_rcnn_r50_dc5_1x_c04_needle1020_filter_aggre/mask_selsa_faster_rcnn_r50_dc5_1x_c04_needle1020_filter_aggre.py --checkpoint /srv/fenster/people/Ningtao/Project/USVideo/tracking/mmtracking/work_dirs/mask_selsa_faster_rcnn_r50_dc5_1x_c04_needle1020_filter_aggre/epoch_$i.pth --eval bbox segm
done