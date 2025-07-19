#!/bin/bash

base_dir="./result/dataset"
dst_dir=${base_dir}/"data"
param="cminmax.txt"
vol_file="vrec.txt"
cnt=2001
for i in {201..300};do
    src=${base_dir}/${i}
    cmin=$(head -n 1 "${src}/${param}" | awk '{print $1}')
    cmax=$(head -n 2 "${src}/${param}" | tail -n 1 | awk '{print $1}')
    vol=$(head -n 1 "${src}/${vol_file}" | awk '{print $1}')

    if [ "$(echo "$cmin>-1000 && $cmax<1000" | bc -l)" -eq 1 ] ;then
        step=$(echo "scale=10;($cmax-$cmin)/9" | bc)
        for j in {1..10};do
            target=$(echo "scale=10; ($j-1)*$step+$cmin" | bc)
            dst=${dst_dir}/${cnt}/
            mkdir -p "$dst"
            ((cnt++))
            echo $dst
            echo $target
            echo $cmin
            echo $cmax
            ./robtop -jsonfile=femur/config3.json -meshfile=femur/femur.obj -outdir="$dst" -power_penalty=3 -volume_ratio="$vol" -filter_radius=2 -gridreso=128 -damp_ratio=0.5 -shell_width=1 -workmode=wscf -poisson_ratio=0.4 -design_step=0.06 -vol_reduction=0.05 -min_density=1e-3 -nologdensity -nologcompliance  -usespinodal=true -testname=targetc -target_compliance="$target"
        done
    fi
done