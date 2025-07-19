#!/bin/bash

base_dir="./result/dataset"
vmin=0.3
vmax=0.8
n=100
step=$(echo "scale=10;($vmax-$vmin)/($n-1)" | bc)

for i in {101..200}; do
    target=$(echo "scale=10; ($i-101)*$step+$vmin" | bc)
    dir=${base_dir}/${i}/
    if ! mkdir -p "$dir"; then
        echo "错误：无法创建目录 $dir" >&2
        exit 1
    fi
    ./robtop -jsonfile=femur/config2.json -meshfile=femur/femur.obj -outdir="$dir" -power_penalty=3 -volume_ratio="$target" -filter_radius=2 -gridreso=128 -damp_ratio=0.5 -shell_width=1 -workmode=wscf -poisson_ratio=0.4 -design_step=0.06 -vol_reduction=0.05 -min_density=1e-3 -nologdensity -nologcompliance  -usespinodal=true -testname=testspinodalopt -target_compliance=15
done