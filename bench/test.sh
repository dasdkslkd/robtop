#!/bin/bash

cmin=0
cmax=100

# if ! command -v bc &> /dev/null; then
#     echo "错误：需要 bc 命令进行数学计算" >&2
#     exit 1
# fi

for i in $(seq $cmin 1 $cmax); do
    # 计算目标合规性
    target_compliance=$(echo "scale=10; $i / 100" | bc)
    echo $target_compliance
    
    # 执行命令
    # ./robtop -jsonfile=$2 -meshfile=$1 -outdir=./result/test/ -power_penalty=3 -volume_ratio=0.4 -filter_radius=2 -gridreso=128 -damp_ratio=0.5 -shell_width=1 -workmode=wscf -poisson_ratio=0.4 -design_step=0.06 -vol_reduction=0.05 -min_density=1e-3 -logdensity -nologcompliance  -usespinodal=true -testname=targetc -target_compliance=$target_compliance
done