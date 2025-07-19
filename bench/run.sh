start_time=$(date +%s)

./robtop -jsonfile=$2 -meshfile=$1 -outdir=./result/test/ -power_penalty=3 -volume_ratio=0.4 -filter_radius=2 -gridreso=64 -damp_ratio=0.5 -shell_width=1 -workmode=wscf -poisson_ratio=0.4 -design_step=0.06 -vol_reduction=0.05 -min_density=1e-3 -nologdensity -nologcompliance  -usespinodal=true -testname=targetc -target_compliance=15

end_time=$(date +%s)
cost_time=$[ $end_time-$start_time ]
echo "build kernel time is $(($cost_time/60))min $(($cost_time%60))s"