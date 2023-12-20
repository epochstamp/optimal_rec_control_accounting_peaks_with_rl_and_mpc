for K in {1..30}
do
    echo "Computing MPC in simple long without peaks for K=$K"
    ./rec_28_summer_end_mpc_without_peaks.sh $K ${@:2}
done