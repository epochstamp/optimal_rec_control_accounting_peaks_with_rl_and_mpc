for K in {1..30}
do
    echo "Computing MPC in simple long without peaks for K=$K"
    ./simple_long_mpc_without_peaks.sh $K ${@:1}
done