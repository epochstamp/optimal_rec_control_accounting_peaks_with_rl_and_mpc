for K in {1..30}
do
    echo "Computing MPC terminal in simple long without peaks for K=$K"
    ./simple_long_mpc_terminal_without_peaks.sh $K ${@:1}
done