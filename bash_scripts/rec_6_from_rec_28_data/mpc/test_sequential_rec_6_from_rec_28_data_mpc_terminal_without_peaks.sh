for K in {1..30}
do
    echo "Computing MPC terminal in simple long without peaks for K=$K"
    ./rec_6_from_rec_28_data_mpc_terminal_without_peaks.sh $K ${@:2}
done