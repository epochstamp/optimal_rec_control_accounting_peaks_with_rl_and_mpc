for K in {1..30}
do
    echo "Computing MPC terminal in simple long for K=$K"
    ./bash_scripts/rec_28_summer_end/mpc/rec_28_summer_end_mpc_terminal.sh $K ${@:1}
done