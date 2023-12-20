for K in {1..30}
do
    echo "Computing MPC terminal in simple long for K=$K"
    ./bash_scripts/rec_2/mpc/rec_2_commodity_terminal_mpc.sh $K ${@:1}
done