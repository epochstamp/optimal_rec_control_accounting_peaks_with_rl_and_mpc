for K in {1..30}
do
    echo "Computing MPC terminal in rec 6 for K=$K"
    ./bash_scripts/rec_6/mpc/rec_6_commodity_terminal_mpc.sh $K ${@:1}
done