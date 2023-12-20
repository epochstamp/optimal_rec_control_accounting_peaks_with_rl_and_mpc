for K in {1..30}
do
    echo "Computing MPC terminal in rec 6 for K=$K"
    ./bash_scripts/rec_6_from_rec_28_data/mpc/rec_6_mpc_terminal.sh $K ${@:1}
done