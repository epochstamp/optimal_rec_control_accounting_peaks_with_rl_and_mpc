for K in {1..30}
do
    echo "Computing MPC in simple long for K=$K"
    ./rec_28_summer_end_mpc.sh $K ${@:2}
done