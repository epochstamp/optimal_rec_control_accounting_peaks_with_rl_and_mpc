for K in {1..30}
do
    echo "Computing MPC in simple long for K=$K"
    ./rec_6_from_rec_28_data_mpc.sh $K ${@:2}
done