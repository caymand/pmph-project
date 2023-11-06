NRUNS=25

for SIZE in 128 256 512 1024 2048 3072 4096 5120 6144 7168 8128 9600
do    
    ./main $NRUNS $SIZE |grep "tensor optimized" -A 12 |awk 'END{print $NF}'
done