
#OUTPUT_DIRECTORY=data
#mkdir -p $OUTPUT_DIRECTORY
#OUTPUT_FILE=$OUTPUT_DIRECTORY/measurements_`date +%R`.txt

#touch $OUTPUT_FILE

#OUTPUT_FILE=`data.txt`

for i in 128 256 512 1024 2048; do
    for ker in 3 5 ; do
        echo "Size: $i" >> data_gpu.txt;
        echo "kernel: $ker" >> data_cpu.txt;
        ../build/edge_cpu --WC=$i --HC=$i --WK=$ker >> data_cpu.txt;
    done ;
done

for i in 128 256 512 1024 2048; do
    for ker in 3 5; do
        for cho in 1 2; do
            echo "Size: $i" >> data_gpu.txt;
            echo "kernel: $ker" >> data_gpu.txt;
            ../build/edge_cuda --WC=$i --HC=$i --WK=$ker --choice=$cho>> data_gpu.txt;
        done;
    done ;
done
