
#OUTPUT_DIRECTORY=data
#mkdir -p $OUTPUT_DIRECTORY
#OUTPUT_FILE=$OUTPUT_DIRECTORY/measurements_`date +%R`.txt

#touch $OUTPUT_FILE

OUTPUT_FILE=`data.txt`

for i in 128 256 512 1024 2048; do
    for ker in 3 5 7; do
        echo "Size: $i" >> data.txt;
        echo "kernel: $ker" >> data.txt;
        ../build/edge_cpu --WC=$i --HC=$i --WK=$ker >> data.txt;
    done ;
done
