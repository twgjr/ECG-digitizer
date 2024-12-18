#!/bin/bash

# this assumes 8 cores and 32GB RAM

echo "Running image making script over batches"

# Loop from 0 to 21
# for i in {0..7}
# do
#     echo "Processing batch $i"
#     python image_maker.py --batch $i &
# done

# wait

for i in {8..15}
do
    echo "Processing batch $i"
    python image_maker.py --batch $i &
done

wait

for i in {16..21}
do
    echo "Processing batch $i"
    python image_maker.py --batch $i &
done

wait

echo "Completed all batches"