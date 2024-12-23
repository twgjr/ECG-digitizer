#!/bin/bash

echo "Running image making script over groups"

# Total number of groups
total_groups=22
groups_per_batch=4

# Outer loop to process batches
for start_group in $(seq 0 $((groups_per_batch)) $((total_groups - 1)))
do
    end_group=$((start_group + groups_per_batch - 1))
    if [ $end_group -ge $total_groups ]; then
        end_group=$((total_groups - 1))
    fi

    for i in $(seq $start_group $end_group)
    do
        echo "Processing group $i"
        python image_maker.py --group $i --title 'base' &
    done

    wait
    echo "Completed groups $start_group to $end_group"
done

echo "Completed all groups"
