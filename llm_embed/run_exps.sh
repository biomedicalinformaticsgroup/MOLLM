#!/bin/bash

# Path to the Python script
SCRIPT_PATH="main.py"

# List of start_row_num values
start_indices=(0 1000 2000 3000)

# List of event_id values
event_ids=('V06') # ('BL' 'V02' 'V04')

# List of section values
sections=('2' '3' '23' '1' '4') # ('23') # (2 3 1 4)

# Nested loops with start_row_num in the innermost loop
for event_id in "${event_ids[@]}"; do
    for section in "${sections[@]}"; do
        for start_row in "${start_indices[@]}"; do
            echo "Running with event_id=${event_id}, section=${section}, start_row_num=${start_row}..."
            python $SCRIPT_PATH dataset.event_id=${event_id} dataset.section=${section} dataset.start_row_num=${start_row}
            
            if [ $? -ne 0 ]; then
                echo "Script failed for event_id=${event_id}, section=${section}, start_row_num=${start_row}. Skipping to next iteration."
                continue  # Skip to the next start_row_num
            fi
            
            echo "Completed run with event_id=${event_id}, section=${section}, start_row_num=${start_row}."
        done
    done
done

echo "All runs completed (with potential failures skipped)."