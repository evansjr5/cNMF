data_dir=/home/james/data/immune_exclusion_data/seperate_sample_data
output_dir=/home/james/data/cNMF_out/per_sample_cnmf

for file in "$data_dir"/*; do
    echo "$file"
    sample_prefix=$(basename "$file" | cut -d'/' -f1 | cut -d'.' -f1)
    sample_name=${sample_prefix}_cNMF
    
    echo "Starting prepare command:"
    cnmf prepare "$file" \
    --output-dir "$output_dir" \
    --name "$sample_name" \
    -k 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33 \
    -n 30 \
    -j 1
    echo "Finished prepare command!"

    echo "Starting Factorization:"
    cnmf factorize \
    --output-dir "$output_dir" \
    --name "$sample_name" \
    --n-jobs 1 \
    --worker-index 0
    echo "Finished Factorization!"
    
    echo "Combine and plot:"
    cnmf combine \
    --output-dir "$output_dir" \
    --name "$sample_name" \
    -k 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33

    cnmf k_selection_plot \
    --output-dir "$output_dir" \
    --name "$sample_name"

    echo "Finished prepare, factorize, combine, and k-selection plot"

done
