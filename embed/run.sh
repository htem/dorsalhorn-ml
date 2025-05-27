#!/bin/bash

# Define your list of coordinates (x y z per line)
coords_list=(
    # "29483 2403 3917"
    # "25360 2484 2342"
    # "29042 2463 245"
    # "31483 2651 816"
    # "30349 2687 1634"
    # "23539 2617 1140"
    # "20095 2699 2026"
    # "36947 2823 4048"
    # "36281 2769 3146"
    # "31124 2480 3694"
    "10628 2709 1214"
    "8989 2416 2940"
    "13350 2971 4089"
    "23797 2542 201"
    "27049 2332 1591"
    "29483 2403 3917"
    "36985 3091 1702"
    "40462 3098 1781"
    "20036 2302 2113"
    "18850 2352 1102"
    "18157 2305 1819"
    "8112 2360 1940"
    "6459 2394 1804"
    "5729 2522 2500"
    "17587 2897 314"
    "25360 2484 2342"
    "25385 2181 2933"
    "29042 2463 245"
    "30053 2559 746"
    "34831 2769 917"
    "43824 3296 1227"
    "33996 2894 518"
    "31483 2651 816"
    "30349 2687 1634"
    "29597 2669 2589"
    "28813 2499 2569"
    "27034 2529 494"
    "22443 2792 709"
    "23539 2617 1140"
    "19942 2458 1313"
    "17915 2635 607"
    "17749 2228 2575"
    "20095 2699 2026"
    "22840 2206 1896"
    "21622 2334 2470"
    "19250 6181 1651"
    "31008 7278 3234"
    "42358 8222 1447"
)

# Config path
config_file="configs/config_inf_flatseg_0429.yaml"

MAX_JOBS=8  # Maximum concurrent cutout jobs

# Function to limit concurrent jobs
wait_for_jobs() {
    while [ "$(squeue -u "$USER" -n gen_cutout -h | wc -l)" -ge "$MAX_JOBS" ]; do
        echo "Waiting for available job slot... ($(date))"
        sleep 120
    done
}

for coord in "${coords_list[@]}"; do
    read -r x y z <<< "$coord"
    job_name="${x}_${y}_${z}"

    wait_for_jobs

    # Submit cutout job and get job ID
    cutout_job_id=$(sbatch --job-name="gen_cutout_${job_name}" --export=ALL,X=$x,Y=$y,Z=$z,config_file=$config_file gen_cutouts.job | awk '{print $4}')
    echo "Submitted gen_cutout_${job_name} with Job ID: $cutout_job_id"

    # Submit embed job dependent on cutout job
    sbatch --dependency=afterok:$cutout_job_id \
           --job-name="gen_embed_${job_name}" \
           --export=ALL,X=$x,Y=$y,Z=$z,config_file=$config_file \
           gen_embeds.job
    echo "Scheduled gen_embed_${job_name} dependent on Job ID: $cutout_job_id"
done
