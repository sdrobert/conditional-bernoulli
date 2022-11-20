#! /usr/bin/env bash

source ./scripts/slurm/common.sh

check_and_do "$cpu_opts" -s 1 -i "$(cd "$timit_dir"; pwd -P)" || exit 1
(sleep 1 && check_and_do "$gpu_opts" -s 2) & lm_pid=$!
(sleep 1 && check_and_do "$gpu_opts --gpus-per-task=2" -s 3 -z full -f 2) & enc_full_pid=$!
(sleep 1 && check_and_do "$gpu_opts" -s 3 -z partial) & enc_indep_pid=$!
(sleep 1 && check_and_do "$gpu_opts" -s 4) & pcb_pid=$!
v=0
wait $lm_pid; ((v+=$?)) || true
wait $enc_full_pid; ((v+=$?)) || true
wait $enc_indep_pid; ((v+=$?)) || true
wait $pcb_pid; ((v+=$?)) || true
((v>0)) && exit 1
