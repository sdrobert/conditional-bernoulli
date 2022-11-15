#! /usr/bin/env bash

set -e

mkdir -p logs/timit

timit_dir="${1:-"$HOME/Databases/TIMIT"}"
cpu_opts="${2:-"-p cpu"}"
gpu_opts="${3:-"-p p100 --gpus-per-task=1"}"

if [ ! -d "$timit_dir" ]; then
  echo "$timit_dir is not a directory"
  exit 1
fi

check_and_do() {
  local opts="$1"
  shift
  local leftover="$(./timit.sh "$@" -xxj)"
  if ((leftover>0)); then
    echo "Starting: ./timit.sh $* -xqj"
    sbatch $opts -a 1-$leftover -W scripts/slurm/timit_wrapper.sh "$@" -xqj
    v=$?
    if [ $? -eq 0 ]; then
      echo "./timit.sh $* -xqj succeeded"
    else
      echo "./timit.sh $* -xqj failed"
    fi
    return $?
  else
    echo "./timit.sh $* -xqj already done"
  fi
}

check_and_do "$cpu_opts --ntasks=1" -s 1 -i "$(cd "$timit_dir"; pwd -P)"
check_and_do "$gpu_opts --ntasks=1" -s 2

( 
  sleep 1
  check_and_do "$gpu_opts --ntasks=1" -s 3 -z partial
  check_and_do "$gpu_opts --ntasks=1" -s 4 -z 'indep partial' -p pretrain
  check_and_do "$cpu_opts --ntasks=1 --cpus-per-task=8" -s 5 -z 'indep partial' -p pretrain
) & outer_pids[$!]="indep_partial_pretrain"
(
  sleep 1
  check_and_do "$gpu_opts --ntasks=1" -s 4 -z 'indep partial' -p flatstart
  check_and_do "$cpu_opts --ntasks=1 --cpus-per-task=8" -s 5 -z 'indep partial' -p flatstart -c cpu
) & outer_pids[$!]="indep_partial_flatstart"
(
  sleep 1
  check_and_do "$gpu_opts --ntasks=4" -s 3 -z full
  check_and_do "$gpu_opts --ntasks=2" -s 4 -e 'srswor ais-c' -p pretrain
  check_and_do "$gpu_tasks --ntasks=4" -s 4 -e 'sf-biased sf-is direct' -p pretrain
  check_and_do "$cpu_opts --ntasks=1 --cpus-per-task=8" -s 5 -e 'sf-biased sf-is direct' -p pretrain -c cpu
) & outer_pids[$!]="full_pretrain"
(
  sleep 1
  check_and_do "$gpu_opts --ntasks=2" -s 4 -e 'srswor ais-c' -p flatstart
  check_and_do "$gpu_tasks --ntasks=4" -s 4 -e 'sf-biased sf-is direct' -p flatstart
  check_and_do "$cpu_opts --ntasks=1 --cpus-per-task=8" -s 5 -e 'sf-biased sf-is direct' -p flatstart -c cpu
) & outer_pids[$!]="full_flatstart"
while [ "${#outer_pids[@]}" -gt 0 ]; do
  wait -n "${!outer_pids[@]}"
  for outer_pid in "${!outer_pids[@]}"; do
    if ! ps -p $outer_pid > /dev/null; then
      wait -n $outer_pid
      vf=$?
      if [ $vf -eq 0 ]; then
        echo "${outer_pids[outer_pid]} succeeded"
      else
        echo "${outer_pids[outer_pid]} failed with pid $vf"
      fi
      unset 'outer_pids[outer_pid]'
    fi
  done
done

./timit.sh -s 100