#! /usr/bin/env bash
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --export=ALL
#SBATCH --output=logs/timit/slurm-%j-%a.log
#SBATCH --nodes=1
#SBATCH --wait
#SBATCH --wait-all-nodes=1

if [ ! -z "${SLURM_ARRAY_TASK_ID}" ]; then
  export TIMIT_OFFSET="$(( ${SLURM_ARRAY_TASK_ID} - 1 ))"
  export TIMIT_STRIDE="${SLURM_ARRAY_TASK_COUNT}"
fi

export MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')"
if [ "${SLURM_NTASKS}" != "1" ]; then
  export MASTER_ADDR"=$(hostname --fqdn)"
  # export MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')"
  export NCCL_IB_DISABLE=1
  export WORLD_SIZE="${SLURM_NTASKS}"
  for ((i=0; i < SLURM_NTASKS; i++)); do
    /opt/slurm/bin/srun \
      --export=ALL,RANK=$i \
      --mem-per-cpu=$SLURM_MEM_PER_CPU \
      --cpus-per-task=$SLURM_CPUS_PER_TASK \
      --ntasks=1 \
      --gpus-per-task=${SLURM_GPUS_PER_TASK:-0} \
      ./scripts/slurm/timit_wrapper_inner.sh "$@" &
  done
  wait
else
  ./scripts/slurm/timit_wrapper_inner.sh "$@" -f "${SLURM_GPUS_PER_TASK:-1}"
fi

exit $!
