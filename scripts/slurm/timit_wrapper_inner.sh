#!/usr/bin/env bash

global_args=( )

if [ -d "/checkpoint/${USER}/${SLURM_JOB_ID}" ]; then
  export TIMIT_CKPT_DIR="/checkpoint/${USER}/${SLURM_JOB_ID}/ckpt"
  mkdir -p "$TIMIT_CKPT_DIR"
  global_args+=( -O "$TIMIT_CKPT_DIR" -w )
fi

./timit.sh "${global_args[@]}" "$@"
