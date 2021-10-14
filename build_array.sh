#! /usr/bin/env bash

conf=conf
res=res
tmpdir=/checkpoint

mcs=( 1 128 256 512 1024 )
p_1s=( 0.25 0.75 )
p_2s=( 0 0.25 0.75 )
estimators=( rej fs srswor ecb ais-cb-count ais-cb-gibbs )
num_seeds=20

template='[DreznerFarnumBernoulliExperimentParameters]
ais_burn_in = 32
estimator = ESTIMATOR
fmax = 16
kl_batch_size = 256
learning_rate = 0.1
num_mc_samples = MC
num_trials = 1024
optimizer = adam
p_1 = P1
p_2 = P2
reduce_lr_factor = 0.01
reduce_lr_patience = 16
reduce_lr_threshold = 1
reduce_lr_min = 0.01
theta_3_std = 1.0
tmax = 32
train_batch_size = 64
vmax = 16
x_std = 1.0
'

mkdir -p "$conf" "$res" "$tmpdir"

num_jobs=0
for mc in "${mcs[@]}"; do
  for p_1 in "${p_1s[@]}"; do
    for p_2 in "${p_2s[@]}"; do
      for estimator in "${estimators[@]}"; do
        # no point in more than one mc sample (?) for ecb and fewer than 1
        # for ais algorithms
        [ "$estimator" = "ecb" ] && [ "$mc" != "1" ] && continue
        [ "$mc" = 1 ] && [[ "$estimator" =~ ^ais-* ]] && continue
        num_jobs=$(( $num_jobs + 1 ))
        echo "$template" | \
          sed "s/MC/${mc}/;s/P1/${p_1}/;s/P2/${p_2}/;s/ESTIMATOR/${estimator}/" \
          > "$conf/df.${num_jobs}.ini"
      done
    done
  done
done

if [ "$num_jobs" = "0" ]; then
  echo "No configs"
  exit 1
fi

echo '#! /usr/bin/env bash
#SBATCH --qos=nopreemption
#SBATCH -p cpu
#SBATCH --array 1-NJ
#SBATCH --mem=16G
#SBATCH --job-name=df
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --export=ALL
#SBATCH --output=logs/%A_%a.out

pwd; hostname; date

set -e

tmpdir="TMP/${USER}/${SLURM_JOB_ID}/csv"
conf="CONF/df.${SLURM_ARRAY_TASK_ID}.ini"
res="RES/df.${SLURM_ARRAY_TASK_ID}.csv.gz"
num_seeds=NS

mkdir -p "$tmpdir"
find "$tmpdir" -name "*.csv" -delete

echo "Handling config $conf for $num_seeds seeds"
echo "---------------------------------------"
cat "$conf"
echo "---------------------------------------"

if [ "$num_seeds" = "0" ]; then
  echo "No seeds"
  echo -n "" | gzip > "$res"
fi

for x in $(seq 1 $num_seeds ); do
  echo "Beginning seed $x:"
  echo "----------------------------------------"
  python df_bernoulli.py --read-ini "$conf" --seed $x "$tmpdir/$x.csv"
  echo "----------------------------------------"
  echo "Done"
  echo "----------------------------------------"
done

if [ "$(find "$tmpdir" -name "*.csv" | wc -l)" != "${num_seeds}" ]; then
  echo "Something mucked up! We do not have enough files"
  exit 1
fi

awk "FNR==1 && NR!=1{next;}{print}" "$tmpdir/"*.csv | gzip > "$res"
' | sed "s/NJ/${num_jobs}/;s:TMP:${tmpdir}:;s:CONF:${conf}:;s:RES:${res}:;s/NS/${num_seeds}/" > df.sh