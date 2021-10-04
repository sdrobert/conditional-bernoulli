#! /usr/bin/env bash

conf=conf
res=res
log=log

mcs=( 1 256 65536 )
gammas=( 0 0.25 0.75 )
probs=( 0.01 0.25 0.75 )
estimators=( rej sswor cb ais-cb-lp ais-cb-gibbs )
num_seeds=10

template='[DreznerFarnumBernoulliExperimentParameters]
batch_size = 1
estimator = ESTIMATOR
fmax = 16
gamma = GAMMA
learning_rate = 0.001
num_mc_samples = MC
num_trials = 4096
optimizer = adam
p = PROB
tmax = 128
vmax = 16
x_std = 1.0
'

mkdir -p "$conf" "$res" "$log"

num_jobs=0
for mc in "${mcs[@]}"; do
  for gamma in "${gammas[@]}"; do
    for prob in "${probs[@]}"; do
      for estimator in "${estimators[@]}"; do
        [ "$estimator" = "cb" ] && [ "$mc" != "1" ] && continue
        num_jobs=$(( $num_jobs + 1 ))
        echo "$template" | \
          sed "s/MC/${mc}/;s/GAMMA/${gamma}/;s/PROB/${prob}/;s/ESTIMATOR/${estimator}/" \
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
#SBATCH -c 32
#SBATCH --export=ALL
#SBATCH --output=LOG/df_%A_%a.out

pwd; hostname; date

set -e

conf="CONF/df.${SLURM_ARRAY_TASK_ID}.ini"
res="RES/df.${SLURM_ARRAY_TASK_ID}."
num_seeds=NS

echo "Handling config $conf for $num_seeds seeds"
echo "---------------------------------------"
cat "$conf"
echo "---------------------------------------"

for x in $(seq 1 $num_seeds ); do
  echo "Beginning seed $x:"
  echo "----------------------------------------"
  python df_bernoulli.py --read-ini "$conf" --seed $x "$res$x.csv"
  echo "----------------------------------------"
  echo "Done"
  echo "----------------------------------------"
done
' | sed "s/NJ/${num_jobs}/;s/LOG/${log}/;s/CONF/${conf}/;s/RES/${res}/;s/NS/${num_seeds}/" > df.sh