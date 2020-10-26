#! /usr/bin/env bash

seed=0
repeat=100
trials_low=
trials_high=6
highs_low=
highs_high=4
batch_low=
batch_high=0
device=cpu
method=R
burnin=100
hist=false
reverse=false

help_message="
Usage:
$0
   -s <seed>        Seed (default: $seed)
   -r <repeat>      Number of repetitions (default: $repeat)
   -T <trials>      Log-number of weights (default: $trials_high)
   -t <min-trials>  Minimum log-number of weights. If set, will repeat the
                    experiment using every integer inclusive between this and
                    the -T flag (inclusive)
   -L <highs>       Log-number of weights in the count (default: $highs_high)
   -l <min-highs>   Miminum log-number of weights in the count. If set, will
                    repeat the experiment using every integer inclusive between
                    this and the -L flag (inclusive)
   -N <batch>       Log-number of elements in the batch (default: $batch_high)
   -n <min-batch>   Minimum log-number of elements in the batch. If set, will
                    repeat the experiment using every integer inclusive between
                    this and the -N flag (inclusive)
   -d <device>      Device to put tensors on (default: $device)
   -m <method>      One of 'R', 'lR', 'R2', or 'R3' (default: $method)
   -b <burn-in>     Number of times to perform the op prior to the experiment
                    (default $burnin)
   -H               If set, ops will return history
   -R               If set, ops will reverse weights
   -h               Print help message to stderr and quit
"

check_bounded_int() {
  name="$1"
  val="$2"
  low="$3"
  high="$4"
  [ "$val" -eq 0 ] 2> /dev/null || [ "$val" -ne  0 ] 2> /dev/null
  if [ $? -ne 0 ]; then
    echo -e "Expected $name to be an integer, got $val"
    exit 1
  fi
  [ -z "$low" ] || [ "$val" -ge "$low" ] 2> /dev/null
  if [ $? -ne 0 ]; then
    echo -e "Expected $name to be less than or equal to $low, got $val"
    exit 1
  fi
  [ -z "$high" ] || [ "$val" -le "$high" ] 2> /dev/null
  if [ $? -ne 0 ]; then
    echo -e "Expected $name to be greater than or equal to $low, got $val"
    exit 1
  fi
}

while getopts ":s:r:T:t:L:l:N:n:d:m:b:HRh" opt ; do
  case $opt in
    s)
      check_bounded_int "s" "$OPTARG" 0
      if [ $? -eq 0 ]; then
        seed="$OPTARG"
      else
        echo -e "$help_message"
        exit 1
      fi
      ;;
    r)
      check_bounded_int "r" "$OPTARG" 1
      if [ $? -eq 0 ]; then
        repeat="$OPTARG"
      else
        echo -e "$help_message"
        exit 1
      fi
      ;;
    T)
      check_bounded_int "T" "$OPTARG" 0
      if [ $? -eq 0 ]; then
        trials_high="$OPTARG"
      else
        echo -e "$help_message"
        exit 1
      fi
      ;;
    t)
      check_bounded_int "t" "$OPTARG" 0
      if [ $? -eq 0 ]; then
        trials_low="$OPTARG"
      else
        echo -e "$help_message"
        exit 1
      fi
      ;;
    L)
      check_bounded_int "L" "$OPTARG" 0
      if [ $? -eq 0 ]; then
        highs_high="$OPTARG"
      else
        echo -e "$help_message"
        exit 1
      fi
      ;;
    l)
      check_bounded_int "l" "$OPTARG" 0
      if [ $? -eq 0 ]; then
        trials_low="$OPTARG"
      else
        echo -e "$help_message"
        exit 1
      fi
      ;;
    N)
      check_bounded_int "N" "$OPTARG" 0
      if [ $? -eq 0 ]; then
        batch_high="$OPTARG"
      else
        echo -e "$help_message"
        exit 1
      fi
      ;;
    n)
      check_bounded_int "n" "$OPTARG" 0
      if [ $? -eq 0 ]; then
        batch_low="$OPTARG"
      else
        echo -e "$help_message"
        exit 1
      fi
      ;;
    d)
      device="$OPTARG"
      ;;
    m)
      method="$OPTARG"  # fail happens in python
      ;;
    b)
      check_bounded_int "b" "$OPTARG" 1
      if [ $? -eq 0 ]; then
        burnin="$OPTARG"
      else
        echo -e "$help_message"
        exit 1
      fi
      ;;
    H)
      hist=true
      ;;
    R)
      reverse=true
      ;;
    h)
      echo -e "$help_message"
      exit 0
      ;;
    \?)
      echo -e "Invalid option: -$OPTARG"
      echo -e "$help_message"
      exit 1
      ;;
    :)
      echo -e "Option -$OPTARG requires an argument"
      echo -e "$help_message"
      exit 1
      ;;
  esac
done

if [ -z "$trials_low" ]; then
  trials_low=$trials_high
elif [ "$trials_low" -gt "$trials_high" ]; then
  echo -e "-t is $trials_low but -T is $trials_high"
  echo -e "$help_message"
  exit 1
fi
if [ -z "$highs_low" ]; then
  highs_low=$highs_high
elif [ "$highs_low" -gt "$highs_high" ]; then
  echo -e "-l is $highs_low but -L is $highs_high"
  echo -e "$help_message"
  exit 1
fi
if [ -z "$batch_low" ]; then
  batch_low="$batch_high"
elif [ "$batch_low" -gt "$batch_high" ]; then
  echo -e "-n is $batch_low but -N is $batch_high"
  echo -e "$help_message"
  exit 1
fi

fixed_args="--seed=$seed --repeat=$repeat --device=$device --method=$method speed --burn-in=$burnin"
if $hist ; then
  fixed_args="$fixed_args --hist"
fi
if $reverse ; then
  fixed_args="$fixed_args --reverse"
fi

n_iter=$(echo "($trials_high - $trials_low + 1) * ($highs_high - $highs_low + 1) * ($batch_high - $batch_low + 1)" | bc)

seq 1 $n_iter | xargs -I {} bash -c '
  it=$(($1 - 1))
  shift
  trial_steps=$(($2 - $1 + 1))
  trial_step=$((it % $trial_steps))
  trial=$(( 2 ** ($trial_step + $1)))
  it=$((it / trial_steps))
  shift 2
  high_steps=$(($2 - $1 + 1))
  high_step=$((it % $high_steps))
  highs=$(( 2 ** ($high_step + $1)))
  it=$((it / high_steps))
  shift 2
  batch_step=$it
  batch=$(( 2 ** ($batch_step + $1)))
  shift 2
  python poisson_binomial.py --batch=$batch --trial=$trial --highs=$highs $*
' -- {} "$trials_low" "$trials_high" "$highs_low" "$highs_high" "$batch_low" "$batch_high" $fixed_args