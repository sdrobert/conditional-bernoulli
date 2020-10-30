#! /usr/bin/env bash

# Copyright 2020 Sean Robertson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

repeat=10
trials=( 10 )
highs=( 4 )
device=( cpu )
float=( true )
burnin=10
batch=( 0 )
hist=( false )
ratio=( 0 )
exp=( 0 )

avail_methods=( $(python count_function.py -h | \
  grep "chen94" | head -n 1 | \
  tr -d ' ' | tr -d '{' | tr -d '}' | tr ',' ' ' ) )

help_message="
Usage:
$0 [<opts>] {speed|acc} <method> [<method> [...]]

<method> is one of:
  ${avail_methods[*]}

Common <opts>:
   -h               Print help message to stderr and quit
   -r <int>         Log-base-2-number of repetitions (default: $repeat)
   -t <list>        Log-base-2-number of trials/weights (default: ${trials[0]})
   -l <list>        Log-base-2-number of highs (default: ${highs[0]})
   -d <list>        Device to compute on (default: ${device[0]})
   -f <list>        Single-precision floats (true) or double (false)
                    (default: ${float[0]})

Speed benchmarking <opts>:
   -b <int>         Log-base-2 of times to compute the op before starting to
                    record times (default: $burnin)
   -n <list>        Log-base-2 batch size per rep (default: ${batch[0]})
   -a <list>        If true, compute entire history (default: ${hist[0]})

Accuracy benchmarking <opts>:
   -s <list>        Seed for randomization (default ${seed[0]})
   -x <list>        Log-base-2 ratio of weight values (default ${ratio[0]})
   -e <list>        Log-base-2 expectation of count value (default ${exp[0]})

<list>={<value>|<value>-<value>|<value>,<list>|<value>-<value>,<list>}

The cartesian product of all settings will be run in series. E.g.

  $0 -t 2,5-6 -l 0-1 speed cumsum chen94

Is the same as running (without the order guarantee)

  $0 -t 2 -l 0 speed cumsum
  $0 -t 2 -l 0 speed chen94
  $0 -t 2 -l 1 speed cumsum
  $0 -t 2 -l 1 speed chen94
  $0 -t 5 -l 0 speed cumsum
  ...
"

check_bounded_int() {
  name="$1"
  low="$2"
  high="$3"
  while read val ; do
    [ "$val" -eq 0 ] 2> /dev/null || [ "$val" -ne  0 ] 2> /dev/null
    if [ $? -ne 0 ]; then
      >&2 echo "Expected $name to be an integer, got $val"
      exit 1
    fi
    [ -z "$low" ] || [ "$val" -ge "$low" ] 2> /dev/null
    if [ $? -ne 0 ]; then
      >&2 echo "Expected $name to be less than or equal to $low, got $val"
      exit 1
    fi
    [ -z "$high" ] || [ "$val" -le "$high" ] 2> /dev/null
    if [ $? -ne 0 ]; then
      >&2 echo "Expected $name to be greater than or equal to $low, got $val"
      exit 1
    fi
    echo "$val"
  done
}

check_boolean() {
  name="$1"
  while read val ; do
    if [[ "$val" =~ ^(([Yy](es)?)|([Tt](ue)?))$ ]]; then
      echo "true"
    elif [[ "$val" =~ ^(([Nn]o?)|([Ff](alse)?))$ ]]; then
      echo "false"
    else
      >&2 echo "Expected true/false value for $name, got $val"
      exit 1
    fi
  done
}

parse_list() {
  IFS=, read -ra as <<< "$1"
  for a in "${as[@]}"; do
    if [[ "$a" =~ ^[0-9]+-[0-9]+$ ]]; then
      low="${a%-*}"
      high="${a#*-}"
      seq "$low" "$high"
    else
      echo "$a"
    fi
  done
}

trap ">&2 echo '$help_message'" ERR
set -e
while getopts "hr:t:l:d:f:b:n:a:s:x:e:" opt ; do
  case $opt in
    h) echo "$help_message" && exit 0;;
    r) repeat="$(echo "$OPTARG" | check_bounded_int "r" 0)";;
    t) trials=( $(parse_list "$OPTARG" | check_bounded_int "t" 0) );;
    l) highs=( $(parse_list "$OPTARG" | check_bounded_int "l" 0) );;
    d) device=( $(parse_list "$OPTARG") );;
    f) float=( $(parse_list "$OPTARG" | check_boolean "f" ) );;
    b) burnin="$(echo "$OPTARG" | check_bounded_int "b" 0)";;
    n) batch=( $(parse_list "$OPTARG" | check_bounded_int "n" 0) );;
    a) hist=( $(parse_list "$OPTARG" | check_boolean "a" ) );;
    s) seed=( $(parse_list "$OPTARG" | check_bounded_int "s" 0) );;
    x) ratio=( $(parse_list "$OPTARG" | check_bounded_int "x" 0) );;
    e) exp=( $(parse_list "$OPTARG" | check_bounded_int "e" 0) );;
  esac
done

shift $((OPTIND - 1))

if [ $# -lt 2 ]; then
  >&2 echo "Expected at least two positional arguments" && false
fi

com="$1"
shift
if [ "$com" != "speed" ] && [ "$com" != "acc" ]; then
  >&2 echo "First argument must be 'speed' or 'acc', got $com" && false
fi

declare -a methods
while [ $# -ne 0 ]; do
  method="$1"
  shift
  if ! printf '%s\n' "${avail_methods[@]}" | grep -qxF "$method" ; then
    >&2 echo "$method is invalid. Must be one of ${avail_methods[*]}" && false
  fi
  methods+=( "$method" )
done

set +e

cartp() {
  var_name="$1";
  shift
  while read hist; do
    for val in "$@"; do
      echo "$hist; $var_name='$val'"
    done
  done
}

echo "com='$com'; repeat='$repeat'; burnin='$burnin'" | \
  cartp "method" "${methods[@]}" | \
  cartp "trials" "${trials[@]}" | \
  cartp "highs" "${highs[@]}" | \
  cartp "device" "${device[@]}" | \
  cartp "float" "${float[@]}" | \
  cartp "batch" "${batch[@]}" | \
  cartp "hist" "${hist[@]}" | \
  cartp "ratio" "${ratio[@]}" | \
  cartp "exp" "${exp[@]}" | \
  xargs -I{} bash -c '
eval "$1";
[ $highs -gt $trials ] && exit 0
a="--log2-repeat $repeat --log2-trials $trials --log2-highs $highs"
a="$a --device $device"
[ "$float" = "false" ] && a="$a --double"
a="$a $method $com"
if [ "$com" = "speed" ]; then
  a="$a --log2-burn-in $burnin --log2-batch $batch"
  [ "$hist" = "true" ] && a="$a --hist"
else
  a="$a --seed $seed --log2-ratio-odds $ratio --log2-expectation $exp"
fi
python count_function.py $a
' -- "{}"