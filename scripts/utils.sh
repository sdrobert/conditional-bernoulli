#! /usr/bin/env bash

# Utility functions
# 
# This script should be sourced in whatever script wants to use these functions

# XXX(sdrobert): we design functions to be composed on the left and arguments
# to be composed on the right. This can get counterintuitive, e.g. in
# "is_a_match", but allows us to do things like
# 
#   seq 1 20 | filter is_not is_a_match 1
# 
# which prints all the numbers between 1 and 20 which don't have a 1 in them

# is the argument a natural number?
is_nat() [[ "$1" =~ ^[0-9]*[1-9][0-9]*$ ]]

# is the argument a directory?
is_dir() [[ -d "$1" ]]

# is the argument writable?
is_writable() [[ -w "$1" ]]

# is the argument readable?
is_readable() [[ -r "$1" ]]

# is the argument readable and writable?
is_rw() [[ -r "$1" && -w "$1" ]]

# is the argument a file?
is_file() [[ -f "$1" ]]

# is the second argument equal to the first?
is_equal() [[ "$1" = "$2" ]]

# is the resolution of the following command not true?
is_not() { "$@" && return 1 || return 0; }

# do all incoming lines satisfy some 'is_' function with the line past as the
# last argument?
all() {
  local x
  while read x; do
    "$@" "$x" || return 1
  done
  return 0
}

# do any of the incoming lines satisfy some 'is_' function with the line past
# as the last argument?
any() {
  local x
  while read x; do
    "$@" "$x" && return 0
  done
  return 1
}

# is the final argument equal to one of the preceding arguments?
is_a_choice() {
  local i
  for (( i = 0; i < $#; i++ )); do
    [ "${!i}" = "${!#}" ] && return 0
  done
  return 1
}

# does the final argument match one of the preceding arguments, which are
# regular expressions?
is_a_match() {
  local i
  for (( i = 0; i < $#; i++ )); do
    [[ "${!#}" =~ ${!i} ]] && return 0
  done
  return 1
}

# A cartesian product of incoming lines with arguments. For each incoming line,
# takes the first argument to this function as a delimiter, then prints one
# line for every remaining argument of the form
# 
#   <line><delimiter><arg>
#
# Derives from
# https://stackoverflow.com/questions/23363003/how-to-produce-cartesian-product-in-bash
prod() {
  local delim="$1"
  shift
  while read x; do
    printf "$x$delim%s\n" "$@"
  done
}

# prints the lines which satisfy some 'is_' function with the line 
filter() {
  local x
  while read x; do
    "$@" "$x" && echo "$x" || true
  done
}


# argcheck functions

_argcheck_is() {
  local cmd="${!#}"
  if ! "$cmd" "${@:3:$#-3}"; then
    echo "$0: argparse -$2 value '${@:$#-1:1}' $1" 1>&2
    declare -F "usage" > /dev/null && usage
    exit 1
  fi
  return 0
}

argcheck_is_nat() { _argcheck_is "is not natural" "$@" is_nat; }
argcheck_is_dir() { _argcheck_is "is not a directory" "$@" is_dir; }
argcheck_is_writable() { _argcheck_is "is not writable" "$@" is_writable; }
argcheck_is_readable() { _argcheck_is "is not readable" "$@" is_readable; }
argcheck_is_rw() { _argcheck_is "is either not readable or not writable" "$@" is_rw; }
argcheck_is_file() { _argcheck_is "is not a file" "$@" is_file; }
argcheck_is_a_choice() { _argcheck_is "is not in choices '${*:2:$#-2}'" "$@" is_a_choice; }
argcheck_is_a_match() { _argcheck_is "does not match any of '${*:2:$#-2}'" "$@" is_a_match; }

_argcheck_all() {
  local cmd="${!#}"
  local a i
  read -ra a <<<"${@:$#-1:1}"
  for (( i = 0; i < "${#a[@]}"; i++ )); do
    if ! "$cmd" "${@:3:$#-4}" "${a[$i]}"; then
      echo "$0: argparse -$2 value $(($i+1)), '${a[$i]}', $1" 1>&2
      declare -F "usage" > /dev/null && usage
      exit 1
    fi
  done
  return 0
}

argcheck_all_nat() { _argcheck_all "is not natural" "$@" is_nat; }
argcheck_all_dir() { _argcheck_all "is not a directory" "$@" is_dir; }
argcheck_all_writable() { _argcheck_all "is not writable" "$@" is_writable; }
argcheck_all_readable() { _argcheck_all "is not readable" "$@" is_readable; }
argcheck_all_file() { _argcheck_all "is not a file" "$@" is_file; }
argcheck_all_a_choice() { _argcheck_all "is not in choices '${*:2:$#-2}'" "$@" is_a_choice; }
argcheck_all_a_match() { _argcheck_all "does not match any of '${*:2:$#-2}'" "$@" is_a_match; }

