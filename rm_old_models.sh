#!/usr/bin/env bash

# Usage: rm_old_models.sh folder n
# Removes n oldest files in folder
# folders default value is the models folder
# n defaults such that there are only 2 files left

rm_oldest_file() {
  # $1 is folder
  last_file=$(ls -t $1 | tail -1)
  echo "Deleting $1/$last_file"
  rm "$1/$last_file"
}

folder=${1:-"models"}
n_files=$(ls -1 $folder | wc -l)

min_files=2
((max_delete = $n_files - $min_files))
n=${2:-$max_delete}

if [ $n -gt 0 ]; then
  for i in $(seq 1 $n); do
    rm_oldest_file $folder
  done
fi
