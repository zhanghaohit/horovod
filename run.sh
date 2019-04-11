#! /usr/bin/env bash

mode=mpi
if [[ $# -ge 1 ]]; then
  mode=$1
fi

log_level=debug
master=sgrg0003:12345
exe="python3 examples/tensorflow_mnist.py"

for number in 2
do
  rm -rf checkpoints
  start_time=$(date +%s)
  if [[ $mode = mpi ]]; then
    echo "Using MPI"
    HOROVOD_LOG_LEVEL=$log_level python3 ./bin/horovodrun -np $number -H localhost:$number $exe
  else
    echo "Using socket"
    for i in $(seq 0 $((number-1)))
    do
      HOROVOD_LOG_LEVEL=$log_level HOROVOD_MASTER=$master HOROVOD_NUM_RANKS=$number HOROVOD_RANK=$i $exe &
    done
    wait
  fi
  end_time=$(date +%s)
  echo elapsed: $((end_time-start_time))s
done
