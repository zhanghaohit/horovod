#! /usr/bin/env bash

bin=`dirname $0`
bin=`cd $bin; pwd`

mode=mpi
if [[ $# -ge 1 ]]; then
  mode=$1
fi

log_level=info
port=12345
exe="python3 examples/tensorflow_mnist.py"
pythonpath=$bin/build/lib.linux-x86_64-3.5
# exe="python3 examples/tensorflow_synthetic_benchmark.py"

old_IFS=$IFS
IFS=$'\n'
hosts=()
slots=()
num_hosts=0
for line in `cat hosts`
do
  host=`echo $line | cut -d ' ' -f1`
  slot=`echo $line | cut -d ' ' -f2 | cut -d '=' -f2`
  hosts[$num_hosts]=$host
  slots[$num_hosts]=$slot
  num_hosts=$((num_hosts+1))

  if [[ $host != `hostname` ]]; then
    echo "rsync to $host"
    rsync -r --links $bin/* $host:$bin
  fi
done
IFS=$old_IFS

for number in 4
do
  rm -rf checkpoints
  start_time=$(date +%s)
  if [[ $mode = mpi ]]; then
    echo "pythonpath = " $pythonpath
    echo "Using MPI"
    mpirun -np $number -hostfile hosts --mca btl_tcp_if_include bond0 --mca oob_tcp_if_include bond0 -x PYTHONPATH=$pythonpath -x NCCL_SOCKET_IFNAME=bond0 -x HOROVOD_LOG_LEVEL=$log_level $exe
  else
    echo "Using socket"
    rank=0
    for h in $(seq 0 $((num_hosts-1)))
    do
      host=${hosts[$h]}
      slot=${slots[$h]}
      if [[ -z $master ]]; then
        master=$host:$port
        echo "Master is $master"
      fi

      for i in $(seq 0 $((slot-1)))
      do
        echo "Running rank-$rank on $host:$slot-$i"
        ssh $host "cd $bin; PYTHONPATH=$pythonpath NCCL_SOCKET_IFNAME=bond0 HOROVOD_LOG_LEVEL=$log_level HOROVOD_MASTER=$master HOROVOD_NUM_RANKS=$number HOROVOD_RANK=$rank $exe" &
        rank=$((rank+1))
        if [[ $rank = $number ]]; then
          break
        fi
      done

      if [[ $rank = $number ]]; then
        break
      fi
    done
    wait
  fi
  end_time=$(date +%s)
  echo elapsed: $((end_time-start_time))s
done
