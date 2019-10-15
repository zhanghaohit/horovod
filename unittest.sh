#! /usr/bin/env bash

if [[ $# -lt 1 ]]; then
  echo "Usage: ./unittest.sh [command]"
  echo "Command: build/run"
fi

command=$1
shift

avail_gpus=0,2,9
SRC="horovod/common/net.cc horovod/common/logging.cc horovod/common/controller_client.cc horovod/common/grpcservice*.cc"
LIBS="-lpthread -lgtest -lgtest_main -lprotobuf -lgrpc++ -lstdc++"

if [[ $command == build ]]; then
  echo "compile unittest"
  g++ -std=c++14 -O2 -g -o unittest horovod/test/*.cc $SRC $LIBS
elif [[ $command == run ]]; then
  echo "run unittest"
  ./unittest --gtest_filter=-ControllerClientTest.*
  cd horovod
  for num_ranks in 1 2 3
  do
    for i in $(seq 0 $((num_ranks-1)))
    do
      CUDA_VISIBLE_DEVICES=$avail_gpus NCCL_DEBUG=WARN HOROVOD_LOG_LEVEL=warn AUTOBOT_NUM_RANKS=$num_ranks AUTOBOT_RANK=$i AUTOBOT_MASTER_URI=localhost:12345 pytest -s $@ &
    done
    wait
    cp res.csv res.csv-num_ranks$num_ranks > /dev/null 2>&1
  done
fi
