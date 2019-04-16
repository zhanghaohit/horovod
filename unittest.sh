#! /usr/bin/env bash

if [[ $# -lt 1 ]]; then
  echo "Usage: ./unittest.sh [command]"
  echo "Command: build/run"
fi

command=$1

SRC="horovod/common/net.cc horovod/common/logging.cc horovod/common/controller_client.cc horovod/common/grpcservice*.cc"
LIBS="-lpthread -lgtest -lgtest_main -lprotobuf -lgrpc++ -lstdc++"

if [[ $command == build ]]; then
  echo "compile unittest"
  g++ -std=c++14 -O2 -g -o unittest horovod/test/*.cc $SRC $LIBS
elif [[ $command == run ]]; then
  echo "run unittest"
fi
