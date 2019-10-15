#! /usr/bin/env bash

CC=g++-4.9 LIBRARY_PATH=$HOME/workspace/nccl-src/build/lib:$LIBRARY_PATH LD_LIBRARY_PATH=$LIBRARY_PATH DYNAMIC_SCHEDULE=1 GRPC_LIB_HOME=$HOME/lib ./setup.py build
