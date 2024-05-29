#!/bin/bash

PID=$1
CMD=$2

while kill -0 $PID 2> /dev/null; do
  sleep 1
done

eval $CMD
