#!/bin/bash
for i in {1..10}
do
  echo "Starting Iteration $i..."
  ./ralph-once.sh
  # Optional: break if agent signals "COMPLETE"
done
