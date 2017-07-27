#!/bin/bash
PID=26834
ps
while kill -0 "$PID" >/dev/null 2>&1; do
 sleep 20 
 echo "waiting"
done
echo "restarting"
python "train_models.py"
exit 0
