#!/bin/bash

for TAG in  "$@"
do
        echo "start $TAG"
        QUERY=${TAG//+/ }	
	echo $QUERY > queries.txt
	python miner.py -n 10000000 flickr queries.txt "./images_2/$TAG"
        echo "done $TAG"
done
