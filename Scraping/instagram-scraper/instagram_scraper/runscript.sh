#!/bin/bash

for TAG in  "$@"
do
	echo "start $TAG"
	python3 app.py --tag $TAG --media_type image --destination "instagram/${TAG}"
	echo "done $TAG"
done
