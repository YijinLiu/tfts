# Find more versions here: https://hub.docker.com/r/tensorflow/tensorflow/tags

WORK_DIR:=$(shell readlink -f $(dir $(lastword $(MAKEFILE_LIST))))

default:
	cd $(WORK_DIR)
	cd pkgs ; python3 -m http.server & echo "$$!" > ./HTTP_SERVER_PID && \
	    sleep 2 && kill -0 $$(cat ./HTTP_SERVER_PID)
	nvidia-docker build --network=host -t ${USER}/tfts:2.14.0-gpu .
	kill $$(cat pkgs/HTTP_SERVER_PID) && rm pkgs/HTTP_SERVER_PID
