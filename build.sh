#!/bin/bash
# setting variables
NAME="covid19_model_on_pytorch"
CNAME="container_${NAME}"
TAG="1.0.0"

if [ "$1" == "start" ]
then
    # if arg is "start"
    ## build docker image
    sudo docker rm -f $(docker ps -aq) || echo "Everything is clean." &&
    sudo docker build -t $NAME:$TAG . &&
    #sudo docker run --name $CNAME -d -v $(pwd)/scr:/home:rw $NAME:$TAG /bin/sh -c "while true; do echo container is working; sleep 1; done"  &&
    sudo docker run -p 8888:8888 --name $CNAME -v $(pwd)/scr:/home:rw $NAME:$TAG &&
    ## remove unused image and volume
    sudo docker image prune -a -f &&
    sudo docker volume prune -f
else
    # if arg is "stop" or anything else
    sudo docker rm -f $(docker ps -aq)
fi

# unsetting variables
unset NAME
unset CNAME
unset TAG

# show result
sudo docker image ls &&
sudo docker ps