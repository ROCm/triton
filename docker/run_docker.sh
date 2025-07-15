#!/bin/bash

#CONTAINER_NAME=dtanner-triton_dev-1
CONTAINER_NAME=dtanner-triton_dev_6.4b-1 

# Check which containers are running.
# docker ps

# Spin up container for the first time.
# cd docker_triton/
# docker-compose up -d

# Start container, e.g. after reboot
# docker start ${CONTAINER_NAME}

# Enter running container.
docker exec -it ${CONTAINER_NAME} /bin/bash

# Enter running container as root.
#docker exec --user root -it ${CONTAINER_NAME} /bin/bash
