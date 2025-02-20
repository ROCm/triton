#!/bin/bash

# Check which containers are running.
# docker ps

# Spin up container for the first time.
# cd docker_triton/
# docker-compose up -d

# Start container, e.g. after reboot
# docker start dtanner_triton_dev_1

# Enter running container.
#docker exec -it dtanner-triton_dev-1 /bin/bash

# run with extra volume mounted
docker exec -it dtanner-triton_dev-1 /bin/bash
#docker exec --user root -it dtanner-triton_dev-1 /bin/bash

# Enter running container as root.
#docker exec --user root -it dtanner-triton_dev-1 /bin/bash
