#! /bin/bash

BLOB_PATH="/blob"
DATA_PATH="/data"

[ "`whoami`" = "azureuser" ] || {
    echo "Go script must be run as azureuser!"
    exit 1
}

# We expect to run this in the user's home directory
cd "$HOME"

# Go ahead and run the init script as root.
echo "Running VM Initialization ..."
if sudo -H ./visual-autolabel_init.sh
then 
else echo ""
     echo "Failed to run initialization script."
     exit 2
fi

# Figure out what our VM name and parameters are.
source /vmconfig.sh
if ([ -z "$VM_START" ] || [ -z "$VM_STOP" ] || [ -z "$VM_STRIDE" ])
then echo "start, stop, and stride tags not found!"
     exit 3
fi

# Load and run the docker.
echo "Loading Docker Image ..."
docker load -i /data/docker-images/gridsearch_0.1.tar
echo "Executing Grid Search using start=${VM_START}, stride=${VM_STRIDE}, stop=${VM_STOP}"
docker run --rm -it \
       -v /blob:/data:rw \
       nben/visual-autolabel_gridsearch:0.1 \
       "$VM_START" "$VM_STRIDE" "$VM_STOP"
echo ""
echo "Search complete!"
