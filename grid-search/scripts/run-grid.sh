#! /bin/bash

# This script is to be run by the Azure nodes that are performing the
# grid-search 

#===============================================================================
# Configuration

# How many cells are there in the grid search?
CELLCOUNT=4200
# How many nodes we are using.
NODECOUNT=50

# The path of the data directory.
DATA="/data/visual-autolabel"
# The path of of the data inputs.
DATAINPUTS="${DATA}/data"
# The path of the models cache.
DATAMODELS="${DATA}/models"
# The path of the docker-image files (if any).
DATADOCKER="${DATA}/docker-images"
# The run directories.
DATARUN="${DATA}/run"

# Go ahead and source the config.env file.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "$SCRIPT_DIR"/../../config.env

# The name of the docker image we are using.
DOCKERTAG="$GRIDSEARCH_DOCKERTAG"
# The filename of the docker image with the above tag if one exists. If this is
# the empty string, then the above tag is instead pulled.
DOCKERIMG="${DATADOCKER}/${GRIDSEARCH_FILENAME}"
[ -r "$DOCKERIMG" ] || DOCKERIMG=""

# How we handle errors.
function error {
    echo "$@"
    exit 1
}


#===============================================================================
# Docker Code

# docker_init (no parameters)
function docker_init {
    if [ 2 -eq `docker image ls "$DOCKERTAG" | wc -l` ]
    then echo "Docker image ${DOCKERTAG} ready."
    elif [ -n "$DOCKERIMG" ]
    then docker load -q -i "$DOCKERIMG" \
            || error "Could not load docker image: $DOCKERIMG"
    else docker pull -q "$DOCKERTAG" \
            || error "Could not pull docker image: $DOCKERTAG"
    fi
}
# docker_run <node_id>
function docker_run {
    NODEPATH="${DATARUN}/nodes/`printf '%03d' $1`"
    mkdir -p "$NODEPATH"
    LOGFL="$NODEPATH"/node.log
    echo -n "NODE $1 BEGIN AT " >> "$LOGFL"
    date >> "$LOGFL"
    echo -n "########################################" >> "$LOGFL"
    echo    "########################################" >> "$LOGFL"
    echo "" >> "$LOGFL"
    docker run --rm -it \
               -v "${DATAINPUTS}:/data/inputs:ro" \
               -v "${DATAMODELS}:/data/models:rw" \
               -v "${DATARUN}:/data/run:rw" \
               "$DOCKERTAG" \
               "$1" "$NODECOUNT" "$CELLCOUNT" \
        &> "$LOGFL"
    echo "" >> "$LOGFL"
    echo -n "########################################" >> "$LOGFL"
    echo    "########################################" >> "$LOGFL"
    echo -n "NODE $1 END AT " >> "$LOGFL"
    date >> "$LOGFL"
}


#===============================================================================
# Run the script.

# Initialize the docker image.
docker_init
# Figure out what node id we have.
NODEID=$1 #TODO
# Then run the docker image for this node.
docker_run $NODEID

# That's it.
exit 0
