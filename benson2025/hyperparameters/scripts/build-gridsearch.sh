#! /bin/bash

# Load the tag from the tag environment.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"/../..
source ./config.env

# Run the build command.
docker build --tag "$GRIDSEARCH_DOCKERTAG" --no-cache $PWD

# If a directory is given, write the docker image into that directory.
[ -n "$1" ] && [ -d "$1" ] && {
    FLNM="$1"/"${GRIDSEARCH_FILENAME}"
    echo "Saving ${GRIDSEARCH_DOCKERTAG} to ${GRIDSEARCH_FILENAME} ..."
    docker save -o "$FLNM" "${GRIDSEARCH_DOCKERTAG}"
    chmod 644 "$FLNM"
}

exit 0
