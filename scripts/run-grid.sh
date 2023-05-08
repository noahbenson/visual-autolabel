#! /bin/bash

TRAIN="/opt/visual-autolabel/scripts/train.py"
GENPARAMS="/opt/visual-autolabel/scripts/gen-gridparams.py"
GRIDPARAMS_JSON="/opt/visual-autolabel/grid-search/grid-params.json"

# This script runs the grid-search for a single node. The required arguments
# are (1) the starting grid-search cell, (2) the stride of the search, and (3)
# the total number of cells in the search.
# Note that this script is designed to work only inside the docker-image
# produced from the visual-autolabel GitHub repo.

start="$1"
stride="$2"
n="$3"
if [ -z "$start" ] || [ -z "$stride" ] || [ -z "$n" ]
then echo "SYNTAX: grid-search.sh <start> <stride> <cellcount>"
     exit 1
fi

# Iterate through our grid cells.
for (( ii=$start; ii<$n; ii=$(( $ii + $stride )) ))
do key="`printf 'grid%05d' $ii`"
   # We generate the files we need for this particular run; they will get
   # automatically saved into the model directory by the training script.
   "$GENPARAMS" "$GRIDPARAMS_JSON" "$ii"
   # Now run the training.
   "$TRAIN" "$key" opts.json plan.json #&> /dev/null
done

# That's it.
exit 0
