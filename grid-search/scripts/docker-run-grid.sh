#! /bin/bash

TRAIN="/opt/visual-autolabel/scripts/train.py"
GENPARAMS="/opt/visual-autolabel/grid-search/scripts/gen-gridparams.py"
GRIDPARAMS_JSON="/opt/visual-autolabel/grid-search/grid-params.json"
DATA=/data
RUNDIR=/data/run
OUTDIR=/data/models
INDIR=/data/inputs

# This script runs the grid-search for a single node. The required arguments
# are (1) the starting grid-search cell, (2) the stride of the search, and (3)
# the total number of cells in the search.
# Note that this script is designed to work only inside the docker-image
# produced from the visual-autolabel GitHub repo.

start="$1"
stride="$2"
n="$3"
device="$4"
if [ -z "$start" ] || [ -z "$stride" ] || [ -z "$n" ]
then echo "SYNTAX: grid-search.sh <start> <stride> <cellcount>"
     exit 1
fi

# The OVERWRITE environment variable may be set in order to specify that the
# script should overwrite (instead of skip) previously run models.
if [ -z "$OVERWRITE" ] || [ "$OVERWRITE" = "no" ] || [ "$OVERWRITE" = "n" ]
then OVERWRITE=no
elif [ "$OVERWRITE" = "yes" ] || [ "$OVERWRITE" = "y" ]
then OVERWRITE=yes
else echo "OVERWRITE must be \"yes\" or \"no\""
     exit 2
fi

# How many times does each model get run?
UNTIL=$(grep until "$GRIDPARAMS_JSON" \
            | sed -E 's/^ *"until" *: *([0-9]+) *[,}]? *$/\1/')

# Log prefix.
echo "# Node Runtime, $start"
echo ""
echo "## Parameters"
echo '```yaml'
echo "start: $start"
echo "stride: $stride"
echo "end: $n"
echo '```'
echo ""
echo "## Log"

# Iterate through our grid cells.
for (( ii=$start; ii<$n; ii=$(( $ii + $stride )) ))
do key="`printf 'grid%05d' $ii`"
   rundir="${RUNDIR}/cells/${key}"
   outdir="${OUTDIR}/${key}"
   logfn="${outdir}/training.log"
   # If we're not overwriting and the output directory already has a full log,
   # then we can skip this entry.
   [ "$OVERWRITE" = "no" ] \
       && [ -r "$logfl" ] \
       && [ $(grep '^Best val loss' "$logfl" | wc -l) -eq $((3*$UNTIL)) ] \
       && continue
   mkdir -p "$rundir" && cd "$rundir"
   # We generate the files we need for this particular run; they will get
   # automatically saved into the model directory by the training script.
   "$GENPARAMS" "$GRIDPARAMS_JSON" $ii
   # If we're not running on CPU, fix the options.
   if [ -n "$device" ] && [ "$device" != "cpu" ]
   then mv opts.json opts_nodev.json
	cat opts_nodev.json | sed -e 's/}$/, "device": "'"${device}"'"}/' > opts.json
   fi
   # Now run the training.
   echo "* KEY: $key  "
   echo "  BEGIN $key: `date`  "
   "$TRAIN" "$key" opts.json plan.json &> ./run.log
   echo "  END $key: `date`"
done

# That's it.
exit 0
