#!/bin/bash

hcp_restricted_path='/home/dambam/data/hcp/meta/RESTRICTED_full.csv'
params_dir="dave_params"

thisPath="$(dirname $(realpath "$0"))"
basePath="$(dirname $thisPath)"
lastPath=$(pwd)


modelKey="$1"
if [[ -z $modelKey ]]; then
    echo "Must specify modelKey for arg1"
    exit 1
fi

main () {
    model_key_p="$1"
    partition_p="$2"
    inputs_p="$3"
    region_p="$4"
    plan_name_p="$5"

    optsDir="$basePath/$params_dir/options/$partition_p/"
    planDir="$basePath/$params_dir/plans/$partition_p/"

    optsFile="$optsDir"HCP_"$inputs_p"_"$region_p".json
    planFile="$planDir""$plan_name_p".json
    if [[ ! -f $optsFile ]]; then
        echo "optsFile $optsFile does not exist"
        exit 1
    fi
    if [[ ! -f $planFile ]]; then
        echo "planFile $planFile does not exist"
        exit 1
    fi
    trap 'cd "$pw"' EXIT ERR INT TERM

    cd "$basePath"
    eval "$(conda shell.bash hook)"

    #export NPYTHYRC="$HOME/Code/MRICnn/.npythyrc"
    conda activate dave && python -m visual_autolabel.benson2024.hcp.train "$model_key_p" "$optsFile" "$planFile" $hcp_restricted_path

}

case "$modelKey" in
    test1)
        partition=area
        inputs=trac
        region=dorsal
        planName=plan1
        ;;
    *)
        echo "Invalid modelKey"
        exit 1
        ;;
esac
main "$modelKey" "$partition" "$inputs" "$region" "$planName"

