#!/bin/bash

hcp_restricted_path='/home/dambam/data/hcp/meta/RESTRICTED_full.csv'
params_dir="dave_params"

thisPath="$(dirname $(realpath "$0"))"
basePath="$(dirname $thisPath)"
lastPath=$(pwd)


if [[ -z $1 ]]; then
    echo "Must specify modelKey for arg1"
    exit 1
else
    modelKey="$1"
fi
if [[ -z $2 ]]; then
    pyenv=dave
else
    pyenv="$2"
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
    echo $basePath
    echo $pyenv
    conda activate $pyenv && \
        env NPYTHYRC=/home/dambam/Code/visual-autolabel/.npythyrc \
        python -m visual_autolabel.benson2025.hcp.train "$model_key_p" "$optsFile" "$planFile"

}

case "$modelKey" in
    # Test
    test0)
        partition=area
        inputs=anat
        region=central
        planName=plan1
        ;;
    test1)
        partition=area
        inputs=trac
        region=dorsal
        planName=plan1
        ;;
    hyper1)
        partition=area
        inputs=trac
        region=dorsal
        planName=planHyper
        ;;
    # Dorsal
    h_d_anat)
        partition=area
        inputs=anat
        region=dorsal
        planName=planHyper
        ;;
    h_d_t1t2)
        partition=area
        inputs=t1t2
        region=dorsal
        planName=planHyper
        ;;
    h_d_func)
        partition=area
        inputs=func
        region=dorsal
        planName=planHyper
        ;;
    h_d_trac)
        partition=area
        inputs=trac
        region=dorsal
        planName=planHyper
        ;;
    h_d_full)
        partition=area
        inputs=full
        region=dorsal
        planName=planHyper
        ;;
    # Ventral
    h_v_anat)
        partition=area
        inputs=anat
        region=ventral
        planName=planHyper
        ;;
    h_v_t1t2)
        partition=area
        inputs=t1t2
        region=ventral
        planName=planHyper
        ;;
    h_v_func)
        partition=area
        inputs=func
        region=ventral
        planName=planHyper
        ;;
    h_v_trac)
        partition=area
        inputs=trac
        region=ventral
        planName=planHyper
        ;;
    h_v_full)
        partition=area
        inputs=full
        region=ventral
        planName=planHyper
        ;;
    *)
        echo "Invalid modelKey"
        exit 1
        ;;
esac
main "$modelKey" "$partition" "$inputs" "$region" "$planName"

