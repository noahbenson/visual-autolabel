#!/bin/bash

hcp_restricted_path='/home/dambam/data/hcp/meta/RESTRICTED_full.csv'
params_dir='dave_params'
npythyrc='/home/dambam/Code/visual-autolabel/.npythyrc'

dataset=HCP
segmentation=area

thisPath="$(dirname $(realpath "$0"))"
basePath="$(dirname $thisPath)"
lastPath=$(pwd)


if [[ -z $1 ]]; then
    echo "Must specify modelAlias for arg1"
    exit 1
else
    modelAlias="$1"
fi
shift 1

if [[ -n $2 ]] && conda env list | awk '{print $1}' | grep -qx "$2"; then
    pyenv="$2"
    shift 1
elif  [[ -n $2 ]] && [[ $2 != -* ]]; then
    pyenv='va-vd'
elif  [[ -z $2 ]]; then
    pyenv='va-vd'
else
    echo "Invalid conda environment $2"
    exit 1
fi

debug=0
i=1
j=2
for arg in "$@"; do
    if [[ "$arg" == "--inputs" ]]; then
        inputs="${!j}"
    fi
    if [[ "$arg" == "--base_model" ]]; then
        base_model="${!j}"
        base_model_str="_$base_model"
    fi
    if [[ "$arg" == "--debug" ]]; then
        debug=1
    fi
    ((i++))
    ((j++))
done

params="$@"

main () {
    model_key_p="$1"
    plan_name_p="$2"
    segmentation_p="$3"
    region_p="$4"
    inputs_p="$5"
    genOpt_p="$6"
    if [[ -z $genOpt_p ]]; then
        genOpt_p=0
    fi

    optsDir="$basePath/$params_dir/options/$segmentation_p/"
    planDir="$basePath/$params_dir/plans/$segmentation_p/"

    if [[ $genOpt -eq 0 ]] ;then
        optsFile="$optsDir$dataset"_"$inputs_p"_"$region_p".json
    else
        optsFile="$optsDir$dataset"_"$region_p".json
    fi
    planFile="$planDir"plan_"$plan_name_p".json
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
    echo "$model_key_p"
    echo "   Params:    $params"
    if [[ $debug -eq 1 ]]; then
        echo "   Path:      $basePath"
        echo "   Conda:     $pyenv"
        echo "   opts_file: $optsFile"
        echo "   plan_file: $planFile"
    fi
    #conda run -n "$pyenv" \
    conda activate "$pyenv" && \
        env NPYTHYRC="$npythyrc" \
        python -m visual_autolabel.benson2025.hcp.train "$model_key_p" "$optsFile" "$planFile" $params
    exit $?

}

case "$modelAlias" in
    # Test
    test0)
        region=central
        planName=1
        model_key="$modelAlias"
        ;;
    test1)
        inputs=trac
        region=dorsal
        planName=1
        model_key="$modelAlias"
        ;;
    hyper1)
        inputs=trac
        region=dorsal
        planName=hyper
        model_key="$modelAlias"
        ;;
    *)
        #h_d_anat)
        IFS='_' read -r -a split <<< "$modelAlias"
        planName="${split[0]}"
        region="${split[1]}"
        genOpt=1
        if [[ ${#split[@]} -eq 3 ]]; then
            inputs="${split[2]}"
        fi
        if [[ -n $inputs ]]; then
            inputs_str="_$inputs"
        else
            echo "No model inputs found"
            exit 1
        fi
        model_key="$planName""$inputs_str"_"$region"_"$segmentation""$base_model_str"
        #$inputs_$segmentation
        ;;
esac
main "$model_key" "$planName" "$segmentation" "$region" "$inputs" "$genOpt"

