#all_inputs=("anat" "t1t2" "func" "trac" "not2" "nofn" "nodw" "full" "null")
# nofn, t1t2, full
all_inputs=("anat" "full" "func" "trac")
#all_inputs=("nofn" "full" "func")
all_models=( "resnet18" "resnet34" )
regions=( "ventral" "dorsal" "central")
plan="hyper"

thisPath="$(dirname $(realpath "$0"))"
for region in "${regions[@]}"; do
    for model in "${all_models[@]}"; do
        for inputs in "${all_inputs[@]}"; do
            echo '*************************************************************************************'
            echo '*************************************************************************************'
            "$thisPath"/train.sh "$plan"_"$region" --inputs "$inputs" --base_model "$model" --resume_type exit_if_exists --error_on_interrupt "$@"
            if [[ ! $? -eq 0 ]]; then
                exit $?
            fi
        done
    done
done

