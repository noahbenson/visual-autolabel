#!/bin/bash

prjroot=$(dirname $(dirname $(realpath "$0")))
path="$prjroot/dave_params/dbstr"

db=$(cat "$path")
conda run -n va-vd optuna-dashboard "$db"
