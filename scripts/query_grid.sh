#!/bin/bash

# partition, inputs, base_model

key=$1
dir=/data/visual-autolabel/grid/grid-search/
#dir=/data/visual-autolabel/grid/models/

find "$dir" -type f -name "*.json" | while read -r file; do
    grep -o '"'"$key"'": [^,]*' "$file" | sed 's/"'"$key"'": //'
done | sort | uniq
