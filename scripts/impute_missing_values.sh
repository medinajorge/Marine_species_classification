#! /bin/bash

id=stdin_files/impute_env_data_v2

run -c 1 -m 20 -t 10:00 -o "$id".out -e "$id".err "python impute_missing_values.py"
