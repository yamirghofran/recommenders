#! /bin/sh

# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

GROUP="$1"
TYPE="$2"
CONFIG_FILE="$3"

# # Run tests in parallel
# pytest -n auto --durations 0 $(yq ".${TYPE}.${GROUP} | map(@sh) | join(\" \")" "${CONFIG_FILE}")

nvidia-smi
