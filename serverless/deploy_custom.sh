#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

nuctl create project cvat

nuctl deploy --project-name cvat \
    --path "$SCRIPT_DIR/custom/opencv_dnn_fd/nuclio" \
    --platform local

nuctl deploy --project-name cvat \
    --path "$SCRIPT_DIR/custom/insightface_flm/nuclio" \
    --platform local

nuctl get function
