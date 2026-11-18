#!/usr/bin/env bash
set -euo pipefail
mapping_package="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$mapping_package/.venv-maps/bin/python" "$mapping_package/code/robot_mapping/pipeline.py" "$@"
