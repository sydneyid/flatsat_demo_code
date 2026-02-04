#!/usr/bin/env bash
# Run record_filter_display.py (record from camera, filter, save RAW + MP4, display).
# Usage: ./run_record_filter_display.sh [optional args, e.g. -o /path/to/output]
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEMO_DIR="${DEMO_DIR:-/home/jetson/flatsat_demo_code}"
cd "$DEMO_DIR"
exec python record_filter_display.py "$@"
