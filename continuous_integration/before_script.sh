#!/bin/sh

set -e

if [[ "$COVERAGE" == "false" ]]; then
     make clean
fi