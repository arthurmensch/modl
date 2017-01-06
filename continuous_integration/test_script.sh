#!/bin/sh

set -e

python continuous_integration/show-python-packages-versions.py;
# We want to back out of the current working directory to make
# sure we are using nilearn installed in site-packages rather
# than the one from the current working directory
# Parentheses (run in a subshell) are used to leave
# the current directory unchanged
if [[ "$COVERAGE" == "true" ]]; then
    cp $OLDPWD/.coveragerc "$TEST_RUN_FOLDER"
    (cd "$TEST_RUN_FOLDER" && make -f $OLDPWD/Makefile test-coverage)
else
    (cd "$TEST_RUN_FOLDER" && make -f $OLDPWD/Makefile test)
fi