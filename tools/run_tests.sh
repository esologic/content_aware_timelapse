#!/usr/bin/env bash

# Run all pytest unit tests in the ./content_aware_timelapse:./test directory.
# Pass an optional pytest marker expression as the first arg (default: all),
# followed by any pytest args or options (e.g. --write-assets, -vv, etc.).

set -euo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd "${DIR}/.."
source ./.venv/bin/activate

export PYTHONPATH=".:./test${PYTHONPATH+:}${PYTHONPATH:-}"

# Detect if first argument is a pytest option (starts with "-")
if [[ ${1:-} =~ ^- ]]; then
    TEST_MARKED="all"
else
    TEST_MARKED=${1:-all}
    shift 1 || true
fi

echo "run tests: ${TEST_MARKED}; args: ${PYTEST_EXTRA_ARGS:-} $@"

pytest -m "${TEST_MARKED/#all/}" ${PYTEST_EXTRA_ARGS:-} "$@"
