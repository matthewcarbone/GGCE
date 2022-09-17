#!/bin/bash

pip install toml
for var in "$@"; do
    if [[ "$var" == "tests" ]]; then
        python3 -c 'import toml; c = toml.load("pyproject.toml"); print("\n".join(c["project"]["optional-dependencies"]["test"]))' | pip install -r /dev/stdin
    elif [[ "$var" == "docs" ]]; then
        python3 -c 'import toml; c = toml.load("pyproject.toml"); print("\n".join(c["project"]["optional-dependencies"]["doc"]))' | pip install -r /dev/stdin
    elif [[ "$var" == "core" ]]; then
        python3 -c 'import toml; c = toml.load("pyproject.toml"); print("\n".join(c["project"]["dependencies"]))' | pip install -r /dev/stdin
    fi
done
