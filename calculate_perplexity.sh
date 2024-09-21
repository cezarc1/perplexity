#!/bin/bash

echoerr() { printf "\e[31;1m%s\e[0m\n" "$*" >&2; }

if ! command -v uv &> /dev/null
then
    echo "uv could not be found."
    read -p "Would you like to install uv now? [y/N]: " response
    case "$response" in
        [yY][eE][sS]|[yY])
            echo "Installing uv..."
            curl -LsSf https://astral.sh/uv/install.sh | sh
            if ! command -v uv &> /dev/null
            then
                echoerr "Failed to install uv. Please install uv manually and retry. See https://docs.astral.sh/uv/guides/install-python/"
                exit 1
            else
                echo "uv installed successfully."
            fi
            ;;
        *)
            echoerr "uv is required to run this script. Exiting."
            exit 1
            ;;
    esac
fi

uv run --with-requirements requirements.txt calculate_perplexity.py "$@"
