#!/usr/bin/env bash

# Shared machine-local path loader.
#
# Priority:
# 1. Existing environment variable
# 2. config/local_paths.env
# 3. Repository-derived ALPASIM_ROS2_WS

_PATH_LOADER_DIRECTORY="$(
    cd "$(dirname "${BASH_SOURCE[0]}")" &&
    pwd
)"

_DERIVED_ROS2_WS="$(
    cd "${_PATH_LOADER_DIRECTORY}/.." &&
    pwd
)"

_LOCAL_PATH_CONFIG="$(
    printf '%s/config/local_paths.env' \
        "${_DERIVED_ROS2_WS}"
)"

_read_config_value() {
    local requested_key="$1"
    local key
    local value

    if [[ ! -f "${_LOCAL_PATH_CONFIG}" ]]; then
        return 1
    fi

    while IFS='=' read -r key value; do
        key="${key#"${key%%[![:space:]]*}"}"
        key="${key%"${key##*[![:space:]]}"}"

        if [[ -z "${key}" ]] ||
           [[ "${key}" == \#* ]]; then
            continue
        fi

        if [[ "${key}" == "${requested_key}" ]]; then
            value="${value#"${value%%[![:space:]]*}"}"
            value="${value%"${value##*[![:space:]]}"}"
            value="${value%\"}"
            value="${value#\"}"
            value="${value%\'}"
            value="${value#\'}"
            printf '%s\n' "${value}"
            return 0
        fi
    done < "${_LOCAL_PATH_CONFIG}"

    return 1
}

if [[ -z "${ALPASIM_ROS2_WS:-}" ]]; then
    ALPASIM_ROS2_WS="$(
        _read_config_value ALPASIM_ROS2_WS ||
        printf '%s\n' "${_DERIVED_ROS2_WS}"
    )"
fi

if [[ -z "${ALPASIM_ROOT:-}" ]]; then
    ALPASIM_ROOT="$(
        _read_config_value ALPASIM_ROOT
    )"
fi

if [[ -z "${ALPASIM_DATA_ROOT:-}" ]]; then
    ALPASIM_DATA_ROOT="$(
        _read_config_value ALPASIM_DATA_ROOT
    )"
fi

if [[ -z "${ALPASIM_ROOT}" ]]; then
    echo \
        "ALPASIM_ROOT is not configured in ${_LOCAL_PATH_CONFIG}" \
        >&2
    return 1
fi

if [[ -z "${ALPASIM_DATA_ROOT}" ]]; then
    echo \
        "ALPASIM_DATA_ROOT is not configured in ${_LOCAL_PATH_CONFIG}" \
        >&2
    return 1
fi

export ALPASIM_ROOT
export ALPASIM_ROS2_WS
export ALPASIM_DATA_ROOT

unset _PATH_LOADER_DIRECTORY
unset _DERIVED_ROS2_WS
unset _LOCAL_PATH_CONFIG
