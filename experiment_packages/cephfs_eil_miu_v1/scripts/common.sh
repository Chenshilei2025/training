#!/usr/bin/env bash

resolve_project_root() {
  if [[ -n "${LOYAL_PROJECT_ROOT:-}" ]]; then
    printf '%s\n' "${LOYAL_PROJECT_ROOT}"
    return 0
  fi
  local script_dir pkg_root candidate
  script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  pkg_root="$(cd -- "${script_dir}/.." && pwd)"
  candidate="$(cd -- "${pkg_root}/../.." && pwd)"
  if [[ -f "${candidate}/scripts/experiment_runner.py" ]]; then
    printf '%s\n' "${candidate}"
    return 0
  fi
  if [[ -f "${pkg_root}/../../scripts/experiment_runner.py" ]]; then
    printf '%s\n' "$(cd -- "${pkg_root}/../.." && pwd)"
    return 0
  fi
  echo "cannot find training repo root; set LOYAL_PROJECT_ROOT" >&2
  return 2
}
