#!/usr/bin/env bash
#SBATCH --job-name=lalf-orca
#SBATCH --array=1-{{VALID_COUNT}}
#SBATCH --cpus-per-task={{NPROCS}}

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
ORCA_BIN="${ORCA_BIN:-orca}"
SCRATCH_ROOT="${LALF_SCRATCH_ROOT:-${TMPDIR:-/tmp}}"
FILE_NUMBER="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${SUBMIT_DIR}/ids.txt")"

if [[ -z "${FILE_NUMBER}" ]]; then
  echo "Missing file number for array task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

if [[ -f "${SUBMIT_DIR}/${FILE_NUMBER}.done" ]]; then
  exit 0
fi

WORK_DIR="$(mktemp -d "${SCRATCH_ROOT}/lalf_${SLURM_JOB_ID}_${FILE_NUMBER}_XXXX")"
cleanup() { rm -rf "${WORK_DIR}"; }
trap cleanup EXIT

cp "${SUBMIT_DIR}/a${FILE_NUMBER}.xyz" "${WORK_DIR}/"
cp "${SUBMIT_DIR}/${FILE_NUMBER}.inp" "${WORK_DIR}/"
cp "${SUBMIT_DIR}/${FILE_NUMBER}_td.inp" "${WORK_DIR}/"
cd "${WORK_DIR}"

if ! "${ORCA_BIN}" "${FILE_NUMBER}.inp" > "${FILE_NUMBER}.out" 2>&1; then
  cp "${FILE_NUMBER}.out" "${SUBMIT_DIR}/" || true
  touch "${SUBMIT_DIR}/${FILE_NUMBER}.failed"
  exit 10
fi
if ! grep -q "ORCA TERMINATED NORMALLY" "${FILE_NUMBER}.out"; then
  cp "${FILE_NUMBER}.out" "${SUBMIT_DIR}/"
  touch "${SUBMIT_DIR}/${FILE_NUMBER}.failed"
  exit 11
fi
if [[ ! -f "${FILE_NUMBER}.xyz" ]]; then
  cp "${FILE_NUMBER}.out" "${SUBMIT_DIR}/"
  touch "${SUBMIT_DIR}/${FILE_NUMBER}.failed"
  exit 12
fi

if ! "${ORCA_BIN}" "${FILE_NUMBER}_td.inp" > "${FILE_NUMBER}_td.out" 2>&1; then
  cp "${FILE_NUMBER}.out" "${FILE_NUMBER}_td.out" "${SUBMIT_DIR}/" || true
  touch "${SUBMIT_DIR}/${FILE_NUMBER}.failed"
  exit 20
fi
cp "${FILE_NUMBER}.out" "${FILE_NUMBER}_td.out" "${FILE_NUMBER}.xyz" "${SUBMIT_DIR}/"
if grep -q "ORCA TERMINATED NORMALLY" "${FILE_NUMBER}_td.out"; then
  touch "${SUBMIT_DIR}/${FILE_NUMBER}.done"
  rm -f "${SUBMIT_DIR}/${FILE_NUMBER}.failed"
else
  touch "${SUBMIT_DIR}/${FILE_NUMBER}.failed"
  exit 21
fi

