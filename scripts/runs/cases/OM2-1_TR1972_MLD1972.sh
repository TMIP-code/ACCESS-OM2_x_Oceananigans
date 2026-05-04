#!/usr/bin/env bash
# Sourced by scripts/runs/run_case.sh.
# Test case: TR×MLD comparison (4/4) — TR=1972, MLD=1972.

export PARENT_MODEL=ACCESS-OM2-1
export EXPERIMENT=1deg_jra55_iaf_omip2_cycle6
export TIME_WINDOW=1972
export MLD_TIME_WINDOW=1972

export JOB_CHAIN=TMbuild..plotNK
export TM_SOURCE=const
export LINEAR_SOLVER=Pardiso
export LUMP_AND_SPRAY=yes
export MONTHLY_KAPPAV=yes
