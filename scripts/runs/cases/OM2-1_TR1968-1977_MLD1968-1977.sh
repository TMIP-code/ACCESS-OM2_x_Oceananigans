#!/usr/bin/env bash
# Sourced by scripts/runs/run_case.sh.
# Test case: TR×MLD comparison (1/4) — TR=1968-1977, MLD=1968-1977.
# Both windows equal; routed under test/ (alongside the off-diagonal cases)
# so all 4 cases live in one comparison root.

export PARENT_MODEL=ACCESS-OM2-1
export EXPERIMENT=1deg_jra55_iaf_omip2_cycle6
export TIME_WINDOW=1968-1977
export MLD_TIME_WINDOW=1968-1977

export JOB_CHAIN=TMbuild..plotNK
export TM_SOURCE=const
export LINEAR_SOLVER=Pardiso
export LUMP_AND_SPRAY=yes
export MONTHLY_KAPPAV=yes
