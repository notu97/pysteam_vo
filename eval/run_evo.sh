#!/usr/bin/env bash
# M8: score trajectories on identical timestamps with evo.
# Usage (from ORB_based_pysteam/):  bash pysteam_vo/eval/run_evo.sh [pysteam_vo/results/m8/common]
#
# -as (Sim(3) alignment) is mandatory: monocular trajectories are up to scale.
set -euo pipefail
DIR="${1:-pysteam_vo/results/m8/common}"
cd "$DIR"
export MPLBACKEND=Agg
rm -f ./*_ape.zip ./*_rpe.zip
for m in steap vggt vo; do
  echo "================ ${m}: APE (Sim3) ================"
  evo_ape tum gt.tum "${m}.tum" -as --save_results "${m}_ape.zip" --no_warnings
  echo "================ ${m}: RPE (Sim3, delta 1 m) ================"
  evo_rpe tum gt.tum "${m}.tum" -as --delta 1 --delta_unit m --save_results "${m}_rpe.zip" --no_warnings
done
echo "================ APE comparison ================"
evo_res steap_ape.zip vggt_ape.zip vo_ape.zip --use_filenames --save_table ape_table.csv --no_warnings
echo "================ RPE comparison ================"
evo_res steap_rpe.zip vggt_rpe.zip vo_rpe.zip --use_filenames --save_table rpe_table.csv --no_warnings
