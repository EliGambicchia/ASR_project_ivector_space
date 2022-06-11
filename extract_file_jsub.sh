#!/bin/sh
#$ -N extract_files
#$ -cwd
#$ -l h_rt=24:00:00
#$ -l h_vmem=32G
#$ -pe gpu-titanx 1
#$ -o /exports/eddie/scratch/s2065084/extract_files.stdout
#$ -e /exports/eddie/scratch/s2065084/extract_files.stderr
#$ -M s2065084@ed.ac.uk
#$ -m beas

source /etc/profile.d/modules.sh

module load cuda/11.0.2
module load anaconda
source activate slptorch

. /exports/applications/support/set_cuda_visible_devices.sh

# can only set these after conda setup
set -euo pipefail

tar xzvf /exports/eddie/scratch/s2065084/en.tar.gz
