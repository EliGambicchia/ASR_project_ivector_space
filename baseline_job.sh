#!/bin/sh
#$ -N baseline_diversity
#$ -cwd
#$ -l h_rt=18:00:00
#$ -l h_vmem=60G
#$ -pe gpu 1
#$ -o /exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s2065084_Elisa_Gambicchia/baseline_diversity.stdout
#$ -e /exports/chss/eddie/ppls/groups/lel_hcrc_cstr_students/s2065084_Elisa_Gambicchia/baseline_diversity.stderr
#$ -M s2065084@ed.ac.uk
#$ -m beas

source /etc/profile.d/modules.sh

module load cuda/11.0.2
module load anaconda
source activate slptorch

. /exports/applications/support/set_cuda_visible_devices.sh

#ulimit -v

# can only set these after conda setup
set -euo pipefail

bash kaldi-accents1/my_run.sh