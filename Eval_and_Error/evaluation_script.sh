#!/bin/bash
#SBATCH --job-name=eval_dev_test
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/bitbucket/cl6523/nlp_cw/Eval_and_Error/logs.out
#SBATCH --error=/vol/bitbucket/cl6523/nlp_cw/Eval_and_Error/logs.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cl6523

source /vol/cuda/11.4.120-cudnn8.2.4/setup.sh

source /vol/bitbucket/cl6523/nlp_cw/.venvbeech11/bin/activate

echo "To project directory"

cd /vol/bitbucket/cl6523/nlp_cw/

TERM=vt100
/usr/bin/nvidia-smi
uptime

python Eval_and_Error/evaluation.py