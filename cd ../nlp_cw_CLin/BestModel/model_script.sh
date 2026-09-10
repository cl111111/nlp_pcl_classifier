#!/bin/bash
#SBATCH --job-name=model_train
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/bitbucket/cl6523/nlp_cw/BestModel/model/logs.out
#SBATCH --error=/vol/bitbucket/cl6523/nlp_cw/BestModel/model/logs.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cl6523

source /vol/cuda/11.4.120-cudnn8.2.4/setup.sh

source /vol/bitbucket/cl6523/nlp_cw/.venvbeech11/bin/activate

echo "To project directory"

cd /vol/bitbucket/cl6523/nlp_cw/

TERM=vt100
/usr/bin/nvidia-smi
uptime

python BestModel/model.py