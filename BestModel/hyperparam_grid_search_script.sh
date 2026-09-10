#!/bin/bash
#SBATCH --job-name=deberta_grid
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/bitbucket/cl6523/nlp_cw/grid_logs.out
#SBATCH --error=/vol/bitbucket/cl6523/nlp_cw/grid_logs.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cl6523

source /vol/cuda/11.4.120-cudnn8.2.4/setup.sh

source /vol/bitbucket/cl6523/nlp_cw/.venvbeech11/bin/activate

echo "To project directory"

cd /vol/bitbucket/cl6523/nlp_cw/

TERM=vt100
/usr/bin/nvidia-smi
uptime

echo "Starting the massive Python grid search..."
python BestModel/hyperparameter_script.py
echo "Python script has finished running."