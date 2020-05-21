#!/bin/bash
#SBATCH --mail-user=salari.m1375@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=def-mori_gpu
#SBATCH --job-name=pytorch-classification
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --mem=0
#SBATCH --gres=gpu:p100l:4
#SBATCH --cpus-per-task=12

cd $SLURM_TMPDIR
cp -r ~/scratch/pytorch-classification .
cd pytorch-classification

module load python/3.7 cuda/10.0
virtualenv --no-download venv
source venv/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

python imagenet.py -a resnet18 --epochs 90 --schedule 31 61 --gamma 0.1 \
-c checkpoints/imagenet/resnet18 --workers 0 --log-interval 100 
