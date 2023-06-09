#!/bin/sh
#SBATCH --job-name=your-job-name
#SBATCH --time=72:00:00

### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --chdir=/home/cap6412.student28/pytorch-slurm-multi-node-demo
#SBATCH --output=/home/cap6412.student28/pytorch-slurm-multi-node-demo/%x-%j.out

module purge
module load cuda
module load gcc/gcc-9.1.0
module load oneapi/mkl

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12340
export WORLD_SIZE=4

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

### init virtual environment if needed
source ./venv/bin/activate

### the command to run
srun python main.py --lr 1e-3 --epochs 50
