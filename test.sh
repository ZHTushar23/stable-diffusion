#!/bin/bash 
#SBATCH --job-name=stable        # Job name
#SBATCH --mail-user=ztushar1@umbc.edu       # Where to send mail
#SBATCH --mem=20000                       # Job memory request
#SBATCH --gres=gpu:1                     # Number of requested GPU(s) 
#SBATCH --time=72:00:00                   # Time limit days-hrs:min:sec
#SBATCH --constraint=rtx_6000            # Specific hardware constraint
#SBATCH --error=stable_exp.err                # Error file name
#SBATCH --output=stable_exp.out               # Output file name

export CUDA_LAUNCH_BLOCKING=1
# python scripts/txt2img.py --prompt "a photograph of an astronaut riding a horse" --plms 
python demo.py
