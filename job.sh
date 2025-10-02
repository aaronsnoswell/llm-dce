#!/bin/bash -l
# ^ Make sure I get a login shell (so I can load modules!)

# N.b.The directory the job was submitted from is available as $PBS_O_WORKDIR

# Number of Nodes (default 1), CPUs (default 1), RAM (default 1) per individual job
# Better to use select command, rather than individual commands
#PBS -l select=1:ncpus=4:mem=32GB:ngpus=1

# Max wall time per individual job - format is H:MM:SS (default 1hr) - don't bother specifying seconds
#PBS -l walltime=0:15:00

# Configure job array to run 50 replicate experiments
# The python sacred library will take care of giving each experiment a unique
# seed and putting results in a unique directory
#     #PBS -J 1-2:1

# Send mail events for (a)bort, (b)egin, and (e)nd
#PBS -m abe

# Merge standard error into the stdout file
#PBS -j oe

# Set job name
#PBS -N mistral-dce

# Activate Python3.6 with experimental libraries already installed
# Best to use $HOME rather than /mnt/etc...

# Load a python version
# Get Python ready
module purge
#module load GCCcore/13.2.0 Python/3.11.5
#module load GCCcore/12.3.0 Python/3.11.3
module load GCCcore/14.2.0 Python/3.13.1
#source $PBS_O_WORKDIR/../llm-dce-py/bin/activate

# Check python version
python --version

# Make temp venv
python -m venv $PBS_O_WORKDIR/llm-dce-py-tmp
source $PBS_O_WORKDIR/llm-dce-py-tmp/bin/activate

# Check python version
python --version

# Install reqs
pip install --upgrade pip
pip install -r $PBS_O_WORKDIR/requirements.txt

# Load CUDA
module load CUDA/12.8.0

# Check GPU is available
nvidia-smi

# Launch ollama server in background
$PBS_O_WORKDIR/../ollama/bin/ollama serve > /dev/null 2>&1 &

# Check ollama is working 
$PBS_O_WORKDIR/../ollama/bin/ollama list

# Try running mistral run
python -m llm_dce --num_responses 10 "ollama/mistral:latest"
