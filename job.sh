#!/bin/bash -l
# ^ Make sure I get a login shell (so I can load modules!)

# N.b.The directory the job was submitted from is available as $PBS_O_WORKDIR

# Number of Nodes (default 1), CPUs (default 1), RAM (default 1) per individual job
# Better to use select command, rather than individual commands
#PBS -l select=1:ncpus=4:mem=32GB:ngpus=1:gpu_id=H100

# Max wall time per individual job - format is H:MM:SS (default 1hr) - don't bother specifying seconds
#PBS -l walltime=18:00:00

# Configure job array to run 50 replicate experiments
# The python sacred library will take care of giving each experiment a unique
# seed and putting results in a unique directory
#     #PBS -J 1-2:1

# Send mail events for (a)bort, (b)egin, and (e)nd
#PBS -m abe

# Merge standard error into the stdout file
#PBS -j oe

# Set job name
#PBS -N qwen3-30b

# Activate Python3.6 with experimental libraries already installed
# Best to use $HOME rather than /mnt/etc...

# Load a python version
# Get Python ready
module purge

# Fix for eResearch JOB-971: if we're running on an A100 node, we need to set MODULEPATH differently to avoid a Python core dump error
# N.b. also, A100 nodes only support [module load GCCcore/13.2.0 Python/3.11.5] and [module load GCCcore/12.3.0 Python/3.11.3]
# A100s DO NOT support [module load GCCore/13.3.0 Python/3.12.3] - this only works on H100s
if [[ $HOSTNAME =~ "gpu0" ]]
  then
  export MODULEPATH=/mnt/weka/pkg/rhel94/AMD-25-1/modules/all/Core
fi

module load GCCcore/13.2.0 Python/3.11.5
#module load GCCcore/12.3.0 Python/3.11.3
#module load GCCcore/14.2.0 Python/3.13.1


# Check python version
python --version

# Make temp venv ...
#rm -rf $PBS_O_WORKDIR/llm-dce-py-tmp
#python -m venv $PBS_O_WORKDIR/llm-dce-py-tmp
#source $PBS_O_WORKDIR/llm-dce-py-tmp/bin/activate
#python --version
#pip install --upgrade pip
#pip install -r $PBS_O_WORKDIR/requirements.txt

# ... OR Source existing venv
source $PBS_O_WORKDIR/../llm-dce-py/bin/activate
python --version

# Ensure requirements are up-to-date
pip install -r $PBS_O_WORKDIR/requirements.txt

# Load CUDA
module load CUDA/12.8.0

# Check GPU is available
nvidia-smi

# Print CUDA devices
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.current_device()); print(torch.cuda.get_device_name(0))"

# Choose a random port to run ollama on
OLLAMA_RND_PORT=
while [ -z ${OLLAMA_RND_PORT} ]
do
    let "OLLAMA_TMP_RND_PORT = 11434 + ${RANDOM:0:3}"
    echo "Checking port ${OLLAMA_TMP_RND_PORT} is free"
    nc -z localhost "${OLLAMA_TMP_RND_PORT}" || OLLAMA_RND_PORT=${OLLAMA_TMP_RND_PORT}
done
echo "Port: $OLLAMA_RND_PORT is free"

export OLLAMA_HOST=127.0.0.1:${OLLAMA_RND_PORT}
export OLLAMA_API_BASE=http://127.0.0.1:${OLLAMA_RND_PORT}

# Launch ollama server in background
nohup $PBS_O_WORKDIR/../ollama/bin/ollama serve &

# Wait for ollama server to spin up
echo "Waiting for Ollama server to start..."
for i in {1..30}; do
    if $PBS_O_WORKDIR/../ollama/bin/ollama list > /dev/null 2>&1; then
        echo "Ollama server is ready!"
        break
    fi
    echo "Attempt $i/30: Server not ready yet, waiting..."
    sleep 2
done

# Check ollama is working 
$PBS_O_WORKDIR/../ollama/bin/ollama list

# Kick off run
cd $PBS_O_WORKDIR
python -m llm_dce --num_responses 1000 "ollama/qwen3:30b"
