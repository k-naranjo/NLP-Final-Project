#!/bin/bash -l

# --- Resource related ---
#SBATCH -A uvailp-llm
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 72:00:00 # Day-Hour:Minute
#SBATCH -p gpu
#SBATCH --gres=gpu:3
#SBATCH --mem-per-cpu=64GB

# --- Task related ---
#SBATCH --job-name="llama2_7b_karo_test"
#SBATCH --output="/scratch/kn3cs/NLP-Final-Project/llm-finetuning/train/logfiles/%x_%j.log"
#SBATCH --error="/scratch/kn3cs/NLP-Final-Project/llm-finetuning/train/logfiles/%x_%j.err"

export FPATH_LOG="/scratch/kn3cs/NLP-Final-Project/llm-finetuning/lora/scripts/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
export TRAIN_FILE="/scratch/kn3cs/NLP-Final-Project/data/train.json"
export VAL_FILE="/scratch/kn3cs/NLP-Final-Project/data/val.json"
export TEMPLATE_NAME="toxigen_karo"
export MODEL_NAME='meta-llama/Llama-2-7b-hf'
export FP16=False
export LEARNING_RATE="3e-6"
export NUM_EPOCHS=10

export OUTPUT_DIR="/scratch/kn3cs//NLP-Final-Project/llm-finetuning/outputs/${SLURM_JOB_NAME}/_${SLURM_JOB_ID}/"
mkdir -p ${OUTPUT_DIR}

echo 'Created Output Directory'
# $ wandb login

export ENV_LOCATION="/home/kn3cs/nlproject"
export HF_CACHE_LOCATION="/scratch/kn3cs/huggingface"

pwd

echo "HOSTNAME -> $HOSTNAME"

nvidia-smi

module load anaconda cuda cudnn

export HF_DATASETS_CACHE="${HF_CACHE_LOCATION}/datasets"
export TRANSFORMERS_CACHE="${HF_CACHE_LOCATION}/hub"

conda info --envs
conda env list
conda activate "${ENV_LOCATION}"
conda activate ${ENV_LOCATION}
echo "which python -> $(which python)"
start=$(date +%s)
python -c "import torch.cuda; print('torch.cuda.is_available():', torch.cuda.is_available())"

python /home/kn3cs/scratch_kn3cs/NLP-Final-Project/llm-finetuning/lora/train.py \
    --base_model_name "${MODEL_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --train_file "${TRAIN_FILE}" \
    --val_file "${VAL_FILE}" \
    --prompt_template_name "${TEMPLATE_NAME}" \
    --learning_rate "${LEARNING_RATE}" \
    --num_epochs "${NUM_EPOCHS}" \
    --fp16 "${FP16}"

end=$(date +%s)

secs=$((end-start))

printf 'Runtime: %dh:%dm:%ds\n' $((secs/3600)) $((secs%3600/60)) $((secs%60))  
echo "Finished Training"
