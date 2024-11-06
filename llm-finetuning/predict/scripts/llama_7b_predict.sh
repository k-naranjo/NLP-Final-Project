#!/bin/bash -l

# --- Resource related ---
#SBATCH -A cs6501-nlp-24fa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 72:00:00 # Day-Hour:Minute
#SBATCH -p gpu
#SBATCH --gres=gpu:3
#SBATCH --mem-per-cpu=64GB

# --- Task related ---

#SBATCH --job-name=llama2_7b_predict_XXX

#SBATCH --output="/scratch/UVAid/llm-finetuning/lora/logfiles/predict//%x_%j.log" 
#SBATCH --error="/scratch/UVAid/llm-finetuning/lora/logfiles/predict//%x_%j.err"

# ijob -A cs6501-nlp-24fa --ntasks=1 --cpus-per-task=8 -t 1:00:00 -p gpu --gres=gpu:1 --mem-per-cpu=64GB

LORA_CHECKPOINT_DIR="/scratch/UVAid/llm-finetuning/llama2_7b/checkpoint-XXX/"
MAX_NEW_TOKENS=300
EVAL_FILENAME="/scratch/UVAid/llm-finetuning/data/dev.json"
OUTPUT_FILENAME="/scratch/UVAid/llm-finetuning/lora/logfiles/predict/llama2_7b_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.json"
DO_SAMPLE=False
TEMPLATE_NAME="toxigen_predict"
MODEL_NAME='meta-llama/Llama-2-7b-hf'
BEAM_SIZE=1
BATCH_SIZE=32


# ---- env---------
export ENV_LOCATION="/scratch/UVAid/name_env"
export HF_CACHE_LOCATION="/scratch/UVAid/huggingface"

pwd

echo "HOSTNAME -> $HOSTNAME"

nvidia-smi

module load anaconda

export HF_DATASETS_CACHE="${HF_CACHE_LOCATION}/datasets"
export TRANSFORMERS_CACHE="${HF_CACHE_LOCATION}/hub"

conda activate ${ENV_LOCATION}
echo "which python -> $(which python)"
start=$(date +%s)

python -c "import torch.cuda; print('torch.cuda.is_available():', torch.cuda.is_available())"

python /home/UVAid/scratch_UVAid/llm-finetuning/lora/predict.py \
    --eval_filename $EVAL_FILENAME \
    --base_model_name $MODEL_NAME \
    --output_filename $OUTPUT_FILENAME \
    --lora_checkpoint_dir $LORA_CHECKPOINT_DIR \
    --prompt_template_name $TEMPLATE_NAME \
    --do_sample $DO_SAMPLE \
    --max_new_tokens $MAX_NEW_TOKENS \
    --num_beams $BEAM_SIZE \
    --batch_size $BATCH_SIZE

end=$(date +%s)

secs=$((end-start))

printf 'Runtime: %dh:%dm:%ds\n' $((secs/3600)) $((secs%3600/60)) $((secs%60))  
echo "Finished"
