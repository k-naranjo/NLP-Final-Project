#!/bin/bash -l

# --- Resource related ---
#SBATCH -A grad-nlp-cs6501
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 72:00:00 # Day-Hour:Minute
#SBATCH -p gpu
#SBATCH --gres=gpu:3
#SBATCH --mem-per-cpu=64GB

# --- Task related ---

#SBATCH --job-name= "llama2_7b_name_predict"
#SBATCH --output="/scratch/UVAid/NLP-Final-Project/llm-finetuning/predict/logfiles/%x_%j.log"
#SBATCH --error="/scratch/UVAid/NLP-Final-Project/llm-finetuning/predict/logfiles/%x_%j.err"


LORA_CHECKPOINT_DIR="/scratch/UVAid/NLP-Final-Project/llm-finetuning/llama2_7b/checkpoint-XXX/"
MAX_NEW_TOKENS=500
EVAL_FILENAME="/scratch_UVAid/NLP-Final-Project/data/dev.json"
OUTPUT_FILENAME="/scratch/UVAid/NLP-Final-Project/llm-finetuning/lora/predict/outputs/llama2_7b_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.json"
DO_SAMPLE=False
TEMPLATE_NAME="toxigen_predict_teammate_name"
MODEL_NAME='meta-llama/Llama-2-7b-hf'
BEAM_SIZE=1
BATCH_SIZE=32


# ---- env---------
export ENV_LOCATION="/home/UVAid/name_env"
export HF_CACHE_LOCATION="/scratch/UVAid/huggingface"

pwd

echo "HOSTNAME -> $HOSTNAME"

nvidia-smi

module load anaconda cuda cudnn

export HF_DATASETS_CACHE="${HF_CACHE_LOCATION}/datasets"
export TRANSFORMERS_CACHE="${HF_CACHE_LOCATION}/hub"

conda activate ${ENV_LOCATION}
echo "which python -> $(which python)"
start=$(date +%s)

python -c "import torch.cuda; print('torch.cuda.is_available():', torch.cuda.is_available())"

python /home/UVAid/scratch_UVAid//NLP-Final-Project/llm-finetuning/lora/predict.py \
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
