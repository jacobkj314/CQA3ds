MODEL_NAME=$1 #unifiedqa-v2-t5-base-1251000, unifiedqa-v2-t5-large-1251000, unifiedqa-v2-t5-3b-1251000
SEEDS=(70) #(70 69 68 67 66)
PREDICTION_DIR=$2

for SEED in "${SEEDS[@]}"; do
  # # # # #python compute_unifiedqa_stats.py --model_name $MODEL_NAME --seed $SEED --predictions_dir $PREDICTION_DIR
  deepspeed compute_unifiedqa_stats.py --per_device_eval_batch_size 1 --gradient_accumulation_steps 1 --deepspeed deepspeed_config.json --model_name $MODEL_NAME --seed $SEED --predictions_dir $PREDICTION_DIR
done


