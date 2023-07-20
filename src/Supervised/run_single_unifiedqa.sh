export TASK_NAME=negqa

MAKE_NAME=$1 # # # # # I am including this to be able to use allenai, google, "" models, etc.
MODEL_NAME=$2 #unifiedqa-v2-t5-base-1251000, unifiedqa-v2-t5-large-1251000, unifiedqa-v2-t5-3b-1251000
TRAIN_FILES=(unifiedqa)

DATA_DIR="../../data/unifiedqa_formatted_data/"

SEEDS=(70) # 69 68 67 66)

mkdir -p $3

if $4 ; then
  action="deepspeed run_negatedqa_t5.py --per_device_eval_batch_size 1 --gradient_accumulation_steps 1 --deepspeed deepspeed_config_2.json "
  echo "USING DEEPSPEED"
else
  action="python run_negatedqa_t5.py "
  echo "NOT using deepspeed"
fi

export TEST_FILE=unifiedqa

for SEED in "${SEEDS[@]}"; do
  for SETTING in "${TRAIN_FILES[@]}"; do

      OUTPUT_DIR=$3/${MODEL_NAME}_negation_all_${SEED}_train_${SETTING}_test_${TEST_FILE}
      mkdir -p $OUTPUT_DIR

      if $5 ; then
        OUTPUT_DIR=$MAKE_NAME$MODEL_NAME
        echo "CONTINUING TRAINING ON EXISTING CHECKPOINT AT $OUTPUT_DIR"
      else
        echo "TRAINING FROM BASE MODEL ($MAKE_NAME$MODEL_NAME)"
      fi

      $action \
        --model_name_or_path $MAKE_NAME$MODEL_NAME \
        --train_file ${DATA_DIR}/condaqa_train_unifiedqa.json \
        --validation_file ${DATA_DIR}/condaqa_dev_unifiedqa.json \
        --test_file ${DATA_DIR}/condaqa_test_unifiedqa.json \
        --do_train \
        --per_device_train_batch_size 12 \
        --gradient_accumulation_steps 2 \
        --learning_rate 1e-5 \
        --num_train_epochs 5 \
        --output_dir $OUTPUT_DIR \
        --logging_strategy epoch \
        --evaluation_strategy epoch \
        --report_to wandb \
        --save_strategy epoch \
        --overwrite_cache \
        --seed $SEED \
        --summary_column answer \
        --text_column input \
        --source_prefix "" \
        --max_source_length 512 \
        --max_target_length 16 \
        --load_best_model_at_end True\
        --overwrite_output_dir # # # > ${MODEL_NAME}_results_all_${SEED}_train_${SETTING}_test_${TEST_FILE}.txt # # # We don't need this output to a file
  done
done

