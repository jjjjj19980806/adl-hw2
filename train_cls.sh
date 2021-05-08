# prepare datasets
python dataset_cls.py --do_train

# training
python text-classification/run_glue.py \
  --model_name_or_path hfl/chinese-macbert-large \
  --train_file ./cache/cls_train.json \
  --validation_file ./cache/cls_dev.json \
  --do_train \
  --do_eval \
  --max_seq_length 512 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_step 16 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --output_dir ./ckpt/cls/ \
  --overwrite_output_dir yes \
  --fp16 yes \
  --evaluation_strategy steps \
  --eval_steps 100 \
  --report_to wandb \
  --run_name ADL2-macbert-large-cls \
