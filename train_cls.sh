# prepare datasets
python dataset_cls.py --do_train

# training
python text-classification/run_glue.py \
  --model_name_or_path bert-base-chinese \
  --train_file ./cache/cls_train.json \
  --validation_file ./cache/cls_dev.json \
  --do_train \
  --do_eval \
  --max_seq_length 512 \
  --per_device_train_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ./ckpt/cls/ \
  --overwrite_output_dir yes \
  --fp16 yes \
  # --resume_from_checkpoint ./ckpt/cls/checkpoint-xxx \


## per_device_train_batch_size 8   for 2080-ti
## per_device_train_batch_size 32  for 3090
  