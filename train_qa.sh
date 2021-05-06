# prepare datasets
python dataset_qa.py --do_train

# training
python question-answering/run_qa.py \
  --model_name_or_path hfl/chinese-roberta-wwm-ext \
  --train_file ./cache/qa_train.json \
  --validation_file ./cache/qa_dev.json \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --overwrite_output_dir yes \
  --output_dir ./ckpt/qa/ \
  --fp16 yes \
  # --evaluation_strategy steps \
  # --eval_steps 100 \
  # --report_to wandb \
  # --run_name qa_eval \