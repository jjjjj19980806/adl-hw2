# prepare datasets
python dataset_qa.py --do_train

# training
python question-answering/run_qa.py \
  --model_name_or_path hfl/chinese-macbert-large \
  --train_file ./cache/qa_train.json \
  --validation_file ./cache/qa_dev.json \
  --do_train \
  --do_eval \
  --max_seq_length 512 \
  --doc_stride 128 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_step 16 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --output_dir ./ckpt/qa/ \
  --overwrite_output_dir yes \
  --fp16 yes \
  --evaluation_strategy steps \
  --eval_steps 100 \
  --report_to wandb \
  --run_name ADL2-macbert-large-qa \
