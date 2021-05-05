#
#   Usage example
#     bash ./run.sh /path/to/context.json /path/to/public.json  /path/to/pred/public.json
#     bash ./run.sh /path/to/context.json /path/to/private.json /path/to/pred/private.json
#

#####  CLS part #####

python dataset_cls.py --do_predict

python text-classification/run_glue.py \
  --model_name_or_path ./ckpt/cls \
  --train_file ./cache/cls_train.json \
  --validation_file ./cache/cls_dev.json \
  --test_file ./cache/cls_test.json \
  --do_predict \
  --max_seq_length 512 \
  --output_dir ./cache/ \
  --fp16 yes \

##### QA part #####

python dataset_qa.py --do_predict

# python question-answering/run_qa.py \
#   --model_name_or_path ./ckpt/qa \
#   --train_file ./cache/qa_train.json \
#   --validation_file ./cache/qa_dev.json \
#   --test_file ./cache/qa_test.json \
#   --do_predict \
#   --max_seq_length 384 \
#   --doc_stride 128 \
#   --output_dir ./cache/ \
#   --fp16 yes \
