#
#   Usage example
#     bash ./run.sh /path/to/context.json /path/to/public.json  /path/to/pred/public.json
#     bash ./run.sh /path/to/context.json /path/to/private.json /path/to/pred/private.json
#
#     "${1}": path to the context file.
#     "${2}": path to the testing file.
#     "${3}": path to the output predictions.
#

#####  CLS part #####

python dataset_cls.py --do_train
python dataset_cls.py --do_predict --context_path "${1}" --test_path "${2}" 

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

python dataset_qa.py --do_train
python dataset_qa.py --do_predict --context_path "${1}" --test_path "${2}"

python question-answering/run_qa.py \
  --model_name_or_path ./ckpt/qa \
  --train_file ./cache/qa_train.json \
  --validation_file ./cache/qa_dev.json \
  --test_file ./cache/qa_test.json \
  --do_predict \
  --max_seq_length 512 \
  --doc_stride 128 \
  --output_dir ./cache/ \
  --fp16 yes \

cp ./cache/test_predictions.json "${3}"
