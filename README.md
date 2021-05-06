# ADL Homework 2

## Installation

```bash
conda create --name <env_name> python=3.8
python -m pip install -r requirements.txt

# for eval.py
python -m spacy download zh_core_web_md
```

## Usage

### Training
*model saved at `./ckpt/<task_name>`*

```bash
# training by yourself
bash train_cls.sh  # train paragraphs classification model
bash train_qa.sh   # train question-answering model

# just download model checkpoint from dropbox
bash download.sh
```
### Prediction

```bash
bash run.sh <context_file_path> <test_file_path> <prediction_file_path>
```

## Reference
+ [`text-classification`](https://github.com/huggingface/transformers/tree/v4.5.0/examples/text-classification)
+ [`question-answering`](https://github.com/huggingface/transformers/tree/v4.5.0/examples/question-answering)