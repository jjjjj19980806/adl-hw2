import json
import logging
from pathlib import Path
from argparse import ArgumentParser
import numpy as np

logging.basicConfig(
    format="[%(levelname)s] %(message)s",
    level=logging.INFO
)


def main(args):

    contexts_list = json.loads(args.context_path.read_text())

    if args.do_train:

        train_data_list = json.loads(args.train_path.read_text())
        data = list()

        for q in train_data_list:
            # positive sample
            data.append({
                'sentence1': q['question'],
                'sentence2': contexts_list[q['relevant']],
                'label': 1,
            })

            # negative sample
            neg_id_set = set(q['paragraphs']) - set([q['relevant']])
            neg_id = np.random.choice(list(neg_id_set))
            data.append({
                'sentence1': q['question'],
                'sentence2': contexts_list[neg_id],
                'label': 0,
            })
        
        num_training_sample = int(len(data) * (1 - args.dev_ratio))
        train_spilt = {'data': data[:num_training_sample]}
        args.cls_train_path.write_text(json.dumps(train_spilt, indent=2))
        logging.info(f'saving cls_train at {args.cls_train_path}')

        dev_spilt = {'data': data[num_training_sample:]}
        args.cls_dev_path.write_text(json.dumps(dev_spilt, indent=2))
        logging.info(f'saving cls_dev at {args.cls_dev_path}')

    if args.do_predict:
        
        test_data_list = json.loads(args.test_path.read_text())
        data = list()

        for q in test_data_list:
            for p in q['paragraphs']:
                data.append({
                    'sentence1': q['question'],
                    'sentence2': contexts_list[p],
                    'label': 0,
                    'id': q['id'],
                    'paragraphs': p,
                })
        
        test_spilt = {'data': data}
        args.cls_test_path.write_text(json.dumps(test_spilt, indent=2))
        logging.info(f'saving cls_test at {args.cls_test_path}')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--context_path",
        type=Path,
        help="Path to the context file.",
        default="./dataset/context.json",
    )
    parser.add_argument(
        "--train_path",
        type=Path,
        help="Path to the train file.",
        default="./dataset/train.json",
    )
    parser.add_argument(
        "--test_path",
        type=Path,
        help="Path to the test file.",
        default="./dataset/public.json",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Directory to save the processed file.",
        default="./cache/",
    )

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_predict', action='store_true')

    parser.add_argument("--cls_train_path", type=Path, default='./cache/cls_train.json')
    parser.add_argument("--cls_dev_path", type=Path, default='./cache/cls_dev.json')
    parser.add_argument("--cls_test_path", type=Path, default='./cache/cls_test.json')

    parser.add_argument("--dev_ratio", type=float, default=0.1)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(args)
