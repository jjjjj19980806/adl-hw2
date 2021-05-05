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

    test_file = json.loads(args.test_file_path.read_text())
    y_true = np.array([q['relevant'] for q in test_file])

    pred_file = json.loads(args.pred_file_path.read_text())
    y_pred = np.array([q['relevant'] for q in pred_file['data']])

    assert(len(y_true) == len(y_pred))
    logging.info(f'acc = {np.mean(y_true == y_pred)}')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file_path",
        type=Path,
        default="./dataset/public.json",
    )
    parser.add_argument(
        "--pred_file_path",
        type=Path,
        default="./cache/qa_test.json",
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
