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
            data.append({
                'context': contexts_list[q['relevant']],
                'question': q['question'],
                'answers': {
                    "answer_start": [ans['start'] for ans in q['answers']],
                    "text": [ans['text'] for ans in q['answers']],
                },
                'id': q['id'],
            })
        
        num_training_sample = int(len(data) * (1 - args.dev_ratio))
        train_spilt = {'data': data[:num_training_sample]}
        args.qa_train_path.write_text(json.dumps(train_spilt, indent=2))
        logging.info(f'saving qa_train at {args.qa_train_path}')

        dev_spilt = {'data': data[num_training_sample:]}
        args.qa_dev_path.write_text(json.dumps(dev_spilt, indent=2))
        logging.info(f'saving qa_dev at {args.qa_dev_path}')

    if args.do_predict:
        
        prob = args.cls_pred_path.read_text()
        prob_list = prob.strip().split('\n')[1:]
        prob_list = [p.split('\t')[-1] for p in prob_list]

        test_data_list = json.loads(args.test_path.read_text())
        data = list()
        num_paragraphs = 0

        for q in test_data_list:
            logist = prob_list[num_paragraphs:num_paragraphs+len(q['paragraphs'])]
            num_paragraphs += len(q['paragraphs'])
            max_logist_id = np.argmax(logist)
            data.append({
                'context': contexts_list[q['relevant']] if args.cheat else contexts_list[q['paragraphs'][max_logist_id]],
                'question': q['question'],
                'answers': {
                    "answer_start": [-1],
                    "text": [""],
                },
                'id': q['id'],
                'relevant': q['relevant'] if args.cheat else q['paragraphs'][max_logist_id], # for debugging
            })
        
        assert(num_paragraphs == len(prob_list)), "length must matched !"
        
        test_spilt = {'data': data}
        args.qa_test_path.write_text(json.dumps(test_spilt, indent=2))
        logging.info(f'saving qa_test at {args.qa_test_path}')


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
        "--cls_pred_path",
        type=Path,
        help="Path to the paragraphs prediction file.",
        default="./cache/test_results_None.txt",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Directory to save the processed file.",
        default="./cache/",
    )

    parser.add_argument('--cheat', action='store_true')

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_predict', action='store_true')

    parser.add_argument("--qa_train_path", type=Path, default='./cache/qa_train.json')
    parser.add_argument("--qa_dev_path", type=Path, default='./cache/qa_dev.json')
    parser.add_argument("--qa_test_path", type=Path, default='./cache/qa_test.json')

    parser.add_argument("--dev_ratio", type=float, default=0.1)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(args)
