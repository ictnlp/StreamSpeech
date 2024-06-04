import os
import argparse
import sys
import re

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
from examples.speech_to_text.data_utils import load_df_from_tsv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-tsv", type=str, required=True)
    parser.add_argument("--output-txt", type=str, required=True)
    args = parser.parse_args()
    df = load_df_from_tsv(args.input_tsv)
    data = list(df.T.to_dict().values())
    with open(args.output_txt, "w") as f:
        for item in data:
            f.write(re.sub(r"[^\w\s]", "", item["src_text"].lower()) + "\n")


if __name__ == "__main__":
    main()
