import os
import argparse
import pandas as pd
from examples.speech_to_text.data_utils import load_df_from_tsv, save_df_to_tsv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-tsv", type=str, required=True)
    parser.add_argument("--output-tsv", type=str, required=True)
    args = parser.parse_args()
    df = load_df_from_tsv(args.input_tsv)
    data = list(df.T.to_dict().values())
    for item in data:
        item["tgt_text"] = item["tgt_text"][::2]
    df = pd.DataFrame.from_dict(data)
    save_df_to_tsv(df, args.output_tsv)


if __name__ == "__main__":
    main()
