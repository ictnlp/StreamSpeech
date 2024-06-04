import tqdm
import argparse
import pandas as pd
import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
from pathlib import Path
from examples.speech_to_text.data_utils import (
    load_df_from_tsv,
    save_df_to_tsv,
    gen_config_yaml,
)
from fairseq.data.audio.data_cfg import S2SDataConfig

MANIFEST_COLUMNS = ["audio", "tgt_text"]


def process(args):
    cvss_dir = Path(args.cvss_dir)
    covost2_dir = Path(args.covost2_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    for split in ["train", "dev", "test"]:
        Path(f"{out_dir}/{split}").mkdir(exist_ok=True)
        with open(cvss_dir / f"{split}.tsv", "r") as f:
            data = f.read().splitlines()
        with open(f"{out_dir}/{split}/wav_list.txt", "w") as f_wav:
            with open(f"{out_dir}/{split}/target.txt", "w") as f_tgt:
                for x in data:
                    wav, tgt = x.split("\t")
                    f_wav.write(f"{covost2_dir}/clips/{wav}" + "\n")
                    f_tgt.write(tgt + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cvss-dir")
    parser.add_argument("--covost2-dir")
    parser.add_argument("--out-dir")
    args = parser.parse_args()
    process(args)


if __name__ == "__main__":
    main()
