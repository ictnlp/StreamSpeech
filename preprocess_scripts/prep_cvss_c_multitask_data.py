import os
import tqdm
import argparse
import pandas as pd
import sentencepiece as spm

from pathlib import Path
from tempfile import NamedTemporaryFile
import sys
import os
import re

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from examples.speech_to_text.data_utils import (
    load_df_from_tsv,
    save_df_to_tsv,
    gen_vocab,
)
from examples.speech_synthesis.data_utils import ipa_phonemize


MANIFEST_COLUMNS = ["id", "tgt_text"]
SPLITS = ["train", "dev", "test"]


def learn_spm_vocab(args, train_text):
    with NamedTemporaryFile(mode="w") as f:
        for t in train_text:
            f.write(t + "\n")
        gen_vocab(
            Path(f.name),
            Path(args.output_dir).absolute() / f"spm_{args.vocab_type}_{args.lang}",
            args.vocab_type,
            args.vocab_size,
        )


def process(args):
    output_dir = Path(args.output_dir).absolute()
    output_dir.mkdir(exist_ok=True)
    if args.vocab_type in ["char", "unigram"]:
        train_text = []
        df = load_df_from_tsv(Path(args.data_dir).absolute() / "train.tsv")
        data = list(df.T.to_dict().values())
        for item in data:
            if args.is_src_text:
                item["src_text"] = re.sub(r"[^\w\s]", "", item["src_text"].lower())

                train_text.append(item["src_text"])
            else:
                item["tgt_text"] = re.sub(r"[^\w\s]", "", item["tgt_text"].lower())
                train_text.append(item["tgt_text"])
        learn_spm_vocab(args, train_text)
        sp = spm.SentencePieceProcessor(
            model_file=os.path.join(
                output_dir, f"spm_{args.vocab_type}_{args.lang}.model"
            )
        )
        for split in SPLITS:
            df = load_df_from_tsv(Path(args.data_dir).absolute() / f"{split}.tsv")
            data = list(df.T.to_dict().values())
            manifest = {c: [] for c in MANIFEST_COLUMNS}
            for item in data:
                manifest["id"].append(item["id"])
                if args.is_src_text:
                    item["src_text"] = re.sub(r"[^\w\s]", "", item["src_text"].lower())
                    manifest["tgt_text"].append(
                        " ".join(sp.encode(item["src_text"], out_type=str))
                    )
                else:
                    item["tgt_text"] = re.sub(r"[^\w\s]", "", item["tgt_text"].lower())
                    manifest["tgt_text"].append(
                        " ".join(sp.encode(item["tgt_text"], out_type=str))
                    )
            df = pd.DataFrame.from_dict(manifest)
            save_df_to_tsv(df, output_dir / f"{split}.tsv")
    else:
        for split in SPLITS:
            df = load_df_from_tsv(Path(args.data_dir).absolute() / f"{split}.tsv")
            data = list(df.T.to_dict().values())
            manifest = {c: [] for c in MANIFEST_COLUMNS}
            for item in tqdm.tqdm(data):
                manifest["id"].append(item["id"])
                if args.is_src_text:
                    item["src_text"] = re.sub(r"[^\w\s]", "", item["src_text"].lower())
                    manifest["tgt_text"].append(
                        ipa_phonemize(
                            item["src_text"], lang=args.lang, use_g2p=args.use_g2p
                        )
                    )
                else:
                    item["tgt_text"] = re.sub(r"[^\w\s]", "", item["tgt_text"].lower())
                    manifest["tgt_text"].append(
                        ipa_phonemize(
                            item["tgt_text"], lang=args.lang, use_g2p=args.use_g2p
                        )
                    )
            df = pd.DataFrame.from_dict(manifest)
            save_df_to_tsv(df, output_dir / f"{split}.tsv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument(
        "--is-src-text",
        action="store_true",
    )
    parser.add_argument(
        "--vocab-type",
        choices=["char", "phoneme", "unigram"],
        required=True,
    )
    parser.add_argument("--vocab-size", default=6000, type=int)
    parser.add_argument("--use-g2p", action="store_true")
    args = parser.parse_args()
    process(args)


if __name__ == "__main__":
    main()
