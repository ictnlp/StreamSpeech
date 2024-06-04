import tqdm
import argparse
import pandas as pd
import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
from pathlib import Path
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

from examples.speech_to_text.data_utils import (
    load_df_from_tsv,
    save_df_to_tsv,
    gen_vocab,
)
from examples.speech_synthesis.data_utils import ipa_phonemize

from examples.speech_to_text.data_utils import (
    load_df_from_tsv,
    save_df_to_tsv,
    gen_config_yaml,
)
from fairseq.data.audio.data_cfg import S2SDataConfig

MANIFEST_COLUMNS = [
    "id",
    "audio",
    "n_frames",
    "src_text",
    "tgt_text",
    "speaker",
    "src_lang",
    "tgt_lang",
]
LANG_TAG_TEMPLATE = "<lang:{}>"


def process(args):
    s2st_tsv_dir = Path(args.s2st_tsv_dir)
    s2tt_tsv_dir = Path(args.s2tt_mtl_tsv_dir)
    s2tt_tsv_dir.mkdir(exist_ok=True)
    train_text = []
    for split in ["train", "dev", "test"]:
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        df = load_df_from_tsv(s2st_tsv_dir / f"{split}.tsv")
        data = list(df.T.to_dict().values())
        for item in tqdm.tqdm(data):
            item["src_text"] = re.sub(r"[^\w\s]", "", item["src_text"].lower())
            manifest["id"].append(item["id"])
            manifest["audio"].append(item["src_audio"])
            manifest["n_frames"].append(item["src_n_frames"])
            manifest["src_text"].append(item["src_text"])
            manifest["tgt_text"].append(item["tgt_text"])
            manifest["speaker"].append("None")
            manifest["src_lang"].append(args.src_lang)
            manifest["tgt_lang"].append(args.tgt_lang)
            if split == "train":
                train_text.append(item["src_text"])
                train_text.append(item["tgt_text"])
        df = pd.DataFrame.from_dict(manifest)
        save_df_to_tsv(df, s2tt_tsv_dir / f"{split}.tsv")

    with NamedTemporaryFile(mode="w") as f:
        for t in train_text:
            f.write(t + "\n")
        gen_vocab(
            Path(f.name),
            Path(s2tt_tsv_dir).absolute() / f"spm_unigram_joint",
            "unigram",
            args.vocab_size,
            special_symbols=[
                LANG_TAG_TEMPLATE.format(args.tgt_lang),
                LANG_TAG_TEMPLATE.format(args.src_lang),
            ],
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s2st-tsv-dir")
    parser.add_argument("--s2tt-mtl-tsv-dir")
    parser.add_argument("--src-lang")
    parser.add_argument("--tgt-lang")
    parser.add_argument("--vocab-size")
    args = parser.parse_args()
    process(args)


if __name__ == "__main__":
    main()
