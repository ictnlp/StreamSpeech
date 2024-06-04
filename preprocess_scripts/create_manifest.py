import os
import tqdm
import argparse
import soundfile as sf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root")
    args = parser.parse_args()
    for split in ["train", "dev", "test"]:
        fin = open(os.path.join(args.data_root, f"{split}.tsv"))
        fout = open(os.path.join(args.data_root, f"{split}.txt"), "w")
        data = fin.read().splitlines()
        fout.write(args.data_root + "/" + split + "\n")
        for line in tqdm.tqdm(data, desc=split):
            src_audio, _ = line.split("\t")
            src_audio = src_audio + ".wav"
            n_frames = sf.info(os.path.join(args.data_root, split, src_audio)).frames
            fout.write(f"{src_audio}\t{n_frames}\n")


if __name__ == "__main__":
    main()
