import io
import contextlib

import torch
from funasr import AutoModel


class Emotion2Vec:
    def __init__(self, variant="iic/emotion2vec_plus_base", device="cuda"):
        """
        Available emotion2vec model variants:
        - "iic/emotion2vec_base"
        - "iic/emotion2vec_base_finetuned"
        - "iic/emotion2vec_plus_seed"
        - "iic/emotion2vec_plus_base" (default)
        - "iic/emotion2vec_plus_large"
        """
        self.device = device
        self.emotion2vec = AutoModel(
            model=variant,
            hub="hf",  # Use hugging face
            device=self.device
        )

    @torch.no_grad()
    def extract_emotion_embeddings(self, wav_path, output_dir=None):
        # Suppress tqdm / logging output
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            raw_result = self.emotion2vec.generate(
                input=wav_path.as_posix(),
                output_dir=output_dir,
                granularity="utterance",
                extract_embedding=True,
                verbose=False,
                progress=False,
            )[0]

        embeddings = torch.as_tensor(
            data=raw_result["feats"],
            dtype=torch.float32,
            device=self.device
        ).flatten()

        assert embeddings.ndim == 1, f"Expected 1D embedding, got {embeddings.shape}"
        return embeddings

    # @torch.no_grad()
    # def extract_emotion_embeddings(self, wav_path, output_dir=None):
    #     raw_result = self.emotion2vec.generate(
    #         input=wav_path,
    #         output_dir=output_dir,
    #         granularity="utterance",
    #         extract_embedding=True,
    #         verbose=False,
    #         progress=False
    #     )[0]

    #     # emotion2vec outputs a numpy array
    #     # So, convert it to a 1D torch tensor
    #     # raw_result["feats"] contain the actual embeddings
    #     embeddings = torch.as_tensor(
    #         data=raw_result["feats"],
    #         dtype=torch.float32,
    #         device=self.device
    #     ).flatten()

    #     # To check if embeddings shape is 1D
    #     assert embeddings.ndim == 1, f"Expected 1D embedding, got {embeddings.shape}"

    #     return embeddings
