import torch
import torchaudio
from speechbrain.inference.speaker import SpeakerRecognition


class ECAPA:

    def __init__(self, device="cuda"):
        self.ecapa = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={ "device": device }
        )
        self.device = device


    @torch.no_grad()
    def extract_speaker_embeddings(self, wav_path):
        signal, sample_rate = torchaudio.load(wav_path)

        # Move audio to same device as model
        signal = signal.to(self.device)

        # Returns tensor with shape [1, 1, 192] (need to convert to 1D)
        raw_embeddings = self.ecapa.encode_batch(signal)

        # Remove unneccesary dimensions (batch=1, time=1)
        # Reshapes to [192]
        embeddings = raw_embeddings.squeeze()
        return embeddings
