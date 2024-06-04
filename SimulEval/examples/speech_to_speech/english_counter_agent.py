from simuleval.utils import entrypoint
from simuleval.data.segments import SpeechSegment
from simuleval.agents import SpeechToSpeechAgent
from simuleval.agents.actions import WriteAction, ReadAction
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface


class TTSModel:
    def __init__(self):
        models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
            "facebook/fastspeech2-en-ljspeech",
            arg_overrides={"vocoder": "hifigan", "fp16": False},
        )
        TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
        self.tts_generator = task.build_generator(models, cfg)
        self.tts_task = task
        self.tts_model = models[0]
        self.tts_model.to("cpu")
        self.tts_generator.vocoder.to("cpu")

    def synthesize(self, text):
        sample = TTSHubInterface.get_model_input(self.tts_task, text)
        if sample["net_input"]["src_lengths"][0] == 0:
            return [], 0
        for key in sample["net_input"].keys():
            if sample["net_input"][key] is not None:
                sample["net_input"][key] = sample["net_input"][key].to("cpu")

            wav, rate = TTSHubInterface.get_prediction(
                self.tts_task, self.tts_model, self.tts_generator, sample
            )
            wav = wav.tolist()
            return wav, rate


@entrypoint
class EnglishSpeechCounter(SpeechToSpeechAgent):
    """
    Incrementally feed text to this offline Fastspeech2 TTS model,
    with a minimum numbers of phonemes every chunk.
    """

    def __init__(self, args):
        super().__init__(args)
        self.wait_seconds = args.wait_seconds
        self.tts_model = TTSModel()

    @staticmethod
    def add_args(parser):
        parser.add_argument("--wait-seconds", default=1, type=int)

    def policy(self):
        length_in_seconds = round(
            len(self.states.source) / self.states.source_sample_rate
        )
        if not self.states.source_finished and length_in_seconds < self.wait_seconds:
            return ReadAction()
        samples, fs = self.tts_model.synthesize(f"{length_in_seconds} mississippi")

        # A SpeechSegment has to be returned for speech-to-speech translation system
        return WriteAction(
            SpeechSegment(
                content=samples,
                sample_rate=fs,
                finished=self.states.source_finished,
            ),
            finished=self.states.source_finished,
        )
