from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from pydub import AudioSegment
from models.ecapa import ECAPA
from models.emotion2vec import Emotion2Vec

#Convert mp3 to wav
def ensure_wav(path):
    path = Path(path)
    if path.suffix.lower() == ".wav":
        return path
    wav_path = path.with_suffix(".wav")
    audio = AudioSegment.from_file(path)
    audio.set_frame_rate(16000).set_channels(1).export(wav_path, format="wav")
    return wav_path

#Load audio files
audio_path1 = ensure_wav("D:/StreamSpeech/demo/uploads/input.common_voice_es_18311418.mp3")
audio_path2 = ensure_wav("D:/StreamSpeech/demo/uploads/output.common_voice_es_18311418.mp3")

#Extract speaker embeddings 
ecapa = ECAPA(device="cpu")
speaker_vec = ecapa.extract_speaker_embeddings(audio_path1).unsqueeze(0).cpu().numpy()
speaker_vec_out = ecapa.extract_speaker_embeddings(audio_path2).unsqueeze(0).cpu().numpy()

#Extract emotion embeddings
emotion2vec = Emotion2Vec(device="cpu")
emotion_vec = emotion2vec.extract_emotion_embeddings(audio_path1).unsqueeze(0).cpu().numpy()
emotion_vec_out = emotion2vec.extract_emotion_embeddings(audio_path2).unsqueeze(0).cpu().numpy()
emotion_sim = cosine_similarity(emotion_vec, emotion_vec_out)[0][0]

#Compute cosine similarity between two embeddings (speaker and emotion)
speaker_sim = cosine_similarity(speaker_vec, speaker_vec_out)[0][0]
emotion_sim = cosine_similarity(emotion_vec, emotion_vec_out)[0][0]
print(f"Speaker Cosine similarity: {speaker_sim}")
print(f"Emotion Cosine similarity: {emotion_sim}")
