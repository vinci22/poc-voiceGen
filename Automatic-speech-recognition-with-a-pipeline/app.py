from datasets import load_dataset
from datasets import Audio

from transformers import pipeline

asr = pipeline("automatic-speech-recognition","facebook/wav2vec2-large-xlsr-53-spanish")

minds = load_dataset("PolyAI/minds14", name="es-ES", split="train")
minds = minds.cast_column("audio", Audio(sampling_rate=16_000))


example = minds[0]
print(f"real: {example["transcription"]}")
print(f"predicted: {asr(example["audio"]["array"])}")