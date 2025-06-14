import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForCTC
import os
from pathlib import Path

# 📌 Load the model and processor
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# 📌 Upload audio file (must be 16kHz, mono WAV file)
uploaded_file = st.file_uploader("Upload an audio file (.wav, .mp3, .flac)", type=["wav", "mp3", "flac"])

waveform, sample_rate = torchaudio.load(uploaded_file)

# 📌 Resample to 16000 Hz if needed
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)
    sample_rate = 16000



# 📌 Inference
input_values = processor(waveform.squeeze(), sampling_rate=sample_rate, return_tensors="pt").input_values
with torch.no_grad():
    logits = model(input_values).logits

# 📌 Decode prediction
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)[0]

# 📌 Display result
print("📝 Transcribed Text:")
print(transcription)
