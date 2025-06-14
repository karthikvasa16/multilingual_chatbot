import streamlit as st
import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForCTC
from tempfile import NamedTemporaryFile

# Set title
st.title("🎤 Speech to Text Transcriber (Wav2Vec2)")

# Load model and processor
@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
    model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    return processor, model

processor, model = load_model()

# Upload file
uploaded_file = st.file_uploader("Upload a WAV audio file (16kHz mono)", type=["wav"])

if uploaded_file:
    with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Load audio
    waveform, sample_rate = torchaudio.load(tmp_path)

    # Resample if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # Display audio player
    st.audio(tmp_path)

    # Prepare input
    inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits

    # Decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    # Display result
    st.markdown("### 📝 Transcription Result:")
    st.success(transcription)
