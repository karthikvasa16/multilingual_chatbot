import streamlit as st
import torchaudio
import torch
import tempfile
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq

# Set page config
st.set_page_config(page_title="Whisper Speech-to-Text", layout="centered")

st.title("🎙️ Speech-to-Text with OpenAI Whisper")

# Load model and processor
@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained("openai/whisper-large")
    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large")
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        stride_length_s=(5, 2),
        device=0 if torch.cuda.is_available() else -1,
    )
    return pipe

pipe = load_model()

# File uploader
uploaded_file = st.file_uploader("Upload an audio file (.wav, .mp3, .flac)", type=["wav", "mp3", "flac"])

# Optional: record using mic (browser support needed)
st.markdown("Or use [browser recorder](https://voicecoach.ai/record/) to record and download your audio file.")

if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(uploaded_file.read())
        tmp_path = tmpfile.name

    st.info("Transcribing...")
    try:
        waveform, sample_rate = torchaudio.load(tmp_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        
        transcription = pipe(waveform.squeeze().numpy())
        st.success("Transcription complete!")
        st.text_area("📝 Transcribed Text", transcription["text"], height=200)
    except Exception as e:
        st.error(f"❌ Failed to transcribe: {e}")
