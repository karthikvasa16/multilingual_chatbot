import streamlit as st
import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForCTC
from tempfile import NamedTemporaryFile
from googletrans import Translator, LANGUAGES
import google.generativeai as genai

# === CONFIGURATION ===
st.sidebar.header("🔐 Gemini API Key")
api_key = st.sidebar.text_input("Paste your Gemini API Key:", type="password")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")
translator = Translator()

# === LANGUAGE SETTINGS ===
LANGUAGE_CODES = {name.capitalize(): code for code, name in LANGUAGES.items()}
LANGUAGE_NAMES = sorted(LANGUAGE_CODES.keys())  # Alphabetical order

if "messages" not in st.session_state:
    st.session_state.messages = []
if "ui_lang" not in st.session_state:
    st.session_state.ui_lang = "English"

st.sidebar.title("🌐 Language Settings")
selected_lang = st.sidebar.selectbox("Choose Interface Language:", LANGUAGE_NAMES)
st.session_state.ui_lang = selected_lang
lang_code = LANGUAGE_CODES[selected_lang]

# === TRANSLATION HELPER ===
def t(text):
    if lang_code == "en":
        return text
    return translator.translate(text, dest=lang_code).text

# === LOAD ASR MODEL ===
@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
    model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    return processor, model

processor, asr_model = load_model()

# === PAGE TITLE ===
st.title(t("🎤 Speech & Chat Multilingual Assistant"))

st.markdown(t("#### Upload audio to transcribe or chat below"))

# === SPEECH TO TEXT ===
st.subheader(t("🎧 Upload a WAV Audio File (16kHz mono)"))
uploaded_file = st.file_uploader(t("Choose an audio file:"), type=["wav"])

if uploaded_file:
    with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    waveform, sample_rate = torchaudio.load(tmp_path)

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    st.audio(tmp_path)

    inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = asr_model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    st.success(t("Transcription:") + f" {transcription}")

# === CHATBOT SECTION ===
st.subheader(t("💬 Gemini 1.5 Multilingual Chatbot"))

for msg in st.session_state.messages:
    content = msg.get("parts", [msg.get("content")])[0]
    st.chat_message(msg["role"]).markdown(content)

user_input = st.chat_input(t("Type your message..."))

if user_input:
    # Translate to English for Gemini
    translated_input = translator.translate(user_input, dest="en").text

    # Save original user message
    st.session_state.messages.append({"role": "user", "parts": [user_input]})
    st.chat_message("user").markdown(user_input)

    # Ask Gemini
    gemini_input = {"role": "user", "parts": [translated_input]}
    response = model.generate_content([gemini_input])
    reply = response.text

    # Translate Gemini's response to user language
    translated_reply = translator.translate(reply, dest=lang_code).text

    # Show and save reply
    st.session_state.messages.append({"role": "model", "parts": [translated_reply]})
    st.chat_message("model").markdown(translated_reply)
