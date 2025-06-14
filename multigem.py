import streamlit as st
from googletrans import Translator, LANGUAGES
import google.generativeai as genai

# Initialize
translator = Translator()
genai.configure(api_key="AIzaSyDexffYjmTQRUfLPtfkd65yrCXRgYr0S9c")
model = genai.GenerativeModel("gemini-1.5-flash")

# All Google-supported languages (name → code)
LANGUAGE_CODES = {name.capitalize(): code for code, name in LANGUAGES.items()}
LANGUAGE_NAMES = sorted(LANGUAGE_CODES.keys())  # Alphabetical order

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ui_lang" not in st.session_state:
    st.session_state.ui_lang = "English"

# Sidebar: Language selector
st.sidebar.title("🌐 Language Settings")
selected_lang = st.sidebar.selectbox("Choose Interface Language:", LANGUAGE_NAMES)
st.session_state.ui_lang = selected_lang
lang_code = LANGUAGE_CODES[selected_lang]

# Translation helper
def t(text):
    if lang_code == "en":
        return text
    return translator.translate(text, dest=lang_code).text

# App Title
st.title("💬 Gemini 1.5 Multilingual Chatbot")

# Show message history
for msg in st.session_state.messages:
    content = msg.get("parts", [msg.get("content")])[0]
    st.chat_message(msg["role"]).markdown(content)

# User input
user_input = st.chat_input(t("Type your message..."))

if user_input:
    # Translate user input to English (Gemini's input)
    translated_input = translator.translate(user_input, dest="en").text

    # Save and show original user input
    st.session_state.messages.append({"role": "user", "parts": [user_input]})
    st.chat_message("user").markdown(user_input)

    # Ask Gemini
    gemini_input = {"role": "user", "parts": [translated_input]}
    response = model.generate_content([gemini_input])
    reply = response.text

    # Translate response to selected language
    translated_reply = translator.translate(reply, dest=lang_code).text

    # Save and show translated Gemini response
    st.session_state.messages.append({"role": "model", "parts": [translated_reply]})
    st.chat_message("model").markdown(translated_reply)
