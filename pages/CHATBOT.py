# pages/Chatbot.py
# Page Chatbot avec style Bloomberg et intégration Grok API
# Ajoute dans .streamlit/secrets.toml : XAI_API_KEY = "ta_cle_xai_api"

import streamlit as st
import os
from openai import OpenAI
import base64
import io
import time

# Configuration de la page
st.set_page_config(
    page_title="Chatbot Grok",
    page_icon="🤖",
    layout="wide"
)

# Style Bloomberg (noir/orange)
st.markdown("""
<style>
    .main {
        background-color: #000000;
        color: #FFAA00;
        padding: 20px;
    }
    .stButton > button {
        background-color: #333333;
        color: #FFAA00;
        border: 1px solid #FFAA00;
        font-weight: bold;
        border-radius: 0;
        font-family: 'Courier New', monospace;
    }
    .stButton > button:hover {
        background-color: #FFAA00;
        color: #000000;
    }
    h1, h2, h3 {
        color: #FFAA00 !important;
        font-family: 'Courier New', monospace !important;
    }
    .stChatMessage {
        background-color: #1a1a1a;
        border: 1px solid #FFAA00;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header Bloomberg
st.markdown(f"""
<div style="background: linear-gradient(90deg, #FFAA00, #FF6600); padding: 15px; color: #000000; font-weight: bold; font-size: 20px; font-family: 'Courier New', monospace; text-align: center; margin-bottom: 20px;">
    🤖 BLOOMBERG GROK CHATBOT | {time.strftime("%H:%M:%S UTC")}
</div>
""", unsafe_allow_html=True)

# Initialisation de la session
if "messages" not in st.session_state:
    st.session_state.messages = []

if "client" not in st.session_state:
    api_key = st.secrets.get("XAI_API_KEY")
    if not api_key:
        st.error("❌ Ajoute XAI_API_KEY dans .streamlit/secrets.toml (obtiens-la sur https://x.ai/api)")
        st.stop()
    st.session_state.client = OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1"
    )

# Fonction pour encoder l'image en base64
def encode_image(image_file):
    if image_file is not None:
        image_bytes = image_file.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        mime_type = image_file.type or 'image/jpeg'
        return f"data:{mime_type};base64,{image_base64}"
    return None

# Affichage des messages précédents
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "image" in message and message["image"]:
            st.image(message["image"], caption="Uploaded Image")

# Barre de chat avec upload image
col1, col2 = st.columns([4, 1])
with col1:
    user_input = st.chat_input("Pose ta question à Grok... (ou upload une image pour analyse)")

with col2:
    uploaded_file = st.file_uploader("📷 Image", type=["png", "jpg", "jpeg"], key="file_uploader")

if user_input or uploaded_file:
    # Message utilisateur
    user_message = {"role": "user", "content": user_input or "", "image": None}
    if uploaded_file:
        user_image_b64 = encode_image(uploaded_file)
        if user_image_b64:
            user_message["content"] = user_input or "Analyse cette image :"
            user_message["image"] = uploaded_file.getvalue()  # Pour affichage

    with st.chat_message("user"):
        st.markdown(user_message["content"])
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Image")

    st.session_state.messages.append(user_message)

    # Appel à Grok API
    with st.chat_message("assistant"):
        with st.spinner("Grok réfléchit..."):
            if user_message["image"]:
                # Multimodal : texte + image
                response = st.session_state.client.chat.completions.create(
                    model="grok-vision-beta",  # Modèle vision si dispo, sinon fallback
                    messages=[
                        {"role": "system", "content": "Tu es Grok, un assistant intelligent et utile. Réponds de manière concise et précise."},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_message["content"]},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": user_image_b64}
                                }
                            ]
                        }
                    ],
                    max_tokens=1000,
                    temperature=0.7
                )
            else:
                # Texte seulement
                response = st.session_state.client.chat.completions.create(
                    model="grok-beta",  # Modèle par défaut
                    messages=[
                        {"role": "system", "content": "Tu es Grok, un assistant intelligent et utile. Réponds de manière concise et précise."},
                        {"role": "user", "content": user_message["content"]}
                    ],
                    max_tokens=1000,
                    temperature=0.7
                )

            grok_response = response.choices[0].message.content
            st.markdown(grok_response)

    # Ajout à la session
    st.session_state.messages.append({"role": "assistant", "content": grok_response})

# Bouton pour effacer la conversation
if st.button("🗑️ Effacer la conversation", type="secondary"):
    st.session_state.messages = []
    st.rerun()
