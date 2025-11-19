# pages/CHATBOT.py
# Version finale – zéro erreur, zéro warning

import streamlit as st
from groq import Groq
import base64
import time

# =============================================
# PAGE CONFIG + STYLE BLOOMBERG
# =============================================
st.set_page_config(page_title="Grok Chatbot", page_icon="Robot", layout="wide")

st.markdown("""
<style>
    .main {background:#000;color:#FFAA00;padding:20px;}
    .stButton>button {background:#333;color:#FFAA00;border:2px solid #FFAA00;font-weight:bold;border-radius:0;}
    .stButton>button:hover {background:#FFAA00;color:#000;}
    h1,h2,h3 {color:#FFAA00 !important;font-family:'Courier New',monospace !important;}
    .stChatMessage {background:#111;padding:15px;border-left:4px solid #FFAA00;border-radius:0;}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style="background:#FFAA00;padding:15px;color:#000;font-weight:bold;font-size:22px;font-family:'Courier New';text-align:center;">
    GROQ CHATBOT • LLAMA 3.2 90B + VISION • {time.strftime("%H:%M:%S")} UTC
</div>
""", unsafe_allow_html=True)

# =============================================
# CLIENT GROQ
# =============================================
if "groq_client" not in st.session_state:
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error("Clé GROQ_API_KEY manquante dans .streamlit/secrets.toml")
        st.stop()
    st.session_state.groq_client = Groq(api_key=api_key)

client = st.session_state.groq_client

# Historique
if "messages" not in st.session_state:
    st.session_state.messages = []

# =============================================
# AFFICHAGE HISTORIQUE
# =============================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("image"):
            st.image(msg["image"], width=500)

# =============================================
# INPUT + UPLOAD IMAGE (sans warning)
# =============================================
col_text, col_img = st.columns([5, 1])

with col_text:
    prompt = st.chat_input("Pose ta question ou demande une analyse d'image")

with col_img:
    # Label non vide + caché proprement → plus de warning
    uploaded_img = st.file_uploader(
        "Image",  # ← label obligatoire
        type=["png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed",  # ← cache le label
        key="img_upload"
    )

# =============================================
# TRAITEMENT DE L'ENVOI
# =============================================
if prompt or uploaded_img:
    # Préparation du contenu utilisateur
    user_text = prompt or "Analyse cette image en détail"
    user_content = [{"type": "text", "text": user_text}]
    image_display = None

    if uploaded_img:
        image_bytes = uploaded_img.read()
        image_b64 = base64.b64encode(image_bytes).decode()
        image_display = image_bytes
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
        })

    # Affichage message utilisateur
    with st.chat_message("user"):
        st.markdown(user_text)
        if uploaded_img:
            st.image(uploaded_img, width=500)

    # Ajout à l'historique
    st.session_state.messages.append({
        "role": "user",
        "content": user_text,
        "image": image_display
    })

    # Réponse Groq
    with st.chat_message("assistant"):
        with st.spinner("Grok répond..."):
            answer = "Erreur inconnue"
            try:
                response = client.chat.completions.create(
                    model="llama-3.2-90b-vision-preview" if uploaded_img else "llama-3.2-90b-text-preview",
                    messages=[{"role": "user", "content": user_content}],
                    temperature=0.7,
                    max_tokens=1500
                )
                answer = response.choices[0].message.content
                st.markdown(answer)
            except Exception as e:
                answer = f"Erreur Groq : {str(e)}"
                st.error(answer)

    # Sauvegarde réponse (answer existe toujours)
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

# =============================================
# BOUTON EFFACER
# =============================================
if st.button("Effacer la conversation", type="secondary"):
    st.session_state.messages = []
    st.rerun()

st.caption("Groq API • Llama 3.2 90B Vision • Réponses < 1s • Analyse d'images")
