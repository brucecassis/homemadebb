# pages/CHATBOT.py
# Support complet : Images (Llama 4 Scout) + Documents (PDF, Word, Excel, TXT...)

import streamlit as st
from groq import Groq
import base64
import time
import PyPDF2
import docx
import pandas as pd
from io import BytesIO

# =============================================
# PAGE CONFIG + STYLE BLOOMBERG
# =============================================
st.set_page_config(page_title="Groq Chatbot", page_icon="🤖", layout="wide")

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
    GROQ CHATBOT • LLAMA 4 SCOUT VISION + MULTIMODAL • {time.strftime("%H:%M:%S")} UTC
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
# FONCTIONS D'EXTRACTION DE TEXTE
# =============================================
def extract_text_from_pdf(file_bytes):
    """Extrait le texte d'un PDF"""
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        return f"Erreur lecture PDF : {str(e)}"

def extract_text_from_docx(file_bytes):
    """Extrait le texte d'un fichier Word"""
    try:
        doc = docx.Document(BytesIO(file_bytes))
        text = "\n".join([para.text for para in doc.paragraphs])
        return text.strip()
    except Exception as e:
        return f"Erreur lecture Word : {str(e)}"

def extract_text_from_excel(file_bytes, filename):
    """Extrait le texte d'un fichier Excel"""
    try:
        if filename.endswith('.xlsx'):
            df = pd.read_excel(BytesIO(file_bytes), engine='openpyxl')
        else:
            df = pd.read_excel(BytesIO(file_bytes))
        return df.to_string()
    except Exception as e:
        return f"Erreur lecture Excel : {str(e)}"

def extract_text_from_txt(file_bytes):
    """Extrait le texte d'un fichier texte"""
    try:
        return file_bytes.decode('utf-8')
    except:
        try:
            return file_bytes.decode('latin-1')
        except Exception as e:
            return f"Erreur lecture TXT : {str(e)}"

# =============================================
# AFFICHAGE HISTORIQUE
# =============================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("image"):
            st.image(msg["image"], width=500)

# =============================================
# INPUT + UPLOAD FICHIER
# =============================================
col_text, col_file = st.columns([5, 1])

with col_text:
    prompt = st.chat_input("Pose ta question ou envoie un document/image")

with col_file:
    uploaded_file = st.file_uploader(
        "Fichier",
        type=["png", "jpg", "jpeg", "webp", "pdf", "docx", "doc", "xlsx", "xls", "txt"],
        label_visibility="collapsed",
        key="file_upload"
    )

# =============================================
# TRAITEMENT DE L'ENVOI
# =============================================
if prompt or uploaded_file:
    user_text = prompt or "Analyse ce fichier en détail"
    user_content = [{"type": "text", "text": user_text}]
    image_display = None
    use_vision = False

    if uploaded_file:
        file_bytes = uploaded_file.read()
        filename = uploaded_file.name.lower()

        # IMAGES → Vision model
        if filename.endswith(('.png', '.jpg', '.jpeg', '.webp')):
            use_vision = True
            image_b64 = base64.b64encode(file_bytes).decode()
            image_display = file_bytes
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
            })

        # PDF
        elif filename.endswith('.pdf'):
            extracted_text = extract_text_from_pdf(file_bytes)
            user_content[0]["text"] = f"{user_text}\n\n[Contenu PDF]\n{extracted_text[:8000]}"

        # WORD
        elif filename.endswith(('.docx', '.doc')):
            extracted_text = extract_text_from_docx(file_bytes)
            user_content[0]["text"] = f"{user_text}\n\n[Contenu Word]\n{extracted_text[:8000]}"

        # EXCEL
        elif filename.endswith(('.xlsx', '.xls')):
            extracted_text = extract_text_from_excel(file_bytes, filename)
            user_content[0]["text"] = f"{user_text}\n\n[Contenu Excel]\n{extracted_text[:8000]}"

        # TXT
        elif filename.endswith('.txt'):
            extracted_text = extract_text_from_txt(file_bytes)
            user_content[0]["text"] = f"{user_text}\n\n[Contenu TXT]\n{extracted_text[:8000]}"

    # Affichage message utilisateur
    with st.chat_message("user"):
        st.markdown(user_text)
        if uploaded_file:
            if use_vision:
                st.image(uploaded_file, width=500)
            else:
                st.info(f"📄 Document joint : {uploaded_file.name}")

    # Ajout à l'historique
    st.session_state.messages.append({
        "role": "user",
        "content": user_text,
        "image": image_display
    })

    # Réponse Groq
    with st.chat_message("assistant"):
        with st.spinner("Groq analyse..."):
            answer = "Erreur inconnue"
            try:
                # Choix du modèle selon le type de contenu
                if use_vision:
                    model_name = "meta-llama/llama-4-scout-17b-16e-instruct"
                else:
                    model_name = "llama-3.3-70b-versatile"

                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": user_content}],
                    temperature=0.7,
                    max_tokens=2000
                )
                answer = response.choices[0].message.content
                st.markdown(answer)
            except Exception as e:
                answer = f"❌ Erreur Groq : {str(e)}"
                st.error(answer)

    # Sauvegarde réponse
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

# =============================================
# BOUTON EFFACER
# =============================================
if st.button("🗑️ Effacer la conversation", type="secondary"):
    st.session_state.messages = []
    st.rerun()

st.caption("Groq API • Llama 4 Scout Vision + Llama 3.3 70B • Analyse images, PDF, Word, Excel, TXT")
