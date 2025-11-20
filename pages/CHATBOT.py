# pages/CHATBOT.py
# Style Bloomberg Terminal - Support complet multimodal

import streamlit as st
from groq import Groq
import base64
import time
import PyPDF2
import docx
import pandas as pd
from io import BytesIO

# =============================================
# PAGE CONFIG
# =============================================
st.set_page_config(page_title="Bloomberg Terminal - AI Assistant", page_icon="🤖", layout="wide")

# =============================================
# STYLE BLOOMBERG TERMINAL
# =============================================
st.markdown("""
<style>
    /* Reset et fond */
    .main {
        background: #000 !important;
        color: #FFAA00 !important;
        padding: 0 !important;
        font-family: 'Courier New', monospace !important;
        font-size: 12px !important;
    }
    
    /* Suppression padding par défaut */
    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
    
    /* Header style Bloomberg */
    [data-testid="stHeader"] {
        background: #000 !important;
        border-bottom: 2px solid #FFAA00 !important;
    }
    
    /* Boutons style terminal */
    .stButton>button {
        background: #333 !important;
        color: #FFAA00 !important;
        border: 2px solid #FFAA00 !important;
        padding: 8px 20px !important;
        font-size: 12px !important;
        font-weight: bold !important;
        cursor: pointer !important;
        font-family: 'Courier New', monospace !important;
        transition: all 0.3s !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        border-radius: 0 !important;
    }
    
    .stButton>button:hover {
        background: #FFAA00 !important;
        color: #000 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(255, 170, 0, 0.3) !important;
    }
    
    /* Messages de chat style terminal */
    .stChatMessage {
        background: #111 !important;
        padding: 15px !important;
        border: 1px solid #333 !important;
        border-left: 4px solid #FFAA00 !important;
        margin: 10px 0 !important;
        color: #FFF !important;
        border-radius: 0 !important;
    }
    
    .stChatMessage[data-testid="user"] {
        border-left-color: #00FF00 !important;
    }
    
    /* Input de chat */
    .stChatInputContainer {
        background: #111 !important;
        border: 2px solid #FFAA00 !important;
        border-radius: 0 !important;
        padding: 10px !important;
    }
    
    .stChatInput>div>div>input {
        background: #000 !important;
        color: #FFAA00 !important;
        border: 1px solid #333 !important;
        font-family: 'Courier New', monospace !important;
        font-size: 14px !important;
        border-radius: 0 !important;
    }
    
    .stChatInput>div>div>input::placeholder {
        color: #666 !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #111 !important;
        border: 2px solid #FFAA00 !important;
        padding: 10px !important;
        border-radius: 0 !important;
    }
    
    [data-testid="stFileUploader"] label {
        color: #FFAA00 !important;
        font-family: 'Courier New', monospace !important;
        font-size: 12px !important;
        font-weight: bold !important;
    }
    
    /* Spinner */
    .stSpinner>div {
        border-color: #FFAA00 transparent transparent transparent !important;
    }
    
    /* Info boxes */
    .stAlert {
        background: #111 !important;
        border: 1px solid #FFAA00 !important;
        color: #FFAA00 !important;
        border-radius: 0 !important;
    }
    
    /* Titres */
    h1, h2, h3 {
        color: #FFAA00 !important;
        font-family: 'Courier New', monospace !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
    }
    
    /* Caption/footer */
    .caption-text {
        color: #666 !important;
        font-size: 11px !important;
        font-family: 'Courier New', monospace !important;
        text-align: center !important;
        padding: 10px !important;
        border-top: 1px solid #333 !important;
    }
    
    /* Colonnes */
    [data-testid="column"] {
        background: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# HEADER BLOOMBERG
# =============================================
current_time = time.strftime("%H:%M:%S", time.gmtime())
st.markdown(f"""
<div style="background:#000;color:#FFAA00;padding:8px 20px;font-size:14px;font-weight:bold;border-bottom:2px solid #FFAA00;display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;">
    <div style="display:flex;align-items:center;gap:15px;">
        <div>BLOOMBERG ENS® | AI ASSISTANT</div>
        <a href="bloomberg_v5.html" style="background:#333;color:#FFAA00;border:1px solid #FFAA00;padding:4px 12px;font-size:11px;text-decoration:none;transition:all 0.2s;">RETOUR TERMINAL</a>
    </div>
    <div>{current_time} UTC</div>
</div>
""", unsafe_allow_html=True)

# =============================================
# CLIENT GROQ
# =============================================
if "groq_client" not in st.session_state:
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error("⚠️ Clé GROQ_API_KEY manquante dans .streamlit/secrets.toml")
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
# ZONE CENTRALE
# =============================================
st.markdown("""
<div style="text-align:center;padding:20px 0;margin-bottom:20px;">
    <div style="color:#FFAA00;font-size:18px;font-weight:bold;margin-bottom:10px;text-transform:uppercase;letter-spacing:2px;">
        AI ASSISTANT MULTIMODAL
    </div>
    <div style="color:#FFF;font-size:12px;line-height:1.6;">
        Analyse d'images, documents (PDF, Word, Excel) et conversations en langage naturel
    </div>
</div>
""", unsafe_allow_html=True)

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
col_file, col_clear = st.columns([8, 2])

with col_file:
    uploaded_file = st.file_uploader(
        "📎 JOINDRE UN FICHIER",
        type=["png", "jpg", "jpeg", "webp", "pdf", "docx", "doc", "xlsx", "xls", "txt"],
        key="file_upload",
        help="Images, PDF, Word, Excel, TXT supportés"
    )

with col_clear:
    if st.button("🗑️ EFFACER", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Input de chat
prompt = st.chat_input("BBG> Tapez votre question ou commande...")

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
        with st.spinner("⚡ GROQ TRAITE LA REQUÊTE..."):
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
                answer = f"❌ ERREUR GROQ : {str(e)}"
                st.error(answer)

    # Sauvegarde réponse
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

# =============================================
# FOOTER TERMINAL
# =============================================
st.markdown("""
<div style="margin-top:40px;padding-top:20px;border-top:1px solid #333;">
    <div style="color:#666;font-size:11px;text-align:center;line-height:1.6;">
        <div>BLOOMBERG ENS® v4.0 - Système IA opérationnel</div>
        <div>Groq API • Llama 4 Scout Vision + Llama 3.3 70B</div>
        <div>Analyse images, PDF, Word, Excel, TXT • Réponses < 1s</div>
        <div>Données de marché disponibles • Connexion établie - Paris, France</div>
    </div>
</div>
""", unsafe_allow_html=True)
