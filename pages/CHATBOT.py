# pages/CHATBOT.py
# Style Bloomberg Terminal - Support complet multimodal avec contexte de conversation

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
# BARRE DE COMMANDE BLOOMBERG
# À ajouter après le header, avant les données de marché
# =============================================

# Style pour la barre de commande
st.markdown("""
<style>
    .command-container {
        background: #111;
        border: 2px solid #FFAA00;
        padding: 10px 15px;
        margin: 10px 0 20px 0;
    }
    .command-prompt {
        color: #FFAA00;
        font-weight: bold;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Dictionnaire des commandes et leurs pages
COMMANDS = {
    "EDGAR": "pages/EDGAR.py",
    "NEWS": "pages/NEWS.py",
    "PRICE": "pages/PRICING.py",
    "CHAT": "pages/CHATBOT.py",
    "BT": "pages/BACKTESTING.py",
    "ANA": "pages/COMPANY_ANALYSIS.py",
    "CRYPTO":"pages/CRYPTO_SCRAPER.py",
    "ECO":"pages/ECONOMICS.py", 
    "EU":"pages/EUROPE.py",
    "SIMU":"pages/PORTFOLIO_SIMU.py",
    "PY":"pages/PYTHON_EDITOR.py",
    "SQL":"pages/SQL_EDITOR.py",
    "BONDS":"pages/BONDS.py",
    "HOME":"pages/HOME.py",
}

# Affichage de la barre de commande
st.markdown('<div class="command-container">', unsafe_allow_html=True)

col_prompt, col_input = st.columns([1, 11])

with col_prompt:
    st.markdown('<span class="command-prompt">BBG&gt;</span>', unsafe_allow_html=True)

with col_input:
    command_input = st.text_input(
        "Command",
        placeholder="Tapez une commande: EDGAR, NEWS, CHATBOT, PRICING, HELP...",
        label_visibility="collapsed",
        key="bloomberg_command"
    )

st.markdown('</div>', unsafe_allow_html=True)

# Traitement de la commande
if command_input:
    cmd = command_input.upper().strip()
    
    if cmd == "HELP" or cmd == "H":
        st.info("""
        **📋 COMMANDES DISPONIBLES:**
        - `EDGAR` → SEC Filings & Documents
        - `NEWS` → Market News Feed
        - `CHAT` → AI Assistant
        - `PRICE` → Options Pricing
        - `HELP` → Afficher cette aide
        - `BT` → Backesting de strategies
        - `ANA` → Analyse financière de sociétés côtées
        - `CRYPTO` → Scrapping et backtest de strategies liées aux cryptos
        - `ECO` → Données économiques
        - `EU` → Données Européennes
        - `SIMU` → Simulation de portefeuille
        - `PY` → Editeur de code python 
        - `SQL` → Editeur de code SQL
        - `BONDS` → Screener d'obligation
        - `HOME` → Menu
        """)
    elif cmd in COMMANDS:
        st.switch_page(COMMANDS[cmd])
    else:
        st.warning(f"⚠️ Commande '{cmd}' non reconnue. Tapez HELP pour voir les commandes disponibles.")


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

# Historique des messages (pour l'affichage)
if "messages" not in st.session_state:
    st.session_state.messages = []

# NOUVEAU: Historique pour l'API (avec contexte complet)
if "api_messages" not in st.session_state:
    st.session_state.api_messages = []

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

def extract_text_from_csv(file_bytes):
    """Extrait le texte d'un fichier CSV"""
    try:
        # Essayer différents encodages
        for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
            try:
                df = pd.read_csv(BytesIO(file_bytes), encoding=encoding)
                return df.to_string()
            except:
                continue
        return "Erreur: Impossible de lire le CSV avec les encodages standards"
    except Exception as e:
        return f"Erreur lecture CSV : {str(e)}"

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
        Analyse d'images, documents (PDF, Word, Excel, CSV) et conversations contextuelles
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
        type=["png", "jpg", "jpeg", "webp", "pdf", "docx", "doc", "xlsx", "xls", "csv", "txt"],
        key="file_upload",
        help="Images, PDF, Word, Excel, CSV, TXT supportés"
    )

with col_clear:
    if st.button("🗑️ EFFACER", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.api_messages = []  # NOUVEAU: Effacer aussi l'historique API
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

        # CSV (NOUVEAU)
        elif filename.endswith('.csv'):
            extracted_text = extract_text_from_csv(file_bytes)
            user_content[0]["text"] = f"{user_text}\n\n[Contenu CSV]\n{extracted_text[:8000]}"

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

    # Ajout à l'historique d'affichage
    st.session_state.messages.append({
        "role": "user",
        "content": user_text,
        "image": image_display
    })

    # NOUVEAU: Ajout à l'historique API
    st.session_state.api_messages.append({
        "role": "user",
        "content": user_content
    })

    # Réponse Groq
    with st.chat_message("assistant"):
        with st.spinner("⚡ GROQ TRAITE LA REQUÊTE..."):
            answer = "Erreur inconnue"
            try:
                # Choix du modèle selon le type de contenu
                if use_vision:
                    model_name = "meta-llama/llama-4-scout-17b-16e-instruct"
                    # Pour vision, on envoie seulement le message actuel (limitation du modèle)
                    messages_to_send = [{"role": "user", "content": user_content}]
                else:
                    model_name = "llama-3.3-70b-versatile"
                    # NOUVEAU: Pour les modèles texte, on envoie tout l'historique
                    messages_to_send = st.session_state.api_messages

                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages_to_send,
                    temperature=0.7,
                    max_tokens=2000
                )
                answer = response.choices[0].message.content
                st.markdown(answer)
            except Exception as e:
                answer = f"❌ ERREUR GROQ : {str(e)}"
                st.error(answer)

    # Sauvegarde réponse dans l'historique d'affichage
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    # NOUVEAU: Sauvegarde réponse dans l'historique API
    st.session_state.api_messages.append({
        "role": "assistant",
        "content": answer
    })

# =============================================
# FOOTER TERMINAL
# =============================================
st.markdown("""
<div style="margin-top:40px;padding-top:20px;border-top:1px solid #333;">
    <div style="color:#666;font-size:11px;text-align:center;line-height:1.6;">
        <div>BLOOMBERG ENS® v4.1 - Système IA opérationnel avec contexte conversationnel</div>
        <div>Groq API • Llama 4 Scout Vision + Llama 3.3 70B</div>
        <div>Analyse images, PDF, Word, Excel, CSV, TXT • Réponses < 1s</div>
        <div>Données de marché disponibles • Connexion établie - Paris, France</div>
    </div>
</div>
""", unsafe_allow_html=True)
