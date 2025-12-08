"""
Page de gestion du profil utilisateur
Permet de modifier le mot de passe et voir les statistiques
"""

import streamlit as st
from auth_utils import AuthManager, init_session_state, logout, require_auth
from datetime import datetime

# Initialisation
init_session_state()

# Vérifier l'authentification
if not st.session_state.get('authenticated', False):
    st.error("🔒 Accès refusé. Veuillez vous connecter.")
    st.stop()

# =============================================
# PAGE CONFIG
# =============================================
st.set_page_config(
    page_title="Bloomberg Terminal - Profile",
    page_icon="👤",
    layout="wide"
)

# =============================================
# STYLE
# =============================================
st.markdown("""
<style>
    .stApp {
        background-color: #000000 !important;
    }
    
    body {
        font-family: 'Courier New', monospace;
        background: #000;
        color: #FFAA00;
    }
    
    h1, h2, h3 {
        color: #FFAA00 !important;
        font-family: 'Courier New', monospace !important;
        text-transform: uppercase !important;
    }
    
    .profile-box {
        background: #111;
        border: 2px solid #333;
        border-left: 6px solid #FFAA00;
        padding: 20px;
        margin: 20px 0;
    }
    
    .stat-box {
        background: #0a0a0a;
        border: 1px solid #444;
        padding: 15px;
        text-align: center;
        margin: 10px 0;
    }
    
    .stButton > button {
        background-color: #333 !important;
        color: #FFAA00 !important;
        border: 2px solid #FFAA00 !important;
        font-family: 'Courier New', monospace !important;
        text-transform: uppercase !important;
    }
    
    .stButton > button:hover {
        background-color: #FFAA00 !important;
        color: #000 !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# HEADER
# =============================================
st.markdown(f"""
<div style="background:#FFAA00;padding:15px;color:#000;font-weight:bold;font-size:16px;text-align:center;letter-spacing:3px;margin-bottom:30px;">
    👤 BLOOMBERG ENS® TERMINAL - USER PROFILE
</div>
""", unsafe_allow_html=True)

# Récupérer les données utilisateur
user_data = st.session_state.user_data
auth = AuthManager()

# =============================================
# INFORMATIONS DU PROFIL
# =============================================
st.markdown("## 📋 INFORMATIONS DU COMPTE")

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="profile-box">', unsafe_allow_html=True)
    st.markdown(f"""
    **Nom d'utilisateur:** {user_data['username']}  
    **Email:** {user_data['email']}  
    **Rôle:** {user_data['role'].upper()}  
    **ID Utilisateur:** #{user_data['id']}  
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="profile-box">', unsafe_allow_html=True)
    st.markdown(f"""
    **Statut:** 🟢 ACTIF  
    **Session:** ACTIVE  
    **Date actuelle:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  
    **Timezone:** UTC  
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================
# CHANGEMENT DE MOT DE PASSE
# =============================================
st.markdown('<hr style="border-color: #333; margin: 30px 0;">', unsafe_allow_html=True)
st.markdown("## 🔐 CHANGER LE MOT DE PASSE")

with st.form("change_password_form"):
    old_password = st.text_input(
        "Ancien mot de passe",
        type="password",
        placeholder="Entrez votre ancien mot de passe"
    )
    
    new_password = st.text_input(
        "Nouveau mot de passe",
        type="password",
        placeholder="Minimum 8 caractères"
    )
    
    new_password_confirm = st.text_input(
        "Confirmer le nouveau mot de passe",
        type="password",
        placeholder="Retapez le nouveau mot de passe"
    )
    
    submit = st.form_submit_button("✅ MODIFIER LE MOT DE PASSE", use_container_width=True)
    
    if submit:
        if not old_password or not new_password or not new_password_confirm:
            st.error("❌ Veuillez remplir tous les champs")
        elif new_password != new_password_confirm:
            st.error("❌ Les nouveaux mots de passe ne correspondent pas")
        else:
            success, message = auth.change_password(
                user_data['id'],
                old_password,
                new_password
            )
            
            if success:
                st.success(message)
                st.balloons()
            else:
                st.error(message)

# =============================================
# STATISTIQUES DU COMPTE
# =============================================
st.markdown('<hr style="border-color: #333; margin: 30px 0;">', unsafe_allow_html=True)
st.markdown("## 📊 STATISTIQUES DE LA PLATEFORME")

stats = auth.get_user_stats()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="stat-box">
        <h2 style="color: #00FF00; font-size: 32px; margin: 10px 0;">{stats['total']}</h2>
        <p style="color: #888;">UTILISATEURS INSCRITS</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="stat-box">
        <h2 style="color: #FFAA00; font-size: 32px; margin: 10px 0;">{stats['active']}</h2>
        <p style="color: #888;">UTILISATEURS ACTIFS (7J)</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="stat-box">
        <h2 style="color: #00FFFF; font-size: 32px; margin: 10px 0;">{stats['new']}</h2>
        <p style="color: #888;">NOUVELLES INSCRIPTIONS (24H)</p>
    </div>
    """, unsafe_allow_html=True)

# =============================================
# ACTIONS DU COMPTE
# =============================================
st.markdown('<hr style="border-color: #333; margin: 30px 0;">', unsafe_allow_html=True)
st.markdown("## ⚙️ ACTIONS DU COMPTE")

col1, col2 = st.columns(2)

with col1:
    if st.button("🚪 SE DÉCONNECTER", key="logout", use_container_width=True):
        st.info(f"👋 Au revoir {user_data['username']} !")
        import time
        time.sleep(1)
        logout()

with col2:
    if st.button("🏠 RETOUR À L'ACCUEIL", key="home", use_container_width=True):
        st.switch_page("accueil_with_auth.py")

# =============================================
# FOOTER
# =============================================
st.markdown('<hr style="margin-top: 50px;">', unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace; padding: 10px;'>
    © 2025 BLOOMBERG ENS® | USER: {user_data['username'].upper()}<br>
    PROFILE MANAGEMENT • SECURE SESSION • ENCRYPTED DATA
</div>
""", unsafe_allow_html=True)
