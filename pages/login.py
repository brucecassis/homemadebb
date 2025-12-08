"""
Page de connexion pour Bloomberg Terminal
Interface d'authentification avec création de compte
"""

import streamlit as st
from auth_utils import AuthManager, init_session_state

def show_login_page():
    """Affiche la page de connexion/inscription"""
    
    # Initialise les variables de session
    init_session_state()
    
    # Style Bloomberg Terminal
    st.markdown("""
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        .stApp {
            background-color: #000000 !important;
        }
        
        body {
            font-family: 'Courier New', monospace;
            background: #000;
            color: #FFAA00;
            font-size: 12px;
        }
        
        h1, h2, h3 {
            color: #FFAA00 !important;
            font-family: 'Courier New', monospace !important;
            text-transform: uppercase !important;
            letter-spacing: 2px !important;
        }
        
        .stTextInput > div > div > input {
            background-color: #111 !important;
            color: #FFAA00 !important;
            border: 2px solid #333 !important;
            font-family: 'Courier New', monospace !important;
        }
        
        .stButton > button {
            background-color: #333 !important;
            color: #FFAA00 !important;
            font-weight: bold !important;
            border: 2px solid #FFAA00 !important;
            padding: 10px 30px !important;
            text-transform: uppercase !important;
            letter-spacing: 2px !important;
            border-radius: 0px !important;
            font-size: 12px !important;
            font-family: 'Courier New', monospace !important;
            transition: all 0.3s !important;
            width: 100% !important;
        }
        
        .stButton > button:hover {
            background-color: #FFAA00 !important;
            color: #000 !important;
            transform: translateY(-2px) !important;
        }
        
        .login-container {
            max-width: 500px;
            margin: 50px auto;
            padding: 30px;
            background: #111;
            border: 2px solid #FFAA00;
            border-left: 8px solid #FFAA00;
        }
        
        .terminal-header {
            background: #FFAA00;
            color: #000;
            padding: 15px;
            font-weight: bold;
            font-size: 16px;
            text-align: center;
            margin-bottom: 30px;
            letter-spacing: 3px;
        }
        
        .tab-container {
            display: flex;
            gap: 10px;
            margin-bottom: 30px;
        }
        
        .tab {
            flex: 1;
            padding: 10px;
            background: #222;
            border: 1px solid #444;
            color: #888;
            text-align: center;
            cursor: pointer;
            text-transform: uppercase;
            font-weight: bold;
            letter-spacing: 1px;
        }
        
        .tab.active {
            background: #333;
            border: 2px solid #FFAA00;
            color: #FFAA00;
        }
        
        label {
            color: #FFAA00 !important;
            font-family: 'Courier New', monospace !important;
            text-transform: uppercase !important;
            font-size: 10px !important;
            letter-spacing: 1px !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header Bloomberg
    st.markdown("""
    <div class="terminal-header">
        ⬛ BLOOMBERG ENS® TERMINAL - AUTHENTICATION SYSTEM
    </div>
    """, unsafe_allow_html=True)
    
    # Container principal
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    
    # Tabs pour Login / Register
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔐 CONNEXION", key="tab_login", 
                    use_container_width=True):
            st.session_state.auth_page = 'login'
    
    with col2:
        if st.button("📝 INSCRIPTION", key="tab_register", 
                    use_container_width=True):
            st.session_state.auth_page = 'register'
    
    st.markdown('<div style="margin: 20px 0;"></div>', unsafe_allow_html=True)
    
    # Initialise le gestionnaire d'authentification
    auth = AuthManager()
    
    # Affiche le formulaire approprié
    if st.session_state.auth_page == 'login':
        show_login_form(auth)
    else:
        show_register_form(auth)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown('<hr style="border-color: #333; margin: 50px 0 20px 0;">', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 9px; font-family: "Courier New", monospace;'>
        © 2025 BLOOMBERG ENS® TERMINAL | SECURE AUTHENTICATION SYSTEM<br>
        ALL DATA ENCRYPTED • SSL SECURED • GDPR COMPLIANT
    </div>
    """, unsafe_allow_html=True)


def show_login_form(auth: AuthManager):
    """Affiche le formulaire de connexion"""
    
    st.markdown("### 🔐 CONNEXION")
    st.markdown('<div style="margin: 20px 0;"></div>', unsafe_allow_html=True)
    
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input(
            "Nom d'utilisateur",
            placeholder="Entrez votre nom d'utilisateur",
            key="login_username"
        )
        
        password = st.text_input(
            "Mot de passe",
            type="password",
            placeholder="Entrez votre mot de passe",
            key="login_password"
        )
        
        st.markdown('<div style="margin: 20px 0;"></div>', unsafe_allow_html=True)
        
        submit = st.form_submit_button("🚀 SE CONNECTER", use_container_width=True)
        
        if submit:
            if not username or not password:
                st.error("❌ Veuillez remplir tous les champs")
            else:
                with st.spinner("⏳ Authentification en cours..."):
                    success, user_data, message = auth.verify_login(username, password)
                    
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.user_data = user_data
                        st.success(message)
                        st.balloons()
                        
                        # Petit délai pour afficher le message
                        import time
                        time.sleep(1)
                        
                        # Redirection vers la page d'accueil
                        st.rerun()
                    else:
                        st.error(message)
    
    # Lien vers inscription
    st.markdown('<div style="margin: 30px 0 10px 0;"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 11px;">
        Pas encore de compte ?<br>
        Cliquez sur "INSCRIPTION" ci-dessus
    </div>
    """, unsafe_allow_html=True)


def show_register_form(auth: AuthManager):
    """Affiche le formulaire d'inscription"""
    
    st.markdown("### 📝 CRÉER UN COMPTE")
    st.markdown('<div style="margin: 20px 0;"></div>', unsafe_allow_html=True)
    
    with st.form("register_form", clear_on_submit=False):
        username = st.text_input(
            "Nom d'utilisateur",
            placeholder="Minimum 3 caractères",
            help="Votre identifiant unique",
            key="reg_username"
        )
        
        email = st.text_input(
            "Email",
            placeholder="votre@email.com",
            help="Votre adresse email",
            key="reg_email"
        )
        
        password = st.text_input(
            "Mot de passe",
            type="password",
            placeholder="Minimum 8 caractères",
            help="Choisissez un mot de passe sécurisé",
            key="reg_password"
        )
        
        password_confirm = st.text_input(
            "Confirmer le mot de passe",
            type="password",
            placeholder="Retapez votre mot de passe",
            key="reg_password_confirm"
        )
        
        st.markdown('<div style="margin: 10px 0;"></div>', unsafe_allow_html=True)
        
        accept_terms = st.checkbox(
            "J'accepte les conditions d'utilisation",
            key="accept_terms"
        )
        
        st.markdown('<div style="margin: 20px 0;"></div>', unsafe_allow_html=True)
        
        submit = st.form_submit_button("✅ CRÉER MON COMPTE", use_container_width=True)
        
        if submit:
            # Validations
            if not username or not email or not password or not password_confirm:
                st.error("❌ Veuillez remplir tous les champs")
            elif password != password_confirm:
                st.error("❌ Les mots de passe ne correspondent pas")
            elif not accept_terms:
                st.error("❌ Vous devez accepter les conditions d'utilisation")
            else:
                with st.spinner("⏳ Création du compte..."):
                    success, message = auth.create_user(username, email, password)
                    
                    if success:
                        st.success(message)
                        st.balloons()
                        st.info("🔐 Vous pouvez maintenant vous connecter avec vos identifiants")
                        
                        # Attendre un peu puis basculer vers login
                        import time
                        time.sleep(2)
                        st.session_state.auth_page = 'login'
                        st.rerun()
                    else:
                        st.error(message)
    
    # Lien vers connexion
    st.markdown('<div style="margin: 30px 0 10px 0;"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 11px;">
        Vous avez déjà un compte ?<br>
        Cliquez sur "CONNEXION" ci-dessus
    </div>
    """, unsafe_allow_html=True)
    
    # Afficher les stats d'utilisateurs
    stats = auth.get_user_stats()
    st.markdown('<div style="margin: 40px 0 0 0;"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="text-align: center; color: #666; font-size: 10px; 
                padding: 15px; background: #0a0a0a; border: 1px solid #222;">
        📊 STATISTIQUES PLATEFORME<br><br>
        👥 {stats['total']} utilisateurs inscrits<br>
        🟢 {stats['active']} actifs (7 derniers jours)<br>
        ✨ {stats['new']} nouvelles inscriptions (24h)
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    show_login_page()
