"""
Module d'authentification pour Bloomberg Terminal avec Supabase
Gestion des utilisateurs avec PostgreSQL via Supabase
"""

import hashlib
import secrets
from datetime import datetime
import streamlit as st
import os
from supabase import create_client, Client

class AuthManager:
    """Gestionnaire d'authentification des utilisateurs avec Supabase"""
    
    def __init__(self):
    """Initialise la connexion Supabase"""
    from config import SUPABASE_URL, SUPABASE_KEY
    
    supabase_url = SUPABASE_URL
    supabase_key = SUPABASE_KEY
```

### 3️⃣ Dans Render, ajoutez les variables

Dashboard Render → Votre app → **Environment** → **Add Environment Variable** :
```
SUPABASE_URL = https://xxxxx.supabase.co
SUPABASE_KEY = votre_clé_ici
        
        self.supabase: Client = create_client(supabase_url, supabase_key)
    
    def hash_password(self, password: str, salt: str = None) -> tuple:
        """
        Hash un mot de passe avec un salt
        Returns: (password_hash, salt)
        """
        if salt is None:
            salt = secrets.token_hex(32)
        
        pwd_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        
        return pwd_hash.hex(), salt
    
    def create_user(self, username: str, email: str, password: str) -> tuple:
        """
        Crée un nouveau compte utilisateur
        Returns: (success: bool, message: str)
        """
        # Validation
        if len(username) < 3:
            return False, "❌ Le nom d'utilisateur doit contenir au moins 3 caractères"
        
        if len(password) < 8:
            return False, "❌ Le mot de passe doit contenir au moins 8 caractères"
        
        if '@' not in email:
            return False, "❌ Email invalide"
        
        try:
            # Vérifie si l'utilisateur existe déjà
            existing = self.supabase.table('users').select('id').or_(
                f'username.eq.{username},email.eq.{email}'
            ).execute()
            
            if existing.data:
                return False, "❌ Nom d'utilisateur ou email déjà utilisé"
            
            # Hash le mot de passe
            pwd_hash, salt = self.hash_password(password)
            
            # Insert le nouvel utilisateur
            data = {
                'username': username,
                'email': email,
                'password_hash': pwd_hash,
                'salt': salt,
                'role': 'user'
            }
            
            self.supabase.table('users').insert(data).execute()
            
            return True, f"✅ Compte créé avec succès ! Bienvenue {username}"
            
        except Exception as e:
            return False, f"❌ Erreur lors de la création du compte: {str(e)}"
    
    def verify_login(self, username: str, password: str) -> tuple:
        """
        Vérifie les credentials de connexion
        Returns: (success: bool, user_data: dict or None, message: str)
        """
        try:
            # Récupère l'utilisateur
            response = self.supabase.table('users').select('*').eq(
                'username', username
            ).execute()
            
            if not response.data:
                self.log_login_attempt(None, username, False)
                return False, None, "❌ Nom d'utilisateur ou mot de passe incorrect"
            
            user = response.data[0]
            
            if not user.get('is_active', True):
                return False, None, "❌ Compte désactivé. Contactez l'administrateur"
            
            # Vérifie le mot de passe
            pwd_hash, _ = self.hash_password(password, user['salt'])
            
            if pwd_hash != user['password_hash']:
                self.log_login_attempt(user['id'], username, False)
                return False, None, "❌ Nom d'utilisateur ou mot de passe incorrect"
            
            # Mise à jour de last_login
            self.supabase.table('users').update({
                'last_login': datetime.now().isoformat()
            }).eq('id', user['id']).execute()
            
            # Log de connexion réussie
            self.log_login_attempt(user['id'], username, True)
            
            user_data = {
                'id': user['id'],
                'username': user['username'],
                'email': user['email'],
                'role': user['role']
            }
            
            return True, user_data, f"✅ Bienvenue {username} !"
            
        except Exception as e:
            return False, None, f"❌ Erreur de connexion: {str(e)}"
    
    def log_login_attempt(self, user_id: int, username: str, success: bool):
        """Enregistre une tentative de connexion"""
        try:
            data = {
                'user_id': user_id,
                'username': username,
                'success': success
            }
            self.supabase.table('login_logs').insert(data).execute()
        except:
            pass
    
    def get_user_stats(self) -> dict:
        """Récupère les statistiques des utilisateurs"""
        try:
            # Nombre total d'utilisateurs
            total_response = self.supabase.table('users').select('id', count='exact').execute()
            total_users = total_response.count
            
            # Utilisateurs actifs (connectés dans les 7 derniers jours)
            from datetime import timedelta
            seven_days_ago = (datetime.now() - timedelta(days=7)).isoformat()
            
            active_response = self.supabase.table('users').select(
                'id', count='exact'
            ).gte('last_login', seven_days_ago).execute()
            active_users = active_response.count
            
            # Nouvelles inscriptions (dernières 24h)
            one_day_ago = (datetime.now() - timedelta(days=1)).isoformat()
            new_response = self.supabase.table('users').select(
                'id', count='exact'
            ).gte('created_at', one_day_ago).execute()
            new_users = new_response.count
            
            return {
                'total': total_users or 0,
                'active': active_users or 0,
                'new': new_users or 0
            }
        except:
            return {'total': 0, 'active': 0, 'new': 0}
    
    def change_password(self, user_id: int, old_password: str, new_password: str) -> tuple:
        """
        Change le mot de passe d'un utilisateur
        Returns: (success: bool, message: str)
        """
        try:
            # Récupère l'utilisateur
            response = self.supabase.table('users').select(
                'password_hash, salt'
            ).eq('id', user_id).execute()
            
            if not response.data:
                return False, "❌ Utilisateur non trouvé"
            
            user = response.data[0]
            
            # Vérifie l'ancien mot de passe
            old_hash, _ = self.hash_password(old_password, user['salt'])
            
            if old_hash != user['password_hash']:
                return False, "❌ Ancien mot de passe incorrect"
            
            # Valide le nouveau mot de passe
            if len(new_password) < 8:
                return False, "❌ Le nouveau mot de passe doit contenir au moins 8 caractères"
            
            # Hash le nouveau mot de passe
            new_hash, new_salt = self.hash_password(new_password)
            
            # Met à jour
            self.supabase.table('users').update({
                'password_hash': new_hash,
                'salt': new_salt
            }).eq('id', user_id).execute()
            
            return True, "✅ Mot de passe modifié avec succès"
            
        except Exception as e:
            return False, f"❌ Erreur: {str(e)}"


def init_session_state():
    """Initialise les variables de session Streamlit"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_data' not in st.session_state:
        st.session_state.user_data = None
    if 'auth_page' not in st.session_state:
        st.session_state.auth_page = 'login'


def logout():
    """Déconnexion de l'utilisateur"""
    st.session_state.authenticated = False
    st.session_state.user_data = None
    st.rerun()


def require_auth(func):
    """
    Décorateur pour protéger les pages nécessitant une authentification
    """
    def wrapper(*args, **kwargs):
        if not st.session_state.get('authenticated', False):
            st.error("🔒 Accès refusé. Veuillez vous connecter.")
            st.stop()
        return func(*args, **kwargs)
    return wrapper
