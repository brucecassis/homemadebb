"""
Module d'authentification pour Bloomberg Terminal
Gestion des utilisateurs avec SQLite et hashage sécurisé
"""

import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta
import streamlit as st
from pathlib import Path

class AuthManager:
    """Gestionnaire d'authentification des utilisateurs"""
    
    def __init__(self, db_path="users.db"):
        """Initialise la base de données utilisateurs"""
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Crée la base de données si elle n'existe pas"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table des utilisateurs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                role TEXT DEFAULT 'user'
            )
        ''')
        
        # Table des sessions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_token TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                ip_address TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Table des logs de connexion
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS login_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                username TEXT,
                success BOOLEAN,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ip_address TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def hash_password(self, password: str, salt: str = None) -> tuple:
        """
        Hash un mot de passe avec un salt
        Returns: (password_hash, salt)
        """
        if salt is None:
            salt = secrets.token_hex(32)
        
        # Utilise SHA-256 pour le hashage
        pwd_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # 100k iterations
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
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Vérifie si l'utilisateur existe déjà
            cursor.execute("SELECT id FROM users WHERE username = ? OR email = ?", 
                          (username, email))
            if cursor.fetchone():
                conn.close()
                return False, "❌ Nom d'utilisateur ou email déjà utilisé"
            
            # Hash le mot de passe
            pwd_hash, salt = self.hash_password(password)
            
            # Insert le nouvel utilisateur
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, salt)
                VALUES (?, ?, ?, ?)
            ''', (username, email, pwd_hash, salt))
            
            conn.commit()
            conn.close()
            
            return True, f"✅ Compte créé avec succès ! Bienvenue {username}"
            
        except Exception as e:
            return False, f"❌ Erreur lors de la création du compte: {str(e)}"
    
    def verify_login(self, username: str, password: str) -> tuple:
        """
        Vérifie les credentials de connexion
        Returns: (success: bool, user_data: dict or None, message: str)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Récupère l'utilisateur
            cursor.execute('''
                SELECT id, username, email, password_hash, salt, is_active, role
                FROM users 
                WHERE username = ?
            ''', (username,))
            
            user = cursor.fetchone()
            
            if not user:
                self.log_login_attempt(None, username, False)
                conn.close()
                return False, None, "❌ Nom d'utilisateur ou mot de passe incorrect"
            
            user_id, username, email, stored_hash, salt, is_active, role = user
            
            if not is_active:
                conn.close()
                return False, None, "❌ Compte désactivé. Contactez l'administrateur"
            
            # Vérifie le mot de passe
            pwd_hash, _ = self.hash_password(password, salt)
            
            if pwd_hash != stored_hash:
                self.log_login_attempt(user_id, username, False)
                conn.close()
                return False, None, "❌ Nom d'utilisateur ou mot de passe incorrect"
            
            # Mise à jour de last_login
            cursor.execute('''
                UPDATE users 
                SET last_login = CURRENT_TIMESTAMP 
                WHERE id = ?
            ''', (user_id,))
            
            conn.commit()
            
            # Log de connexion réussie
            self.log_login_attempt(user_id, username, True)
            
            conn.close()
            
            user_data = {
                'id': user_id,
                'username': username,
                'email': email,
                'role': role
            }
            
            return True, user_data, f"✅ Bienvenue {username} !"
            
        except Exception as e:
            return False, None, f"❌ Erreur de connexion: {str(e)}"
    
    def log_login_attempt(self, user_id: int, username: str, success: bool):
        """Enregistre une tentative de connexion"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO login_logs (user_id, username, success)
                VALUES (?, ?, ?)
            ''', (user_id, username, success))
            
            conn.commit()
            conn.close()
        except:
            pass  # Silently fail si le log échoue
    
    def create_session(self, user_id: int, duration_hours: int = 24) -> str:
        """
        Crée une session utilisateur
        Returns: session_token
        """
        session_token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(hours=duration_hours)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO sessions (user_id, session_token, expires_at)
            VALUES (?, ?, ?)
        ''', (user_id, session_token, expires_at))
        
        conn.commit()
        conn.close()
        
        return session_token
    
    def get_user_stats(self) -> dict:
        """Récupère les statistiques des utilisateurs"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Nombre total d'utilisateurs
        cursor.execute("SELECT COUNT(*) FROM users")
        total_users = cursor.fetchone()[0]
        
        # Utilisateurs actifs (connectés dans les 7 derniers jours)
        cursor.execute('''
            SELECT COUNT(*) FROM users 
            WHERE last_login >= datetime('now', '-7 days')
        ''')
        active_users = cursor.fetchone()[0]
        
        # Nouvelles inscriptions (dernières 24h)
        cursor.execute('''
            SELECT COUNT(*) FROM users 
            WHERE created_at >= datetime('now', '-1 day')
        ''')
        new_users = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total': total_users,
            'active': active_users,
            'new': new_users
        }
    
    def change_password(self, user_id: int, old_password: str, new_password: str) -> tuple:
        """
        Change le mot de passe d'un utilisateur
        Returns: (success: bool, message: str)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Récupère l'utilisateur
            cursor.execute('''
                SELECT password_hash, salt FROM users WHERE id = ?
            ''', (user_id,))
            
            result = cursor.fetchone()
            if not result:
                conn.close()
                return False, "❌ Utilisateur non trouvé"
            
            stored_hash, salt = result
            
            # Vérifie l'ancien mot de passe
            old_hash, _ = self.hash_password(old_password, salt)
            
            if old_hash != stored_hash:
                conn.close()
                return False, "❌ Ancien mot de passe incorrect"
            
            # Valide le nouveau mot de passe
            if len(new_password) < 8:
                conn.close()
                return False, "❌ Le nouveau mot de passe doit contenir au moins 8 caractères"
            
            # Hash le nouveau mot de passe
            new_hash, new_salt = self.hash_password(new_password)
            
            # Met à jour
            cursor.execute('''
                UPDATE users 
                SET password_hash = ?, salt = ?
                WHERE id = ?
            ''', (new_hash, new_salt, user_id))
            
            conn.commit()
            conn.close()
            
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
        st.session_state.auth_page = 'login'  # 'login' ou 'register'


def logout():
    """Déconnexion de l'utilisateur"""
    st.session_state.authenticated = False
    st.session_state.user_data = None
    st.rerun()


def require_auth(func):
    """
    Décorateur pour protéger les pages nécessitant une authentification
    Usage: @require_auth
    """
    def wrapper(*args, **kwargs):
        if not st.session_state.get('authenticated', False):
            st.error("🔒 Accès refusé. Veuillez vous connecter.")
            st.stop()
        return func(*args, **kwargs)
    return wrapper
