import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from datetime import datetime, timedelta
import csv
import os
import json

# =============================================
# CONFIGURATION - À REMPLIR
# =============================================
SENDER_EMAIL = os.environ.get('NEWSLETTER_EMAIL', 'votre-email@gmail.com')
SENDER_PASSWORD = os.environ.get('NEWSLETTER_PASSWORD', 'votre-mot-de-passe-app')
FINNHUB_API_KEY = os.environ.get('FINNHUB_API_KEY', 'd14re49r01qop9mf2algd14re49r01qop9mf2am0')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# =============================================
# GÉNÉRATION DE SYNTHÈSE AVEC GROK
# =============================================
def generate_synthesis_with_grok(news_list):
    """Génère une synthèse intelligente des news avec Grok"""
    try:
        # Préparer les articles pour Grok
        articles_text = ""
        for i, news in enumerate(news_list[:30], 1):
            headline = news.get('headline', '')
            summary = news.get('summary', '')
            source = news.get('source', '')
            category = news.get('category', 'general')
            
            articles_text += f"\n[Article {i}] ({category.upper()}) - {source}\n"
            articles_text += f"Titre: {headline}\n"
            if summary:
                articles_text += f"Résumé: {summary}\n"
            articles_text += "---\n"
        
        # Prompt pour Grok
        prompt = f"""Tu es un analyste financier Bloomberg. Voici les 30 principaux articles de la semaine des marchés financiers.

{articles_text}

Ta mission: Rédiger une synthèse percutante style Bloomberg Terminal avec:

1. Un paragraphe d'introduction (2-3 phrases) sur le climat général des marchés cette semaine

2. Les 5-7 TENDANCES CLÉS de la semaine, chacune avec:
   - Un titre court et impactant (style Bloomberg)
   - 2-3 phrases d'explication
   - Les faits marquants

3. Une conclusion prospective (1-2 phrases)

Format: Texte fluide et professionnel, sans bullet points. Ton sérieux mais accessible. Mets l'accent sur l'impact pour les investisseurs.

Maximum 8 paragraphes au total."""

        # Appel API Grok
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 2000
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            synthesis = result['choices'][0]['message']['content']
            print("✅ Synthèse générée par Grok")
            return synthesis
        else:
            print(f"❌ Erreur API Grok: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Erreur génération synthèse: {e}")
        return None

# =============================================
# RÉCUPÉRATION DES NEWS DE LA SEMAINE
# =============================================
def get_weekly_news():
    """Récupère les news de la semaine via Finnhub"""
    try:
        # Calcul des dates (lundi à dimanche)
        today = datetime.now()
        days_since_monday = today.weekday()  # 0 = lundi, 6 = dimanche
        monday = today - timedelta(days=days_since_monday)
        
        all_news = []
        
        # Récupérer les news générales
        categories = ["general", "forex", "crypto", "merger"]
        
        for category in categories:
            url = f"https://finnhub.io/api/v1/news?category={category}&token={FINNHUB_API_KEY}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                news = response.json()
                # Filtrer les news de la semaine
                for item in news:
                    timestamp = item.get('datetime', 0)
                    news_date = datetime.fromtimestamp(timestamp)
                    if news_date >= monday:
                        item['category'] = category
                        all_news.append(item)
        
        # Trier par date (plus récent en premier)
        all_news.sort(key=lambda x: x.get('datetime', 0), reverse=True)
        
        return all_news[:30]  # Top 30 news
    except Exception as e:
        print(f"Erreur récupération news: {e}")
        return []

# =============================================
# GÉNÉRATION HTML BLOOMBERG
# =============================================
def generate_newsletter_html(news_list, synthesis_text):
    """Génère l'email HTML style Bloomberg Terminal avec synthèse Grok"""
    
    today = datetime.now()
    week_start = (today - timedelta(days=today.weekday())).strftime("%d/%m/%Y")
    week_end = today.strftime("%d/%m/%Y")
    
    # Convertir la synthèse en HTML (paragraphes)
    synthesis_html = ""
    if synthesis_text:
        paragraphs = synthesis_text.split('\n\n')
        for para in paragraphs:
            if para.strip():
                synthesis_html += f'<p style="color:#AAA;font-size:12px;line-height:1.7;margin-bottom:15px;">{para.strip()}</p>\n'
    else:
        # Fallback si Grok ne marche pas
        synthesis_html = '<p style="color:#AAA;font-size:12px;">Synthèse non disponible cette semaine.</p>'
    
    # Sélectionner quelques articles phares pour la section "Sources"
    top_articles_html = ""
    for news in news_list[:10]:
        headline = news.get('headline', '')
        url = news.get('url', '#')
        source = news.get('source', '')
        
        top_articles_html += f"""
        <div style="background:#0a0a0a;border-left:2px solid #333;padding:8px 12px;margin:6px 0;">
            <a href="{url}" style="color:#00FFFF;text-decoration:none;font-size:10px;" target="_blank">
                {headline[:80]}{'...' if len(headline) > 80 else ''}
            </a>
            <span style="color:#666;font-size:9px;margin-left:10px;">— {source}</span>
        </div>
        """
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body style="margin:0;padding:0;font-family:'Courier New',monospace;background:#000;color:#FFAA00;">
        <div style="max-width:700px;margin:0 auto;background:#000;">
            
            <!-- HEADER -->
            <div style="background:#FFAA00;padding:20px;text-align:center;">
                <div style="color:#000;font-size:24px;font-weight:bold;letter-spacing:3px;">
                    ⬛ BLOOMBERG ENS®
                </div>
                <div style="color:#000;font-size:12px;margin-top:5px;letter-spacing:1px;">
                    WEEKLY MARKET DIGEST
                </div>
            </div>
            
            <!-- INTRO -->
            <div style="background:#111;border-bottom:2px solid #FFAA00;padding:20px;">
                <div style="color:#FFAA00;font-size:14px;font-weight:bold;margin-bottom:10px;">
                    📅 SEMAINE DU {week_start} AU {week_end}
                </div>
                <div style="color:#888;font-size:11px;line-height:1.6;">
                    Analyse synthétique des tendances clés qui ont marqué les marchés cette semaine.
                </div>
            </div>
            
            <!-- SYNTHÈSE GROK -->
            <div style="padding:25px 20px;">
                <div style="background:#FFAA00;color:#000;padding:10px 15px;font-weight:bold;font-size:13px;margin-bottom:20px;letter-spacing:2px;">
                    📊 ANALYSE DE LA SEMAINE
                </div>
                
                <div style="background:#111;border:1px solid #333;border-left:4px solid #FFAA00;padding:20px;">
                    {synthesis_html}
                </div>
            </div>
            
            <!-- SOURCES -->
            <div style="padding:0 20px 25px 20px;">
                <div style="background:#00FFFF;color:#000;padding:8px 15px;font-weight:bold;font-size:11px;margin-bottom:15px;letter-spacing:1px;">
                    📰 SOURCES PRINCIPALES
                </div>
                {top_articles_html}
            </div>
            
            <!-- FOOTER -->
            <div style="background:#111;border-top:2px solid #FFAA00;padding:20px;text-align:center;margin-top:30px;">
                <div style="color:#666;font-size:10px;line-height:1.6;">
                    © 2025 BLOOMBERG ENS® | NEWSLETTER HEBDOMADAIRE<br>
                    Powered by Finnhub API + Grok AI • Envoyé le {today.strftime("%d/%m/%Y à %H:%M")}<br><br>
                    <a href="mailto:{SENDER_EMAIL}?subject=Unsubscribe" style="color:#00FFFF;text-decoration:none;">
                        Se désabonner
                    </a>
                </div>
            </div>
            
        </div>
    </body>
    </html>
    """
    
    return html

# =============================================
# LECTURE DES ABONNÉS
# =============================================
def get_subscribers():
    """Lit la liste des abonnés depuis le CSV"""
    subscribers = []
    try:
        with open('newsletter_subscribers.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('active', '1') == '1':
                    subscribers.append(row['email'])
    except FileNotFoundError:
        print("Fichier newsletter_subscribers.csv non trouvé")
    except Exception as e:
        print(f"Erreur lecture CSV: {e}")
    
    return subscribers

# =============================================
# ENVOI EMAIL
# =============================================
def send_email(to_email, html_content):
    """Envoie l'email à un destinataire"""
    try:
        message = MIMEMultipart("alternative")
        message["Subject"] = f"📰 Bloomberg ENS® Weekly Digest - {datetime.now().strftime('%d/%m/%Y')}"
        message["From"] = SENDER_EMAIL
        message["To"] = to_email
        
        part = MIMEText(html_content, "html")
        message.attach(part)
        
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(message)
        
        print(f"✅ Email envoyé à {to_email}")
        return True
    except Exception as e:
        print(f"❌ Erreur envoi à {to_email}: {e}")
        return False

# =============================================
# FONCTION PRINCIPALE
# =============================================
def send_weekly_newsletter():
    """Fonction principale d'envoi de la newsletter"""
    print(f"\n🚀 Début envoi newsletter hebdomadaire - {datetime.now()}")
    
    # 1. Récupérer les news
    print("📡 Récupération des news de la semaine...")
    news_list = get_weekly_news()
    
    if not news_list:
        print("❌ Aucune news récupérée. Abandon.")
        return
    
    print(f"✅ {len(news_list)} news récupérées")
    
    # 2. Générer la synthèse avec Grok
    print("🤖 Génération de la synthèse avec Grok AI...")
    synthesis = generate_synthesis_with_grok(news_list)
    
    if not synthesis:
        print("⚠️ Synthèse Grok non disponible, utilisation du format basique")
        synthesis = "Synthèse non disponible cette semaine. Veuillez consulter les sources ci-dessous."
    
    # 3. Générer l'HTML
    print("🎨 Génération du template HTML...")
    html_content = generate_newsletter_html(news_list, synthesis)
    
    # 4. Récupérer les abonnés
    print("📋 Lecture des abonnés...")
    subscribers = get_subscribers()
    
    if not subscribers:
        print("❌ Aucun abonné trouvé")
        return
    
    print(f"✅ {len(subscribers)} abonné(s) trouvé(s)")
    
    # 5. Envoyer les emails
    print("📧 Envoi des emails...")
    success_count = 0
    
    for email in subscribers:
        if send_email(email, html_content):
            success_count += 1
        # Pause pour éviter les limites d'envoi
        import time
        time.sleep(2)
    
    print(f"\n✅ Newsletter envoyée à {success_count}/{len(subscribers)} abonné(s)")
    print(f"🏁 Terminé à {datetime.now()}\n")

# =============================================
# EXÉCUTION
# =============================================
if __name__ == "__main__":
    send_weekly_newsletter()
