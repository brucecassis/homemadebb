import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from datetime import datetime, timedelta
import csv
import os

# =============================================
# CONFIGURATION - À REMPLIR
# =============================================
SENDER_EMAIL = os.environ.get('NEWSLETTER_EMAIL', 'votre-email@gmail.com')
SENDER_PASSWORD = os.environ.get('NEWSLETTER_PASSWORD', 'votre-mot-de-passe-app')
FINNHUB_API_KEY = os.environ.get('FINNHUB_API_KEY', 'd14re49r01qop9mf2algd14re49r01qop9mf2am0')

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

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
def generate_newsletter_html(news_list):
    """Génère l'email HTML style Bloomberg Terminal"""
    
    today = datetime.now()
    week_start = (today - timedelta(days=today.weekday())).strftime("%d/%m/%Y")
    week_end = today.strftime("%d/%m/%Y")
    
    # Grouper par catégorie
    news_by_category = {
        "general": [],
        "forex": [],
        "crypto": [],
        "merger": []
    }
    
    for news in news_list:
        cat = news.get('category', 'general')
        if cat in news_by_category:
            news_by_category[cat].append(news)
    
    # Générer les sections HTML
    sections_html = ""
    
    category_names = {
        "general": "📊 GENERAL MARKET",
        "forex": "💱 FOREX & CURRENCIES",
        "crypto": "₿ CRYPTO MARKETS",
        "merger": "🤝 M&A & CORPORATE"
    }
    
    for cat, cat_news in news_by_category.items():
        if not cat_news:
            continue
        
        cat_name = category_names.get(cat, cat.upper())
        
        sections_html += f"""
        <div style="background:#FFAA00;color:#000;padding:10px 15px;font-weight:bold;font-size:13px;margin:25px 0 15px 0;letter-spacing:2px;">
            {cat_name}
        </div>
        """
        
        for news in cat_news[:10]:  # Max 10 par catégorie
            headline = news.get('headline', 'Sans titre')
            url = news.get('url', '#')
            source = news.get('source', 'Source')
            timestamp = news.get('datetime', 0)
            summary = news.get('summary', '')
            
            date_str = datetime.fromtimestamp(timestamp).strftime("%d/%m %H:%M") if timestamp else ""
            
            short_summary = summary[:150] + "..." if len(summary) > 150 else summary
            
            sections_html += f"""
            <div style="background:#111;border:1px solid #333;border-left:4px solid #FFAA00;padding:15px;margin:10px 0;">
                <div style="color:#666;font-size:10px;margin-bottom:5px;">
                    <span style="color:#00FFFF;font-weight:bold;">{source}</span> • {date_str}
                </div>
                <div style="color:#FFAA00;font-size:13px;font-weight:bold;margin-bottom:8px;line-height:1.4;">
                    <a href="{url}" style="color:#FFAA00;text-decoration:none;" target="_blank">{headline}</a>
                </div>
                <div style="color:#AAA;font-size:11px;line-height:1.5;">
                    {short_summary}
                </div>
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
                <div style="color:#AAA;font-size:11px;line-height:1.6;">
                    Voici votre récapitulatif hebdomadaire des principales actualités des marchés financiers.
                    {len(news_list)} articles sélectionnés parmi les sources les plus fiables.
                </div>
            </div>
            
            <!-- CONTENU -->
            <div style="padding:20px;">
                {sections_html}
            </div>
            
            <!-- FOOTER -->
            <div style="background:#111;border-top:2px solid #FFAA00;padding:20px;text-align:center;margin-top:30px;">
                <div style="color:#666;font-size:10px;line-height:1.6;">
                    © 2025 BLOOMBERG ENS® | NEWSLETTER HEBDOMADAIRE<br>
                    Source: Finnhub API • Envoyé le {today.strftime("%d/%m/%Y à %H:%M")}<br><br>
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
    
    # 2. Générer l'HTML
    print("🎨 Génération du template HTML...")
    html_content = generate_newsletter_html(news_list)
    
    # 3. Récupérer les abonnés
    print("📋 Lecture des abonnés...")
    subscribers = get_subscribers()
    
    if not subscribers:
        print("❌ Aucun abonné trouvé")
        return
    
    print(f"✅ {len(subscribers)} abonné(s) trouvé(s)")
    
    # 4. Envoyer les emails
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
