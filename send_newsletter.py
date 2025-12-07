import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from datetime import datetime, timedelta
import os
import json
import base64
from io import BytesIO

# =============================================
# CONFIGURATION - À REMPLIR
# =============================================
SENDER_EMAIL = os.environ.get('NEWSLETTER_EMAIL', 'votre-email@gmail.com')
SENDER_PASSWORD = os.environ.get('NEWSLETTER_PASSWORD', 'votre-mot-de-passe-app')
FINNHUB_API_KEY = os.environ.get('FINNHUB_API_KEY', 'd14re49r01qop9mf2algd14re49r01qop9mf2am0')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# Indices à suivre avec support multi-sources
INDICES_CONFIG = {
    "NASDAQ": {
        "yahoo": "^IXIC",
        "finnhub": "^IXIC",
        "alphavantage": "NDAQ"
    },
    "S&P 500": {
        "yahoo": "^GSPC",
        "finnhub": "^GSPC",
        "alphavantage": "SPX"
    },
    "CAC 40": {
        "yahoo": "^FCHI",
        "finnhub": "^FCHI",
        "alphavantage": "FCHI"
    },
    "Bitcoin": {
        "yahoo": "BTC-USD",
        "finnhub": "BINANCE:BTCUSDT",
        "coinbase": "BTC-USD"
    }
}

# =============================================
# RÉCUPÉRATION DES DONNÉES D'INDICES (MÉTHODE 1: Yahoo Finance via API)
# =============================================
def get_index_data_yahoo(symbol):
    """Récupère les données via Yahoo Finance (API gratuite)"""
    try:
        # Utiliser l'API Yahoo Finance v8 (gratuite)
        now = int(datetime.now().timestamp())
        week_ago = int((datetime.now() - timedelta(days=7)).timestamp())
        
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {
            "period1": week_ago,
            "period2": now,
            "interval": "1d"
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            result = data.get('chart', {}).get('result', [])
            
            if result:
                quotes = result[0].get('indicators', {}).get('quote', [{}])[0]
                closes = quotes.get('close', [])
                timestamps = result[0].get('timestamp', [])
                
                # Filtrer les None
                valid_closes = [c for c in closes if c is not None]
                
                if len(valid_closes) >= 2:
                    start_price = valid_closes[0]
                    end_price = valid_closes[-1]
                    change = end_price - start_price
                    change_pct = (change / start_price) * 100
                    
                    return {
                        'start': start_price,
                        'end': end_price,
                        'change': change,
                        'change_pct': change_pct,
                        'prices': valid_closes,
                        'timestamps': timestamps
                    }
        return None
    except Exception as e:
        print(f"  ⚠️ Erreur Yahoo Finance pour {symbol}: {e}")
        return None

# =============================================
# RÉCUPÉRATION DES DONNÉES D'INDICES (MÉTHODE 2: Finnhub)
# =============================================
def get_index_data_finnhub(symbol):
    """Récupère les données d'un indice via Finnhub"""
    try:
        now = datetime.now()
        week_ago = now - timedelta(days=7)
        
        to_timestamp = int(now.timestamp())
        from_timestamp = int(week_ago.timestamp())
        
        url = f"https://finnhub.io/api/v1/stock/candle?symbol={symbol}&resolution=D&from={from_timestamp}&to={to_timestamp}&token={FINNHUB_API_KEY}"
        
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('s') == 'ok' and data.get('c'):
                closes = data['c']
                if len(closes) >= 2:
                    start_price = closes[0]
                    end_price = closes[-1]
                    change = end_price - start_price
                    change_pct = (change / start_price) * 100
                    
                    return {
                        'start': start_price,
                        'end': end_price,
                        'change': change,
                        'change_pct': change_pct,
                        'prices': closes,
                        'timestamps': data['t']
                    }
        return None
    except Exception as e:
        print(f"  ⚠️ Erreur Finnhub pour {symbol}: {e}")
        return None

# =============================================
# RÉCUPÉRATION DES DONNÉES D'INDICES (MÉTHODE 3: CoinGecko pour crypto)
# =============================================
def get_crypto_data_coingecko(crypto_id='bitcoin'):
    """Récupère les données crypto via CoinGecko (gratuit)"""
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
        params = {
            "vs_currency": "usd",
            "days": "7",
            "interval": "daily"
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            prices_data = data.get('prices', [])
            
            if len(prices_data) >= 2:
                prices = [p[1] for p in prices_data]
                start_price = prices[0]
                end_price = prices[-1]
                change = end_price - start_price
                change_pct = (change / start_price) * 100
                
                return {
                    'start': start_price,
                    'end': end_price,
                    'change': change,
                    'change_pct': change_pct,
                    'prices': prices,
                    'timestamps': [p[0]//1000 for p in prices_data]
                }
        return None
    except Exception as e:
        print(f"  ⚠️ Erreur CoinGecko: {e}")
        return None

# =============================================
# RÉCUPÉRATION INTELLIGENTE MULTI-SOURCES
# =============================================
def get_index_data_smart(name, config):
    """Essaye plusieurs sources dans l'ordre jusqu'à succès"""
    
    # Stratégie par type d'actif
    if "Bitcoin" in name:
        # Pour crypto: essayer CoinGecko d'abord (meilleur pour crypto)
        print(f"  Tentative CoinGecko...")
        data = get_crypto_data_coingecko('bitcoin')
        if data:
            return data
        
        # Fallback Yahoo
        print(f"  Tentative Yahoo Finance...")
        data = get_index_data_yahoo(config['yahoo'])
        if data:
            return data
    else:
        # Pour indices boursiers: essayer Yahoo d'abord (plus fiable)
        print(f"  Tentative Yahoo Finance...")
        data = get_index_data_yahoo(config['yahoo'])
        if data:
            return data
        
        # Fallback Finnhub
        print(f"  Tentative Finnhub...")
        data = get_index_data_finnhub(config['finnhub'])
        if data:
            return data
    
    return None

def generate_sparkline_svg(prices, width=120, height=30):
    """Génère un mini graphique SVG sparkline"""
    if not prices or len(prices) < 2:
        return ""
    
    min_price = min(prices)
    max_price = max(prices)
    price_range = max_price - min_price if max_price != min_price else 1
    
    # Calculer les points
    points = []
    for i, price in enumerate(prices):
        x = (i / (len(prices) - 1)) * width
        y = height - ((price - min_price) / price_range) * height
        points.append(f"{x:.2f},{y:.2f}")
    
    # Déterminer la couleur (vert si hausse, rouge si baisse)
    color = "#00FF00" if prices[-1] >= prices[0] else "#FF0000"
    
    svg = f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
        <polyline points="{' '.join(points)}" fill="none" stroke="{color}" stroke-width="1.5"/>
    </svg>'''
    
    return svg

def get_all_indices():
    """Récupère les données de tous les indices avec système de fallback"""
    indices_data = {}
    
    print("📊 Récupération des indices boursiers...")
    
    for name, config in INDICES_CONFIG.items():
        print(f"\n  {name}:")
        data = get_index_data_smart(name, config)
        
        if data:
            indices_data[name] = data
            print(f"    ✅ {data['end']:.2f} ({data['change_pct']:+.2f}%)")
        else:
            print(f"    ❌ Toutes les sources ont échoué")
    
    return indices_data

# =============================================
# GÉNÉRATION HTML DES INDICES
# =============================================
def generate_indices_html(indices_data):
    """Génère le HTML pour afficher les indices"""
    if not indices_data:
        return '<p style="color:#888;">Données des indices non disponibles</p>'
    
    html = '<div style="display:grid;grid-template-columns:1fr 1fr;gap:15px;">'
    
    for name, data in indices_data.items():
        change_pct = data['change_pct']
        color = "#00FF00" if change_pct >= 0 else "#FF0000"
        arrow = "▲" if change_pct >= 0 else "▼"
        
        sparkline = generate_sparkline_svg(data['prices'])
        
        html += f'''
        <div style="background:#0a0a0a;border:1px solid #333;border-left:3px solid {color};padding:12px;">
            <div style="color:#AAA;font-size:10px;font-weight:bold;margin-bottom:5px;">
                {name}
            </div>
            <div style="color:#FFF;font-size:16px;font-weight:bold;margin-bottom:5px;">
                {data['end']:,.2f}
            </div>
            <div style="color:{color};font-size:11px;font-weight:bold;margin-bottom:8px;">
                {arrow} {change_pct:+.2f}% ({data['change']:+,.2f})
            </div>
            <div style="margin-top:8px;">
                {sparkline}
            </div>
        </div>
        '''
    
    html += '</div>'
    return html

# =============================================
# GÉNÉRATION DE SYNTHÈSE AVEC GROK (AMÉLIORÉE)
# =============================================
def generate_synthesis_with_grok(news_list, indices_data):
    """Génère une synthèse structurée avec Grok"""
    try:
        # Préparer les articles
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
        
        # Préparer les données des indices
        indices_text = "\n\nPERFORMANCES DES INDICES (semaine):\n"
        for name, data in indices_data.items():
            indices_text += f"- {name}: {data['change_pct']:+.2f}% (de {data['start']:.2f} à {data['end']:.2f})\n"
        
        # Prompt amélioré pour Grok
        prompt = f"""Tu es un analyste financier Bloomberg. Voici les données de la semaine:

{articles_text}
{indices_text}

Rédige une synthèse STRUCTURÉE en 5 sections distinctes:

## VUE D'ENSEMBLE
Un paragraphe synthétique (3-4 phrases) résumant l'ambiance générale des marchés et les performances des indices cette semaine.

## MARCHÉS ACTIONS
2-3 phrases sur les tendances des marchés actions (S&P 500, NASDAQ, CAC 40), les secteurs performants/sous-performants, et les catalyseurs principaux.

## CRYPTOMONNAIES
2-3 phrases sur Bitcoin et le marché crypto: évolution, catalyseurs, sentiment du marché.

## ACTUALITÉS MAJEURES
3-4 phrases couvrant les événements clés de la semaine (annonces d'entreprises, données macroéconomiques, actualité géopolitique, fusions/acquisitions).

## PERSPECTIVES
2-3 phrases sur les points d'attention pour la semaine prochaine et les facteurs à surveiller.

IMPORTANT: 
- Utilise les titres de section EXACTEMENT comme indiqués (avec ##)
- Style professionnel mais accessible
- Intègre les chiffres des indices fournis
- Ton objectif et factuel
- Maximum 12 phrases au total"""

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
                "max_tokens": 2500
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

def format_synthesis_html(synthesis_text):
    """Convertit la synthèse en HTML structuré avec sections colorées"""
    if not synthesis_text:
        return '<p style="color:#888;">Synthèse non disponible</p>'
    
    html = ""
    sections = synthesis_text.split('##')
    
    # Couleurs par section
    section_colors = {
        "VUE D'ENSEMBLE": "#FFAA00",
        "MARCHÉS ACTIONS": "#00AAFF",
        "CRYPTOMONNAIES": "#FF9500",
        "ACTUALITÉS MAJEURES": "#00FF88",
        "PERSPECTIVES": "#FF6B9D"
    }
    
    for section in sections:
        if not section.strip():
            continue
        
        lines = section.strip().split('\n', 1)
        if len(lines) == 2:
            title = lines[0].strip()
            content = lines[1].strip()
            
            color = section_colors.get(title, "#FFAA00")
            
            html += f'''
            <div style="margin-bottom:25px;">
                <div style="background:{color};color:#000;padding:8px 12px;font-weight:bold;font-size:11px;letter-spacing:1px;margin-bottom:10px;">
                    {title}
                </div>
                <div style="background:#0a0a0a;border-left:3px solid {color};padding:15px;color:#CCC;font-size:12px;line-height:1.7;">
                    {content}
                </div>
            </div>
            '''
    
    return html

# =============================================
# RÉCUPÉRATION DES NEWS DE LA SEMAINE
# =============================================
def get_weekly_news():
    """Récupère les news de la semaine via Finnhub"""
    try:
        today = datetime.now()
        days_since_monday = today.weekday()
        monday = today - timedelta(days=days_since_monday)
        
        all_news = []
        categories = ["general", "forex", "crypto", "merger"]
        
        for category in categories:
            url = f"https://finnhub.io/api/v1/news?category={category}&token={FINNHUB_API_KEY}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                news = response.json()
                for item in news:
                    timestamp = item.get('datetime', 0)
                    news_date = datetime.fromtimestamp(timestamp)
                    if news_date >= monday:
                        item['category'] = category
                        all_news.append(item)
        
        all_news.sort(key=lambda x: x.get('datetime', 0), reverse=True)
        return all_news[:30]
    except Exception as e:
        print(f"Erreur récupération news: {e}")
        return []

# =============================================
# GÉNÉRATION HTML BLOOMBERG AMÉLIORÉE
# =============================================
def generate_newsletter_html(news_list, synthesis_text, indices_data):
    """Génère l'email HTML style Bloomberg Terminal amélioré"""
    
    today = datetime.now()
    week_start = (today - timedelta(days=today.weekday())).strftime("%d/%m/%Y")
    week_end = today.strftime("%d/%m/%Y")
    
    # Formater la synthèse structurée
    synthesis_html = format_synthesis_html(synthesis_text)
    
    # Générer l'HTML des indices
    indices_html = generate_indices_html(indices_data)
    
    # Articles phares
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
            
            <!-- INDICES BOURSIERS -->
            <div style="padding:25px 20px;">
                <div style="background:#00FFFF;color:#000;padding:10px 15px;font-weight:bold;font-size:13px;margin-bottom:20px;letter-spacing:2px;">
                    📈 PERFORMANCES DE LA SEMAINE
                </div>
                {indices_html}
            </div>
            
            <!-- SYNTHÈSE STRUCTURÉE -->
            <div style="padding:0 20px 25px 20px;">
                <div style="background:#FFAA00;color:#000;padding:10px 15px;font-weight:bold;font-size:13px;margin-bottom:20px;letter-spacing:2px;">
                    📊 ANALYSE DÉTAILLÉE
                </div>
                {synthesis_html}
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
# LECTURE DES ABONNÉS DEPUIS SUPABASE
# =============================================
def get_subscribers():
    """Lit la liste des abonnés depuis Supabase"""
    try:
        supabase_url = os.environ.get('SUPABASE_URL', '')
        supabase_key = os.environ.get('SUPABASE_KEY', '')
        
        if not supabase_url or not supabase_key:
            print("❌ Identifiants Supabase manquants")
            return []
        
        response = requests.get(
            f"{supabase_url}/rest/v1/emails?active=eq.true&select=email",
            headers={
                "apikey": supabase_key,
                "Authorization": f"Bearer {supabase_key}"
            },
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            subscribers = [item['email'] for item in data]
            return subscribers
        else:
            print(f"❌ Erreur Supabase: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"❌ Erreur lecture Supabase: {e}")
        return []

# =============================================
# ENVOI EMAIL
# =============================================
def send_email(to_email, html_content):
    """Envoie l'email à un destinataire"""
    try:
        message = MIMEMultipart("alternative")
        message["Subject"] = f"📊 Bloomberg ENS® Weekly Digest - {datetime.now().strftime('%d/%m/%Y')}"
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
    
    # 1. Récupérer les indices
    indices_data = get_all_indices()
    
    # 2. Récupérer les news
    print("\n📡 Récupération des news de la semaine...")
    news_list = get_weekly_news()
    
    if not news_list:
        print("❌ Aucune news récupérée. Abandon.")
        return
    
    print(f"✅ {len(news_list)} news récupérées")
    
    # 3. Générer la synthèse avec Grok
    print("\n🤖 Génération de la synthèse structurée avec Grok AI...")
    synthesis = generate_synthesis_with_grok(news_list, indices_data)
    
    if not synthesis:
        print("⚠️ Synthèse Grok non disponible")
        synthesis = "Synthèse non disponible cette semaine."
    
    # 4. Générer l'HTML
    print("\n🎨 Génération du template HTML...")
    html_content = generate_newsletter_html(news_list, synthesis, indices_data)
    
    # 5. Récupérer les abonnés
    print("\n📋 Lecture des abonnés...")
    subscribers = get_subscribers()
    
    if not subscribers:
        print("❌ Aucun abonné trouvé")
        return
    
    print(f"✅ {len(subscribers)} abonné(s) trouvé(s)")
    
    # 6. Envoyer les emails
    print("\n📧 Envoi des emails...")
    success_count = 0
    
    for email in subscribers:
        if send_email(email, html_content):
            success_count += 1
        import time
        time.sleep(2)
    
    print(f"\n✅ Newsletter envoyée à {success_count}/{len(subscribers)} abonné(s)")
    print(f"🏁 Terminé à {datetime.now()}\n")

# =============================================
# FONCTION DE TEST (sans envoi)
# =============================================
def test_indices_only():
    """Test rapide pour vérifier que les indices fonctionnent"""
    print("\n🧪 TEST DES INDICES\n")
    indices_data = get_all_indices()
    
    if indices_data:
        print("\n" + "="*50)
        print("RÉSULTATS:")
        print("="*50)
        for name, data in indices_data.items():
            print(f"\n{name}:")
            print(f"  Prix début: {data['start']:.2f}")
            print(f"  Prix fin: {data['end']:.2f}")
            print(f"  Variation: {data['change_pct']:+.2f}%")
            print(f"  Nombre de points: {len(data['prices'])}")
    else:
        print("\n❌ Aucun indice récupéré")

# =============================================
# EXÉCUTION
# =============================================
if __name__ == "__main__":
    # Pour tester uniquement les indices:
    # test_indices_only()
    
    # Pour envoyer la newsletter complète:
    send_weekly_newsletter()
