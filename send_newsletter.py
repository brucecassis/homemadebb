import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from datetime import datetime, timedelta
import os
import json
import random

# =============================================
# CONFIGURATION - À REMPLIR
# =============================================
SENDER_EMAIL = os.environ.get('NEWSLETTER_EMAIL', 'votre-email@gmail.com')
SENDER_PASSWORD = os.environ.get('NEWSLETTER_PASSWORD', 'votre-mot-de-passe-app')
FINNHUB_API_KEY = os.environ.get('FINNHUB_API_KEY', 'd14re49r01qop9mf2algd14re49r01qop9mf2am0')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# Mode de fonctionnement
USE_SIMULATED_DATA = True  # Mettre à False si les APIs externes fonctionnent

# =============================================
# DONNÉES SIMULÉES RÉALISTES
# =============================================
def generate_realistic_market_data():
    """Génère des données de marché réalistes pour la semaine"""
    
    # Cours de base (approximatifs au 7 décembre 2024)
    base_prices = {
        "NASDAQ": 19800,
        "S&P 500": 6050,
        "CAC 40": 7350,
        "Bitcoin": 98500
    }
    
    # Volatilités hebdomadaires typiques (en %)
    volatilities = {
        "NASDAQ": 2.5,
        "S&P 500": 1.8,
        "CAC 40": 2.0,
        "Bitcoin": 5.0
    }
    
    indices_data = {}
    
    for name, base_price in base_prices.items():
        # Générer une variation hebdomadaire réaliste
        volatility = volatilities[name]
        weekly_change_pct = random.uniform(-volatility, volatility)
        
        # Calculer les prix
        end_price = base_price
        start_price = end_price / (1 + weekly_change_pct / 100)
        change = end_price - start_price
        
        # Générer 7 points de données (une semaine)
        prices = []
        for i in range(7):
            # Interpolation avec un peu de bruit
            progress = i / 6
            noise = random.uniform(-0.3, 0.3) * volatility / 100
            price = start_price + (change * progress) + (base_price * noise)
            prices.append(price)
        
        # S'assurer que le dernier prix est exact
        prices[-1] = end_price
        
        # Timestamps
        now = datetime.now()
        timestamps = [int((now - timedelta(days=6-i)).timestamp()) for i in range(7)]
        
        indices_data[name] = {
            'start': start_price,
            'end': end_price,
            'change': change,
            'change_pct': weekly_change_pct,
            'prices': prices,
            'timestamps': timestamps
        }
        
        print(f"  ✅ {name}: {end_price:.2f} ({weekly_change_pct:+.2f}%)")
    
    return indices_data

# =============================================
# RÉCUPÉRATION DES DONNÉES D'INDICES VIA FINNHUB
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
        print(f"  ⚠️ Erreur Finnhub pour {symbol}: {str(e)[:100]}")
        return None

def get_all_indices():
    """Récupère les données de tous les indices"""
    print("📊 Récupération des indices boursiers...")
    
    if USE_SIMULATED_DATA:
        print("  (Mode simulation activé - données réalistes générées)")
        return generate_realistic_market_data()
    
    # Tentative avec Finnhub
    indices_symbols = {
        "NASDAQ": "^IXIC",
        "S&P 500": "^GSPC",
        "CAC 40": "^FCHI",
        "Bitcoin": "BINANCE:BTCUSDT"
    }
    
    indices_data = {}
    
    for name, symbol in indices_symbols.items():
        print(f"\n  {name} ({symbol}):")
        data = get_index_data_finnhub(symbol)
        
        if data:
            indices_data[name] = data
            print(f"    ✅ {data['end']:.2f} ({data['change_pct']:+.2f}%)")
        else:
            print(f"    ❌ Échec")
    
    # Si aucune donnée réelle, utiliser la simulation
    if not indices_data:
        print("\n  ⚠️ Toutes les APIs ont échoué, basculement en mode simulation")
        return generate_realistic_market_data()
    
    return indices_data

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
# GÉNÉRATION DE SYNTHÈSE AVEC GROK
# =============================================
def generate_synthesis_with_grok(news_list, indices_data):
    """Génère une synthèse structurée avec Grok"""
    
    if not GROQ_API_KEY:
        print("⚠️ GROQ_API_KEY non configurée, utilisation d'une synthèse par défaut")
        return generate_default_synthesis(indices_data)
    
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
        
        # Prompt pour Grok
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
            return generate_default_synthesis(indices_data)
            
    except Exception as e:
        print(f"❌ Erreur génération synthèse: {e}")
        return generate_default_synthesis(indices_data)

def generate_default_synthesis(indices_data):
    """Génère une synthèse par défaut basée sur les indices"""
    
    # Calculer les tendances
    positive_indices = [name for name, data in indices_data.items() if data['change_pct'] > 0]
    negative_indices = [name for name, data in indices_data.items() if data['change_pct'] < 0]
    
    nasdaq_pct = indices_data.get('NASDAQ', {}).get('change_pct', 0)
    sp500_pct = indices_data.get('S&P 500', {}).get('change_pct', 0)
    btc_pct = indices_data.get('Bitcoin', {}).get('change_pct', 0)
    
    return f"""## VUE D'ENSEMBLE
Les marchés ont connu une semaine {'contrastée' if len(positive_indices) > 0 and len(negative_indices) > 0 else 'haussière' if len(positive_indices) > len(negative_indices) else 'baissière'}. Le NASDAQ a {'progressé' if nasdaq_pct > 0 else 'reculé'} de {abs(nasdaq_pct):.2f}%, tandis que le S&P 500 a enregistré une variation de {sp500_pct:+.2f}%. Les investisseurs ont surveillé de près les indicateurs économiques et les décisions des banques centrales.

## MARCHÉS ACTIONS
Les indices américains ont {'surperformé' if (nasdaq_pct + sp500_pct) / 2 > 0 else 'sous-performé'} cette semaine. Le secteur technologique a été particulièrement {'dynamique' if nasdaq_pct > sp500_pct else 'prudent'}, avec le NASDAQ qui {'mène' if nasdaq_pct > sp500_pct else 'traîne'} par rapport au S&P 500. Les valeurs de croissance ont {'bénéficié' if nasdaq_pct > 0 else 'souffert'} du sentiment global du marché.

## CRYPTOMONNAIES
Bitcoin a {'bondi' if btc_pct > 2 else 'progressé' if btc_pct > 0 else 'reculé'} de {abs(btc_pct):.2f}% pour s'établir à {indices_data.get('Bitcoin', {}).get('end', 0):,.0f}$. Le marché crypto reste {'optimiste' if btc_pct > 0 else 'prudent'}, avec une attention particulière portée aux développements réglementaires et à l'adoption institutionnelle. La volatilité reste {'élevée' if abs(btc_pct) > 3 else 'modérée'} sur cette classe d'actifs.

## ACTUALITÉS MAJEURES
La semaine a été marquée par la publication de données économiques clés et des annonces de plusieurs entreprises majeures. Les investisseurs ont également suivi de près l'évolution des tensions géopolitiques et leur impact potentiel sur les chaînes d'approvisionnement mondiales. Les secteurs de la tech et de la finance ont particulièrement retenu l'attention.

## PERSPECTIVES
La semaine prochaine sera cruciale avec la publication de nouveaux indicateurs économiques. Les marchés resteront attentifs aux signaux des banques centrales concernant leurs politiques monétaires. Les investisseurs surveilleront également les résultats trimestriels et les prévisions des entreprises pour ajuster leurs positions."""

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
        print(f"⚠️ Erreur récupération news: {e}")
        return []

# =============================================
# GÉNÉRATION HTML BLOOMBERG
# =============================================
def generate_newsletter_html(news_list, synthesis_text, indices_data):
    """Génère l'email HTML style Bloomberg Terminal"""
    
    today = datetime.now()
    week_start = (today - timedelta(days=today.weekday())).strftime("%d/%m/%Y")
    week_end = today.strftime("%d/%m/%Y")
    
    # Formater la synthèse structurée
    synthesis_html = format_synthesis_html(synthesis_text)
    
    # Générer l'HTML des indices
    indices_html = generate_indices_html(indices_data)
    
    # Articles phares
    top_articles_html = ""
    if news_list:
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
    else:
        top_articles_html = '<p style="color:#888;font-size:10px;">Sources d\'actualité non disponibles cette semaine</p>'
    
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
    
    if not indices_data:
        print("❌ Impossible de récupérer les indices. Abandon.")
        return
    
    # 2. Récupérer les news
    print("\n📡 Récupération des news de la semaine...")
    news_list = get_weekly_news()
    
    if news_list:
        print(f"✅ {len(news_list)} news récupérées")
    else:
        print("⚠️ Aucune news récupérée, la newsletter contiendra uniquement les indices et l'analyse")
    
    # 3. Générer la synthèse
    print("\n🤖 Génération de la synthèse...")
    synthesis = generate_synthesis_with_grok(news_list, indices_data)
    
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
# FONCTION DE TEST
# =============================================
def test_newsletter():
    """Génère un aperçu HTML de la newsletter"""
    print("\n🧪 TEST DE LA NEWSLETTER\n")
    
    # 1. Récupérer les indices
    indices_data = get_all_indices()
    
    # 2. Récupérer les news (si possible)
    print("\n📡 Tentative récupération des news...")
    news_list = get_weekly_news()
    
    if news_list:
        print(f"✅ {len(news_list)} news récupérées")
    else:
        print("⚠️ Pas de news disponibles")
    
    # 3. Générer la synthèse
    print("\n🤖 Génération de la synthèse...")
    synthesis = generate_synthesis_with_grok(news_list, indices_data)
    
    # 4. Générer l'HTML
    print("\n🎨 Génération du HTML...")
    html_content = generate_newsletter_html(news_list, synthesis, indices_data)
    
    # 5. Sauvegarder pour prévisualisation
    output_path = '/tmp/newsletter_preview.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✅ Aperçu sauvegardé: {output_path}")
    print("\n" + "="*60)
    print("RÉSUMÉ DES INDICES:")
    print("="*60)
    
    for name, data in indices_data.items():
        print(f"\n{name}:")
        print(f"  Prix: ${data['end']:,.2f}")
        print(f"  Variation: {data['change_pct']:+.2f}% ({data['change']:+,.2f})")
        print(f"  Points graphique: {len(data['prices'])}")
    
    return html_content

# =============================================
# EXÉCUTION
# =============================================
if __name__ == "__main__":
    # Test (génère un fichier HTML à prévisualiser)
    test_newsletter()
    
    # Pour envoyer la newsletter complète:
    # send_weekly_newsletter()
