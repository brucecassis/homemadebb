"""
Utilitaire AdSense pour l'application Bloomberg Terminal
Permet d'afficher les publicités AdSense sur toutes les pages
"""

import streamlit.components.v1 as components

# =============================================
# CONFIGURATION ADSENSE
# =============================================
ADSENSE_CLIENT_ID = "ca-pub-2501507539536413"

AD_SLOTS = {
    "header": "6830409870",  # Header bloom
    "footer": "7757135278"   # Foot bloom
}

def display_adsense(position="header", height=250):
    """
    Affiche une publicité Google AdSense
    
    Args:
        position (str): "header" ou "footer"
        height (int): Hauteur de l'espace publicitaire en pixels
    
    Example:
        display_adsense("header", height=120)
        display_adsense("footer", height=150)
    """
    ad_slot = AD_SLOTS.get(position, AD_SLOTS["header"])
    ad_name = "header bloom" if position == "header" else "foot bloom"
    
    ad_code = f"""
    <div style="background: #111; border: 1px solid #333; padding: 10px; margin: 15px 0; text-align: center;">
        <p style="color: #666; font-size: 9px; margin-bottom: 10px;">ADVERTISEMENT</p>
        <script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client={ADSENSE_CLIENT_ID}"
             crossorigin="anonymous"></script>
        <!-- {ad_name} -->
        <ins class="adsbygoogle"
             style="display:block"
             data-ad-client="{ADSENSE_CLIENT_ID}"
             data-ad-slot="{ad_slot}"
             data-ad-format="auto"
             data-full-width-responsive="true"></ins>
        <script>
             (adsbygoogle = window.adsbygoogle || []).push({{}});
        </script>
    </div>
    """
    components.html(ad_code, height=height)


def add_header_ad():
    """Affiche la pub header avec style par défaut"""
    import streamlit as st
    display_adsense("header", height=120)
    st.markdown('<hr style="border-color: #333; margin: 10px 0;">', unsafe_allow_html=True)


def add_footer_ad():
    """Affiche la pub footer avec style par défaut"""
    import streamlit as st
    st.markdown('<hr style="border-color: #333; margin: 10px 0;">', unsafe_allow_html=True)
    display_adsense("footer", height=150)
    st.markdown('<hr style="border-color: #333; margin: 10px 0;">', unsafe_allow_html=True)
