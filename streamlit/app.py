import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple, Dict, Optional, List
import logging
import streamlit.components.v1 as components

# -------------------------
# Configuration
# -------------------------
st.set_page_config(
    page_title="📊 BSE Option Chain Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Google Ads Configuration - YOUR ACTUAL ADSENSE ID
# -------------------------
ADSENSE_PUBLISHER_ID = "ca-pub-5140463358652561"  # ✅ Your actual AdSense ID
ENABLE_ADS = True  # ✅ Ads enabled

# Ad Slot IDs - You can create specific ad units in AdSense and update these
AD_SLOTS = {
    'top': "0000000001",      # Top banner - Update with your ad slot ID
    'sidebar': "0000000002",   # Sidebar ad - Update with your ad slot ID
    'middle': "0000000003",    # Middle ad - Update with your ad slot ID
    'bottom': "0000000004"     # Bottom ad - Update with your ad slot ID
}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SCRIP = "1"
API_BASE_URL = "https://api.bseindia.com/BseIndiaAPI/api/DerivOptionChain_IV/w"
CACHE_TTL = 30
NEARBY_RANGE = 500

# -------------------------
# Google Ads Rendering Functions
# -------------------------
def render_google_ad(ad_slot: str, ad_format: str = "auto", height: int = 250):
    """Render Google AdSense ad with YOUR publisher ID"""
    
    ad_html = f"""
    <div style="text-align: center; padding: 10px; background-color: #f8f9fa; border-radius: 5px; margin: 10px 0;">
        <script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client={ADSENSE_PUBLISHER_ID}"
             crossorigin="anonymous"></script>
        <ins class="adsbygoogle"
             style="display:block"
             data-ad-client="{ADSENSE_PUBLISHER_ID}"
             data-ad-slot="{ad_slot}"
             data-ad-format="{ad_format}"
             data-full-width-responsive="true"></ins>
        <script>
             (adsbygoogle = window.adsbygoogle || []).push({{}});
        </script>
    </div>
    """
    
    components.html(ad_html, height=height, scrolling=False)


def render_auto_ads():
    """Render Auto Ads (Google will automatically place ads)"""
    
    auto_ads_html = f"""
    <script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client={ADSENSE_PUBLISHER_ID}"
         crossorigin="anonymous"></script>
    """
    
    components.html(auto_ads_html, height=0)


def render_sidebar_ad(ad_slot: str):
    """Render vertical ad for sidebar"""
    render_google_ad(ad_slot, ad_format="vertical", height=600)


def render_privacy_footer():
    """Render privacy policy (required for AdSense)"""
    with st.expander("📋 Privacy Policy & Cookie Notice"):
        st.markdown(f"""
        ### Privacy Policy
        
        This website uses Google AdSense to display advertisements. Google uses cookies to serve ads 
        based on your prior visits to this website or other websites.
        
        **Publisher ID:** {ADSENSE_PUBLISHER_ID}
        
        **What are cookies?**
        Cookies are small text files stored on your device that help us provide a better experience.
        
        **How Google uses data:**
        - Google's use of advertising cookies enables it and its partners to serve ads based on your 
          visit to this site and/or other sites on the Internet.
        - You can opt out of personalized advertising by visiting [Ads Settings](https://www.google.com/settings/ads).
        
        **Third-party vendors:**
        - Third-party vendors, including Google, use cookies to serve ads based on your prior visits.
        - Google's use of the DoubleClick cookie enables it to serve ads based on visits to various sites.
        
        **Your choices:**
        - You can opt out of personalized advertising at [aboutads.info](http://www.aboutads.info/choices/).
        - You can also opt out of Google's personalized ads at [Google Ads Settings](https://www.google.com/settings/ads).
        
        ### Data Usage
        We use Google Analytics and AdSense to improve our service. Data collected includes:
        - Page views and navigation patterns
        - Geographic location (city/country level)
        - Device and browser information
        - No personally identifiable information is collected
        
        ### Contact
        For questions about our privacy policy, please contact us through the feedback form.
        
        *Last updated: {datetime.now().strftime("%B %d, %Y")}*
        """)


# -------------------------
# Fetch Available Expiry Dates
# -------------------------
@st.cache_data(ttl=300)
def fetch_expiry_dates(scrip_cd: str = "1") -> List[str]:
    """Fetch available expiry dates from BSE API"""
    headers = {
        "accept": "application/json, text/plain, */*",
        "origin": "https://www.bseindia.com",
        "referer": "https://www.bseindia.com/",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        session = requests.Session()
        session.get("https://www.bseindia.com", headers=headers, timeout=5)
        
        urls_to_try = [
            f"https://api.bseindia.com/BseIndiaAPI/api/DDlExpiry/w?flag=0&scripcode={scrip_cd}",
            f"https://api.bseindia.com/BseIndiaAPI/api/DefaultData/w?scripcode={scrip_cd}",
        ]
        
        expiry_dates = []
        
        for url in urls_to_try:
            try:
                response = session.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    possible_keys = ['Table', 'expiry', 'expiryDate', 'ExpiryDates', 'Expiry']
                    
                    for key in possible_keys:
                        if key in data:
                            if isinstance(data[key], list):
                                for item in data[key]:
                                    if isinstance(item, dict):
                                        for date_key in ['expiry', 'Expiry', 'ExpiryDate']:
                                            if date_key in item and item[date_key]:
                                                date_str = str(item[date_key]).strip()
                                                if date_str and date_str not in expiry_dates:
                                                    expiry_dates.append(date_str)
                    if expiry_dates:
                        break
            except:
                continue
        
        if not expiry_dates:
            expiry_dates = generate_default_expiries()
        
        expiry_dates = list(set([exp.strip() for exp in expiry_dates if exp]))
        try:
            expiry_dates.sort(key=lambda x: datetime.strptime(x, "%d %b %Y"))
        except:
            pass
        
        return expiry_dates
        
    except Exception as e:
        return generate_default_expiries()


def generate_default_expiries() -> List[str]:
    """Generate default monthly expiries"""
    current_date = datetime.now()
    expiry_dates = []
    
    for i in range(12):
        month = current_date.month + i
        year = current_date.year + (month - 1) // 12
        month = ((month - 1) % 12) + 1
        
        if month == 12:
            last_day = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            last_day = datetime(year, month + 1, 1) - timedelta(days=1)
        
        while last_day.weekday() != 3:
            last_day -= timedelta(days=1)
        
        expiry_dates.append(last_day.strftime("%d %b %Y"))
    
    return expiry_dates


def safe_float_conversion(value, default=0.0) -> float:
    """Safely convert value to float"""
    try:
        return float(str(value).replace(",", "").strip())
    except:
        return default


def format_currency(value: float) -> str:
    """Format as Indian currency"""
    return f"₹{value:,.2f}"


def validate_expiry_format(expiry: str) -> bool:
    """Validate expiry date format"""
    try:
        datetime.strptime(expiry, "%d %b %Y")
        return True
    except:
        return False


# -------------------------
# Data Fetching
# -------------------------
@st.cache_data(ttl=CACHE_TTL)
def fetch_bse_option_chain(expiry: str, scrip_cd: str, strprice: str = "0") -> Tuple:
    """Fetch option chain data from BSE API"""
    url = f"{API_BASE_URL}?Expiry={expiry.replace(' ', '+')}&scrip_cd={scrip_cd}&strprice={strprice}"
    
    headers = {
        "accept": "application/json, text/plain, */*",
        "origin": "https://www.bseindia.com",
        "referer": "https://www.bseindia.com/",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        session = requests.Session()
        session.get("https://www.bseindia.com", headers=headers, timeout=5)
        response = session.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return None, None, f"❌ HTTP Error: {response.status_code}", None, None
        
        data = response.json()
        table = data.get("Table", [])
        
        if not table:
            return None, None, "⚠️ No data found", None, None
        
        df = _process_option_chain_df(table)
        spot_price = _extract_spot_price(data, table, df)
        day_high = _extract_market_value(data, ["High", "high", "DayHigh"])
        day_low = _extract_market_value(data, ["Low", "low", "DayLow"])
        
        return df, spot_price, None, day_high, day_low
        
    except Exception as e:
        return None, None, f"⚠️ Error: {str(e)}", None, None


def _process_option_chain_df(table: list) -> pd.DataFrame:
    """Process option chain dataframe"""
    df = pd.DataFrame(table)
    
    df = df.rename(columns={
        "Strike_Price1": "Strike Price", "Open_Interest": "PE OI", "C_Open_Interest": "CE OI",
        "Vol_Traded": "PE Volume", "C_Vol_Traded": "CE Volume", "Last_Trd_Price": "PE LTP",
        "C_Last_Trd_Price": "CE LTP", "IV": "PE IV", "C_IV": "CE IV",
    })
    
    cols = ["Strike Price", "CE OI", "CE LTP", "CE Volume", "CE IV", "PE OI", "PE LTP", "PE Volume", "PE IV"]
    df = df[cols]
    
    for col in cols:
        df[col] = df[col].astype(str).str.replace(",", "").replace(["", " ", "None"], "0")
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).round(2)
    
    df = df[(df["CE OI"] > 0) | (df["PE OI"] > 0)]
    df = df.sort_values(by="Strike Price").reset_index(drop=True)
    
    return df


def _extract_spot_price(data: dict, table: list, df: pd.DataFrame) -> float:
    """Extract spot price"""
    spot_keys = ["UlaValue", "UnderlyingValue", "spotPrice", "SpotPrice", "IndexValue"]
    
    for key in spot_keys:
        if key in data and data[key]:
            spot = safe_float_conversion(data[key])
            if spot > 0:
                return spot
    
    if table:
        for key in spot_keys:
            if key in table[0]:
                spot = safe_float_conversion(table[0][key])
                if spot > 0:
                    return spot
    
    return df["Strike Price"].median()


def _extract_market_value(data: dict, keys: list) -> Optional[float]:
    """Extract market value"""
    for key in keys:
        if key in data and data[key]:
            value = safe_float_conversion(data[key])
            if value > 0:
                return value
    return None


# -------------------------
# Analysis Classes (abbreviated for brevity - same as before)
# -------------------------
class OptionAnalyzer:
    @staticmethod
    def calculate_pcr_analysis(df):
        total_call_oi = df["CE OI"].sum()
        total_put_oi = df["PE OI"].sum()
        pcr_oi = round(total_put_oi / total_call_oi, 2) if total_call_oi > 0 else 0
        
        if pcr_oi > 1.2:
            sentiment, desc, color = "📈 Bullish", "Strong Put Writing", "green"
        elif pcr_oi < 0.8:
            sentiment, desc, color = "📉 Bearish", "Strong Call Writing", "red"
        else:
            sentiment, desc, color = "⚖️ Neutral", "Balanced Market", "orange"
        
        return {
            "pcr_oi": pcr_oi, "sentiment": sentiment, "description": desc, "color": color,
            "total_call_oi": total_call_oi, "total_put_oi": total_put_oi,
            "total_call_vol": df["CE Volume"].sum(), "total_put_vol": df["PE Volume"].sum()
        }
    
    @staticmethod
    def find_support_resistance(df, spot_price, num_levels=5):
        support_df = df[df["Strike Price"] <= spot_price].nlargest(num_levels, "PE OI")
        resistance_df = df[df["Strike Price"] >= spot_price].nlargest(num_levels, "CE OI")
        
        return (
            support_df[["Strike Price", "PE OI", "PE LTP"]].copy(),
            resistance_df[["Strike Price", "CE OI", "CE LTP"]].copy(),
            support_df["Strike Price"].max() if not support_df.empty else None,
            resistance_df["Strike Price"].min() if not resistance_df.empty else None
        )


class ChartGenerator:
    @staticmethod
    def create_oi_chart(df, spot_price):
        fig = make_subplots(rows=2, cols=1, subplot_titles=("Open Interest", "Volume"), 
                           vertical_spacing=0.12, row_heights=[0.6, 0.4])
        
        fig.add_trace(go.Bar(name="Call OI", x=df["Strike Price"], y=df["CE OI"], 
                            marker_color='rgba(255,99,71,0.7)'), row=1, col=1)
        fig.add_trace(go.Bar(name="Put OI", x=df["Strike Price"], y=df["PE OI"], 
                            marker_color='rgba(60,179,113,0.7)'), row=1, col=1)
        fig.add_trace(go.Bar(name="Call Vol", x=df["Strike Price"], y=df["CE Volume"], 
                            marker_color='rgba(255,140,0,0.7)', showlegend=False), row=2, col=1)
        fig.add_trace(go.Bar(name="Put Vol", x=df["Strike Price"], y=df["PE Volume"], 
                            marker_color='rgba(30,144,255,0.7)', showlegend=False), row=2, col=1)
        
        fig.add_vline(x=spot_price, line_dash="dash", line_color="yellow", 
                     annotation_text=f"Spot: {spot_price:.0f}", row=1, col=1)
        fig.add_vline(x=spot_price, line_dash="dash", line_color="yellow", row=2, col=1)
        
        fig.update_layout(height=600, showlegend=True, barmode='group', template='plotly_dark')
        return fig


# -------------------------
# Main Dashboard
# -------------------------
def main():
    st.title("📈 BSE Option Chain Live Dashboard")
    st.markdown("**Real-time Support, Resistance, and Advanced Market Analytics**")
    
    # Load Auto Ads script in header
    render_auto_ads()
    
    # TOP BANNER AD
    with st.container():
        st.caption("📢 Advertisement")
        render_google_ad(ad_slot=AD_SLOTS['top'], ad_format="horizontal", height=90)
        st.divider()
    
    analyzer = OptionAnalyzer()
    chart_gen = ChartGenerator()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        scrip_cd = st.text_input("Scrip Code", DEFAULT_SCRIP)
        st.divider()
        
        # SIDEBAR AD
        st.caption("📢 Sponsored")
        render_sidebar_ad(ad_slot=AD_SLOTS['sidebar'])
        st.divider()
        
        # Expiry selection
        st.subheader("📅 Expiry Date")
        expiry_method = st.radio("Input method:", ["Dropdown", "Manual Entry"])
        
        if expiry_method == "Dropdown":
            with st.spinner("Loading expiries..."):
                expiry_dates = fetch_expiry_dates(scrip_cd)
            
            if expiry_dates:
                expiry = st.selectbox("Select Expiry", expiry_dates, index=0)
                st.success(f"✅ {len(expiry_dates)} dates loaded")
            else:
                expiry = st.text_input("Enter Expiry", "13 Nov 2025")
        else:
            expiry = st.text_input("Enter Expiry", "13 Nov 2025", help="Format: DD MMM YYYY")
            if validate_expiry_format(expiry):
                st.success("✅ Valid format")
            else:
                st.warning("⚠️ Use format: DD MMM YYYY")
        
        st.divider()
        manual_spot = st.checkbox("Override Spot Price")
        custom_spot = st.number_input("Spot Price", min_value=0.0, value=50000.0, 
                                     step=100.0, disabled=not manual_spot)
        st.divider()
        auto_refresh = st.checkbox("Auto Refresh")
        refresh_rate = st.slider("Refresh (sec)", 10, 120, 30, disabled=not auto_refresh)
        st.divider()
        show_advanced = st.checkbox("Show Advanced Analytics", value=True)
        num_levels = st.slider("Support/Resistance Levels", 3, 10, 5)
        st.divider()
        st.caption(f"🕐 Last Updated: {datetime.now().strftime('%H:%M:%S')}")
        
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    # Fetch Data
    with st.spinner("🔄 Fetching data..."):
        df, spot_price, error, day_high, day_low = fetch_bse_option_chain(expiry, scrip_cd)
    
    if error:
        st.error(error)
        st.stop()
    
    if df is None or df.empty:
        st.warning(f"⚠️ No data for {expiry}")
        st.stop()
    
    if manual_spot and custom_spot:
        spot_price = custom_spot
    
    # Analysis
    pcr_data = analyzer.calculate_pcr_analysis(df)
    supports, resistances, nearest_support, nearest_resistance = analyzer.find_support_resistance(df, spot_price, num_levels)
    
    st.success(f"✅ Data loaded: {expiry} | {len(df)} strikes")
    
    # Metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("💰 Spot", format_currency(spot_price))
    with col2:
        st.metric("📈 High", format_currency(day_high) if day_high else "N/A")
    with col3:
        st.metric("📉 Low", format_currency(day_low) if day_low else "N/A")
    with col4:
        st.metric("📊 PCR", pcr_data["pcr_oi"])
    with col5:
        st.metric("🟢 Support", format_currency(nearest_support) if nearest_support else "N/A")
    with col6:
        st.metric("🔴 Resistance", format_currency(nearest_resistance) if nearest_resistance else "N/A")
    
    # Sentiment
    st.markdown(f"""
    <div style='padding: 15px; background-color: {pcr_data['color']}22; 
                border-left: 5px solid {pcr_data['color']}; border-radius: 5px; margin: 20px 0;'>
        <h3 style='color: {pcr_data['color']}; margin: 0;'>{pcr_data['sentiment']}</h3>
        <p style='margin: 5px 0 0 0;'>{pcr_data['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # MIDDLE AD
    with st.container():
        st.caption("📢 Advertisement")
        render_google_ad(ad_slot=AD_SLOTS['middle'], ad_format="horizontal", height=90)
        st.divider()
    
    # Support & Resistance
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🟢 Support Levels")
        st.dataframe(supports.style.format({
            "Strike Price": lambda x: format_currency(x),
            "PE OI": lambda x: f"{x:,.0f}",
            "PE LTP": lambda x: format_currency(x)
        }), hide_index=True, use_container_width=True)
    
    with col2:
        st.subheader("🔴 Resistance Levels")
        st.dataframe(resistances.style.format({
            "Strike Price": lambda x: format_currency(x),
            "CE OI": lambda x: f"{x:,.0f}",
            "CE LTP": lambda x: format_currency(x)
        }), hide_index=True, use_container_width=True)
    
    st.divider()
    
    # Charts
    st.subheader("📊 OI & Volume Analysis")
    oi_chart = chart_gen.create_oi_chart(df, spot_price)
    st.plotly_chart(oi_chart, use_container_width=True)
    
    # Advanced Analytics
    if show_advanced:
        st.divider()
        st.subheader("🎯 Advanced Analytics")
        tab1, tab2 = st.tabs(["Full Chain", "Summary"])
        
        with tab1:
            st.dataframe(df, hide_index=True, use_container_width=True, height=400)
        
        with tab2:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Call OI", f"{pcr_data['total_call_oi']:,.0f}")
            with col2:
                st.metric("Total Put OI", f"{pcr_data['total_put_oi']:,.0f}")
            with col3:
                oi_diff = pcr_data['total_put_oi'] - pcr_data['total_call_oi']
                st.metric("Net OI", f"{oi_diff:,.0f}")
            with col4:
                st.metric("Active Strikes", len(df))
    
    # Footer
    st.divider()
    
    # BOTTOM AD
    with st.container():
        st.caption("📢 Advertisement")
        render_google_ad(ad_slot=AD_SLOTS['bottom'], ad_format="horizontal", height=90)
        st.divider()
    
    # Privacy & Disclaimer
    col1, col2 = st.columns(2)
    with col1:
        st.caption("⚠️ **Disclaimer:** Educational purposes only. Trading involves risk.")
    with col2:
        render_privacy_footer()
    
    # Auto Refresh
    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()


if __name__ == "__main__":
    main()
