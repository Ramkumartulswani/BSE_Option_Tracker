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
# Google Ads Configuration
# -------------------------
ADSENSE_PUBLISHER_ID = "ca-pub-XXXXXXXXXXXXXXXX"  # REPLACE WITH YOUR PUBLISHER ID
ENABLE_ADS = False  # Set to True when you have AdSense approval

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SCRIP = "1"
API_BASE_URL = "https://api.bseindia.com/BseIndiaAPI/api/DerivOptionChain_IV/w"
CACHE_TTL = 30  # seconds
NEARBY_RANGE = 500  # points

# -------------------------
# Google Ads Rendering Functions
# -------------------------
def render_google_ad(ad_slot: str, ad_format: str = "auto", height: int = 250):
    """
    Render Google AdSense ad
    
    Args:
        ad_slot: Your ad slot ID from AdSense dashboard
        ad_format: Ad format (auto, horizontal, vertical, rectangle)
        height: Height of ad container in pixels
    """
    if not ENABLE_ADS:
        render_ad_placeholder()
        return
    
    ad_html = f"""
    <div style="text-align: center; padding: 10px; background-color: #f0f2f6; border-radius: 5px;">
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


def render_ad_placeholder(message: str = "📢 Advertisement Space"):
    """Render ad placeholder during development"""
    st.markdown(f"""
    <div style='padding: 20px; background: linear-gradient(135deg, #667eea22 0%, #764ba244 100%);
                border: 2px dashed #667eea; border-radius: 10px; text-align: center; 
                color: #667eea; margin: 10px 0;'>
        <h4 style='margin: 0;'>{message}</h4>
        <p style='margin: 5px 0; font-size: 12px;'>Replace ADSENSE_PUBLISHER_ID and set ENABLE_ADS = True</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_ad(ad_slot: str = "0000000000"):
    """Render vertical ad for sidebar"""
    if ENABLE_ADS:
        render_google_ad(ad_slot, ad_format="vertical", height=600)
    else:
        st.markdown("""
        <div style='padding: 15px; background: #f0f2f6; border-radius: 5px; text-align: center;'>
            <p style='margin: 0; font-size: 12px; color: #666;'>Ad Space (300x600)</p>
        </div>
        """, unsafe_allow_html=True)


def render_privacy_footer():
    """Render privacy policy and cookie consent (required for AdSense)"""
    with st.expander("📋 Privacy Policy & Cookie Notice"):
        st.markdown("""
        ### Privacy Policy
        
        This website uses Google AdSense to display advertisements. Google uses cookies to serve ads 
        based on your prior visits to this website or other websites.
        
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
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_expiry_dates(scrip_cd: str = "1") -> List[str]:
    """
    Fetch available expiry dates from BSE API
    
    Args:
        scrip_cd: BSE scrip code
        
    Returns:
        List of expiry dates in 'DD MMM YYYY' format
    """
    headers = {
        "accept": "application/json, text/plain, */*",
        "origin": "https://www.bseindia.com",
        "referer": "https://www.bseindia.com/",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        session = requests.Session()
        session.get("https://www.bseindia.com", headers=headers, timeout=5)
        
        # Try multiple possible endpoints for expiry dates
        urls_to_try = [
            f"https://api.bseindia.com/BseIndiaAPI/api/DDlExpiry/w?flag=0&scripcode={scrip_cd}",
            f"https://api.bseindia.com/BseIndiaAPI/api/DefaultData/w?scripcode={scrip_cd}",
            f"https://api.bseindia.com/BseIndiaAPI/api/DerivExpiryDates/w?scripcode={scrip_cd}",
        ]
        
        expiry_dates = []
        
        for url in urls_to_try:
            try:
                response = session.get(url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Try different possible key names for expiry dates
                    possible_keys = ['Table', 'expiry', 'expiryDate', 'ExpiryDates', 'Expiry', 
                                    'expDates', 'ExpiryList', 'expirylist', 'expiryDt']
                    
                    for key in possible_keys:
                        if key in data:
                            if isinstance(data[key], list):
                                # Extract expiry dates from list
                                for item in data[key]:
                                    if isinstance(item, dict):
                                        # Try different field names
                                        for date_key in ['expiry', 'Expiry', 'ExpiryDate', 'expiry_date', 
                                                        'Expiry_Date', 'expiryDt', 'ExpiryDt']:
                                            if date_key in item and item[date_key]:
                                                date_str = str(item[date_key]).strip()
                                                if date_str and date_str not in expiry_dates:
                                                    expiry_dates.append(date_str)
                                    elif isinstance(item, str) and item.strip():
                                        if item.strip() not in expiry_dates:
                                            expiry_dates.append(item.strip())
                            elif isinstance(data[key], str) and data[key].strip():
                                if data[key].strip() not in expiry_dates:
                                    expiry_dates.append(data[key].strip())
                    
                    if expiry_dates:
                        logger.info(f"Found {len(expiry_dates)} expiry dates from {url}")
                        break
                        
            except Exception as e:
                logger.debug(f"Failed to fetch from {url}: {str(e)}")
                continue
        
        # If API fetch fails, generate approximate monthly expiries as fallback
        if not expiry_dates:
            logger.warning("Could not fetch expiry dates from API, generating defaults")
            expiry_dates = generate_default_expiries()
        
        # Clean and deduplicate
        expiry_dates = list(set([exp.strip() for exp in expiry_dates if exp and exp.strip()]))
        
        # Sort dates
        try:
            expiry_dates.sort(key=lambda x: datetime.strptime(x, "%d %b %Y"))
        except:
            try:
                # Try alternative format
                expiry_dates.sort(key=lambda x: datetime.strptime(x, "%d-%b-%Y"))
            except:
                pass  # Keep original order if parsing fails
        
        logger.info(f"Returning {len(expiry_dates)} expiry dates")
        return expiry_dates
        
    except Exception as e:
        logger.error(f"Error fetching expiry dates: {str(e)}")
        return generate_default_expiries()


def generate_default_expiries() -> List[str]:
    """Generate next 6 monthly expiries (last Thursday of each month)"""
    current_date = datetime.now()
    expiry_dates = []
    
    for i in range(12):  # Generate 12 months worth
        # Calculate target month
        month = current_date.month + i
        year = current_date.year + (month - 1) // 12
        month = ((month - 1) % 12) + 1
        
        # Last day of month
        if month == 12:
            last_day = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            last_day = datetime(year, month + 1, 1) - timedelta(days=1)
        
        # Find last Thursday (weekday 3)
        while last_day.weekday() != 3:
            last_day -= timedelta(days=1)
        
        expiry_dates.append(last_day.strftime("%d %b %Y"))
    
    return expiry_dates


# -------------------------
# Utility Functions
# -------------------------
def safe_float_conversion(value, default=0.0) -> float:
    """Safely convert value to float with fallback"""
    try:
        return float(str(value).replace(",", "").strip())
    except (ValueError, TypeError, AttributeError):
        return default


def format_currency(value: float) -> str:
    """Format value as Indian currency"""
    return f"₹{value:,.2f}"


def validate_expiry_format(expiry: str) -> bool:
    """Validate expiry date format"""
    try:
        datetime.strptime(expiry, "%d %b %Y")
        return True
    except:
        try:
            datetime.strptime(expiry, "%d-%b-%Y")
            return True
        except:
            return False


# -------------------------
# Data Fetching
# -------------------------
@st.cache_data(ttl=CACHE_TTL)
def fetch_bse_option_chain(expiry: str, scrip_cd: str, strprice: str = "0") -> Tuple[Optional[pd.DataFrame], Optional[float], Optional[str], Optional[float], Optional[float]]:
    """
    Fetch option chain data from BSE API with robust error handling
    
    Returns:
        tuple: (dataframe, spot_price, error_message, day_high, day_low)
    """
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
            logger.error(f"HTTP Error: {response.status_code}")
            return None, None, f"❌ HTTP Error: {response.status_code}", None, None
        
        data = response.json()
        table = data.get("Table", [])
        
        if not table:
            return None, None, "⚠️ No data found for given expiry/instrument.", None, None
        
        df = _process_option_chain_df(table)
        spot_price = _extract_spot_price(data, table, df)
        day_high = _extract_market_value(data, ["High", "high", "DayHigh", "dayHigh"])
        day_low = _extract_market_value(data, ["Low", "low", "DayLow", "dayLow"])
        
        logger.info(f"Successfully fetched data: {len(df)} strikes")
        return df, spot_price, None, day_high, day_low
        
    except requests.exceptions.Timeout:
        return None, None, "⏱️ Request timeout. Please try again.", None, None
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error: {str(e)}")
        return None, None, f"🔌 Network error: {str(e)}", None, None
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return None, None, f"⚠️ Error: {str(e)}", None, None


def _process_option_chain_df(table: list) -> pd.DataFrame:
    """Process and clean option chain dataframe"""
    df = pd.DataFrame(table)
    
    df = df.rename(columns={
        "Strike_Price1": "Strike Price",
        "Open_Interest": "PE OI",
        "C_Open_Interest": "CE OI",
        "Vol_Traded": "PE Volume",
        "C_Vol_Traded": "CE Volume",
        "Last_Trd_Price": "PE LTP",
        "C_Last_Trd_Price": "CE LTP",
        "IV": "PE IV",
        "C_IV": "CE IV",
    })
    
    cols = ["Strike Price", "CE OI", "CE LTP", "CE Volume", "CE IV",
            "PE OI", "PE LTP", "PE Volume", "PE IV"]
    df = df[cols]
    
    for col in cols:
        df[col] = df[col].astype(str).str.replace(",", "").replace(["", " ", "None"], "0")
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).round(2)
    
    df = df[(df["CE OI"] > 0) | (df["PE OI"] > 0)]
    df = df.sort_values(by="Strike Price").reset_index(drop=True)
    
    return df


def _extract_spot_price(data: dict, table: list, df: pd.DataFrame) -> float:
    """Extract spot price from API response"""
    spot_keys = ["UlaValue", "UnderlyingValue", "underlyingValue", "Underlying_Value", 
                 "spotPrice", "SpotPrice", "IndexValue", "indexValue"]
    
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
    """Extract market value (high/low) from API response"""
    for key in keys:
        if key in data and data[key]:
            value = safe_float_conversion(data[key])
            if value > 0:
                return value
    return None


# -------------------------
# Analysis Functions (Same as before - keeping for brevity)
# -------------------------
class OptionAnalyzer:
    @staticmethod
    def calculate_pcr_analysis(df: pd.DataFrame) -> Dict:
        total_call_oi = df["CE OI"].sum()
        total_put_oi = df["PE OI"].sum()
        total_call_vol = df["CE Volume"].sum()
        total_put_vol = df["PE Volume"].sum()
        
        pcr_oi = round(total_put_oi / total_call_oi, 2) if total_call_oi > 0 else 0
        pcr_vol = round(total_put_vol / total_call_vol, 2) if total_call_vol > 0 else 0
        
        if pcr_oi > 1.2:
            sentiment = "📈 Bullish"
            description = "Strong Put Writing (Support Building)"
            color = "green"
        elif pcr_oi < 0.8:
            sentiment = "📉 Bearish"
            description = "Strong Call Writing (Resistance Building)"
            color = "red"
        else:
            sentiment = "⚖️ Neutral"
            description = "Balanced Market Conditions"
            color = "orange"
        
        return {
            "pcr_oi": pcr_oi,
            "pcr_vol": pcr_vol,
            "sentiment": sentiment,
            "description": description,
            "color": color,
            "total_call_oi": total_call_oi,
            "total_put_oi": total_put_oi,
            "total_call_vol": total_call_vol,
            "total_put_vol": total_put_vol
        }
    
    @staticmethod
    def find_support_resistance(df: pd.DataFrame, spot_price: float, num_levels: int = 5) -> Tuple:
        support_df = df[df["Strike Price"] <= spot_price].nlargest(num_levels, "PE OI")
        supports = support_df[["Strike Price", "PE OI", "PE LTP"]].copy()
        
        resistance_df = df[df["Strike Price"] >= spot_price].nlargest(num_levels, "CE OI")
        resistances = resistance_df[["Strike Price", "CE OI", "CE LTP"]].copy()
        
        nearest_support = support_df["Strike Price"].max() if not support_df.empty else None
        nearest_resistance = resistance_df["Strike Price"].min() if not resistance_df.empty else None
        
        return supports, resistances, nearest_support, nearest_resistance
    
    @staticmethod
    def analyze_max_pain(df: pd.DataFrame) -> Tuple[Optional[float], pd.DataFrame]:
        strikes = df["Strike Price"].unique()
        pain_values = []
        
        for strike in strikes:
            call_pain = ((df[df["Strike Price"] > strike]["CE OI"] * 
                         (df[df["Strike Price"] > strike]["Strike Price"] - strike)).sum())
            put_pain = ((df[df["Strike Price"] < strike]["PE OI"] * 
                        (strike - df[df["Strike Price"] < strike]["Strike Price"])).sum())
            total_pain = call_pain + put_pain
            pain_values.append({"Strike": strike, "Pain": total_pain})
        
        pain_df = pd.DataFrame(pain_values)
        max_pain_strike = pain_df.loc[pain_df["Pain"].idxmin(), "Strike"] if not pain_df.empty else None
        
        return max_pain_strike, pain_df
    
    @staticmethod
    def get_nearby_strikes(df: pd.DataFrame, spot_price: float, range_points: int = NEARBY_RANGE) -> pd.DataFrame:
        nearby = df[
            (df["Strike Price"] >= spot_price - range_points) & 
            (df["Strike Price"] <= spot_price + range_points)
        ].copy()
        
        nearby["OI Diff"] = nearby["PE OI"] - nearby["CE OI"]
        nearby["OI Ratio"] = nearby.apply(
            lambda row: round(row["PE OI"] / row["CE OI"], 2) if row["CE OI"] > 0 else 0,
            axis=1
        )
        
        median_ce_oi = nearby["CE OI"].median()
        median_pe_oi = nearby["PE OI"].median()
        
        nearby["CE Signal"] = nearby["CE OI"].apply(lambda x: "🔴" if x > median_ce_oi else "")
        nearby["PE Signal"] = nearby["PE OI"].apply(lambda x: "🟢" if x > median_pe_oi else "")
        
        return nearby.sort_values("Strike Price")
    
    @staticmethod
    def generate_trading_signals(df: pd.DataFrame, spot_price: float, pcr_data: Dict, 
                                 nearest_support: Optional[float], nearest_resistance: Optional[float]) -> Dict:
        pcr = pcr_data["pcr_oi"]
        
        signals = {
            "call_buy": [],
            "put_buy": [],
            "call_sell": [],
            "put_sell": [],
            "market_bias": "",
            "strategy": ""
        }
        
        if pcr > 1.3:
            signals["market_bias"] = "Strongly Bullish"
            signals["strategy"] = "Buy Calls or Sell Puts"
        elif pcr > 1.0:
            signals["market_bias"] = "Moderately Bullish"
            signals["strategy"] = "Buy ATM/OTM Calls"
        elif pcr < 0.7:
            signals["market_bias"] = "Strongly Bearish"
            signals["strategy"] = "Buy Puts or Sell Calls"
        elif pcr < 0.9:
            signals["market_bias"] = "Moderately Bearish"
            signals["strategy"] = "Buy ATM/OTM Puts"
        else:
            signals["market_bias"] = "Neutral/Rangebound"
            signals["strategy"] = "Iron Condor or Straddle"
        
        atm_idx = (df['Strike Price'] - spot_price).abs().argsort()[0]
        atm_strike = df.iloc[atm_idx]['Strike Price']
        
        if pcr >= 1.0:
            otm_calls = df[df['Strike Price'] > spot_price].nsmallest(2, 'Strike Price')
            
            signals["call_buy"].append({
                "strike": atm_strike,
                "type": "ATM Call",
                "target": nearest_resistance if nearest_resistance else spot_price + 500,
                "stop_loss": nearest_support if nearest_support else spot_price - 200,
                "reason": "ATM call for bullish move"
            })
            
            if not otm_calls.empty:
                signals["call_buy"].append({
                    "strike": otm_calls.iloc[0]['Strike Price'],
                    "type": "OTM Call",
                    "target": nearest_resistance if nearest_resistance else spot_price + 700,
                    "stop_loss": spot_price - 100,
                    "reason": "OTM call for aggressive bullish trade"
                })
        
        if pcr <= 0.9:
            otm_puts = df[df['Strike Price'] < spot_price].nlargest(2, 'Strike Price')
            
            signals["put_buy"].append({
                "strike": atm_strike,
                "type": "ATM Put",
                "target": nearest_support if nearest_support else spot_price - 500,
                "stop_loss": nearest_resistance if nearest_resistance else spot_price + 200,
                "reason": "ATM put for bearish move"
            })
            
            if not otm_puts.empty:
                signals["put_buy"].append({
                    "strike": otm_puts.iloc[0]['Strike Price'],
                    "type": "OTM Put",
                    "target": nearest_support if nearest_support else spot_price - 700,
                    "stop_loss": spot_price + 100,
                    "reason": "OTM put for aggressive bearish trade"
                })
        
        if pcr >= 1.2:
            strong_supports = df[df["Strike Price"] < spot_price].nlargest(3, "PE OI")
            if not strong_supports.empty:
                strong_support = strong_supports.iloc[0]['Strike Price']
                signals["put_sell"].append({
                    "strike": strong_support,
                    "type": "OTM Put Sell",
                    "target": "Premium collection",
                    "stop_loss": strong_support - 200,
                    "reason": f"Strong support at {strong_support:,.0f} with high PE OI"
                })
        
        if pcr <= 0.8:
            strong_resistances = df[df["Strike Price"] > spot_price].nlargest(3, "CE OI")
            if not strong_resistances.empty:
                strong_resistance = strong_resistances.iloc[0]['Strike Price']
                signals["call_sell"].append({
                    "strike": strong_resistance,
                    "type": "OTM Call Sell",
                    "target": "Premium collection",
                    "stop_loss": strong_resistance + 200,
                    "reason": f"Strong resistance at {strong_resistance:,.0f} with high CE OI"
                })
        
        return signals


# -------------------------
# Chart Generator (keeping abbreviated)
# -------------------------
class ChartGenerator:
    @staticmethod
    def create_oi_chart(df: pd.DataFrame, spot_price: float) -> go.Figure:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Open Interest Distribution", "Volume Distribution"),
            vertical_spacing=0.12,
            row_heights=[0.6, 0.4]
        )
        
        fig.add_trace(
            go.Bar(name="Call OI", x=df["Strike Price"], y=df["CE OI"], 
                   marker_color='rgba(255, 99, 71, 0.7)'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name="Put OI", x=df["Strike Price"], y=df["PE OI"], 
                   marker_color='rgba(60, 179, 113, 0.7)'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(name="Call Vol", x=df["Strike Price"], y=df["CE Volume"], 
                   marker_color='rgba(255, 140, 0, 0.7)', showlegend=False),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(name="Put Vol", x=df["Strike Price"], y=df["PE Volume"], 
                   marker_color='rgba(30, 144, 255, 0.7)', showlegend=False),
            row=2, col=1
        )
        
        fig.add_vline(x=spot_price, line_dash="dash", line_color="yellow", 
                      annotation_text=f"Spot: {spot_price:.0f}", row=1, col=1)
        fig.add_vline(x=spot_price, line_dash="dash", line_color="yellow", row=2, col=1)
        
        fig.update_layout(height=600, showlegend=True, hovermode='x unified',
                         barmode='group', template='plotly_dark')
        
        return fig
    
    @staticmethod
    def create_iv_chart(df: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=df["Strike Price"], y=df["CE IV"],
            mode='lines+markers', name='Call IV', line=dict(color='red', width=2)))
        
        fig.add_trace(go.Scatter(x=df["Strike Price"], y=df["PE IV"],
            mode='lines+markers', name='Put IV', line=dict(color='green', width=2)))
        
        fig.update_layout(title="Implied Volatility Smile",
            xaxis_title="Strike Price", yaxis_title="Implied Volatility (%)",
            height=400, hovermode='x unified', template='plotly_dark')
        
        return fig
    
    @staticmethod
    def create_pain_chart(pain_df: pd.DataFrame, spot_price: float) -> go.Figure:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=pain_df["Strike"], y=pain_df["Pain"],
            mode='lines', fill='tozeroy', name='Total Pain',
            line=dict(color='purple', width=2)))
        
        fig.add_vline(x=spot_price, line_dash="dash", 
                      annotation_text="Spot", line_color="yellow")
        
        fig.update_layout(title="Max Pain Distribution",
            xaxis_title="Strike Price", yaxis_title="Total Pain Value",
            height=300, template='plotly_dark')
        
        return fig


# -------------------------
# UI Components
# -------------------------
def render_day_range(spot_price: float, day_high: Optional[float], day_low: Optional[float]):
    if not (day_high and day_low):
        return
    
    day_range = day_high - day_low
    range_pct = (day_range / day_low) * 100
    position_in_range = ((spot_price - day_low) / day_range) * 100 if day_range > 0 else 50
    
    st.markdown(f"""
    <div style='padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 10px; margin: 15px 0; color: white;'>
        <h3 style='margin: 0 0 15px 0; text-align: center;'>📊 Index Day Range</h3>
        <div style='display: flex; justify-content: space-around; text-align: center;'>
            <div>
                <div style='font-size: 14px; opacity: 0.9;'>Day Low</div>
                <div style='font-size: 24px; font-weight: bold;'>{format_currency(day_low)}</div>
            </div>
            <div>
                <div style='font-size: 14px; opacity: 0.9;'>Current (Spot)</div>
                <div style='font-size: 28px; font-weight: bold; color: #FFD700;'>{format_currency(spot_price)}</div>
                <div style='font-size: 12px; margin-top: 5px;'>{position_in_range:.1f}% in range</div>
            </div>
            <div>
                <div style='font-size: 14px; opacity: 0.9;'>Day High</div>
                <div style='font-size: 24px; font-weight: bold;'>{format_currency(day_high)}</div>
            </div>
        </div>
        <div style='margin-top: 20px; background: rgba(255,255,255,0.2); border-radius: 10px; padding: 3px;'>
            <div style='width: {position_in_range}%; background: linear-gradient(90deg, #00ff00, #ffd700, #ff0000); 
                        height: 25px; border-radius: 8px; transition: width 0.3s;'></div>
        </div>
        <div style='display: flex; justify-content: space-between; margin-top: 10px; font-size: 14px;'>
            <div>Range: {format_currency(day_range)}</div>
            <div>Movement: {range_pct:.2f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_trading_signal(signal: Dict, signal_type: str):
    colors = {
        "call_buy": ("green", "🟢"),
        "put_buy": ("red", "🔴"),
        "put_sell": ("darkgreen", "💰"),
        "call_sell": ("darkred", "💰")
    }
    
    color, icon = colors.get(signal_type, ("gray", "•"))
    
    st.markdown(f"""
    <div style='padding: 15px; background-color: rgba(0, 255, 0, 0.1); border-left: 4px solid {color}; 
                border-radius: 5px; margin: 10px 0;'>
        <h4 style='margin: 0 0 10px 0; color: {color};'>{icon} {signal['type']} - Strike: {format_currency(signal['strike'])}</h4>
        <p style='margin: 5px 0;'><strong>Target:</strong> {format_currency(signal['target']) if isinstance(signal['target'], (int, float)) else signal['target']}</p>
        <p style='margin: 5px 0;'><strong>Stop Loss (Spot):</strong> {format_currency(signal['stop_loss'])}</p>
        <p style='margin: 5px 0;'><strong>Reason:</strong> {signal['reason']}</p>
    </div>
    """, unsafe_allow_html=True)


# -------------------------
# Main Dashboard
# -------------------------
def main():
    st.title("📈 BSE Option Chain Live Dashboard")
    st.markdown("**Real-time Support, Resistance, and Advanced Market Analytics**")
    
    # ===== TOP BANNER AD =====
    if ENABLE_ADS or True:  # Show placeholder even if ads disabled
        with st.container():
            st.caption("Advertisement")
            render_google_ad(ad_slot="1234567890", ad_format="horizontal", height=90)
            st.divider()
    
    analyzer = OptionAnalyzer()
    chart_gen = ChartGenerator()
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Scrip Code
        scrip_cd = st.text_input("Scrip Code", DEFAULT_SCRIP, help="BSE scrip code")
        
        st.divider()
        
        # ===== SIDEBAR AD =====
        st.caption("📢 Sponsored")
        render_sidebar_ad(ad_slot="9999999999")
        st.divider()
        
        # ===== HYBRID EXPIRY SELECTION =====
        st.subheader("📅 Expiry Date Selection")
        
        # Radio button to choose input method
        expiry_input_method = st.radio(
            "Choose input method:",
            ["Dropdown (Auto-loaded)", "Manual Entry"],
            help="Use dropdown for common dates or manual entry for any date"
        )
        
        if expiry_input_method == "Dropdown (Auto-loaded)":
            # Fetch and display dropdown
            with st.spinner("🔄 Loading expiry dates..."):
                expiry_dates = fetch_expiry_dates(scrip_cd)
            
            if expiry_dates:
                expiry = st.selectbox(
                    "Select Expiry Date",
                    options=expiry_dates,
                    index=0,
                    help="Select from available expiry dates"
                )
                st.success(f"✅ {len(expiry_dates)} dates loaded")
            else:
                st.error("Failed to load expiry dates")
                expiry = st.text_input(
                    "Enter Expiry Date",
                    "13 Nov 2025",
                    help="Format: DD MMM YYYY (e.g., 13 Nov 2025)"
                )
        else:
            # Manual entry
            expiry = st.text_input(
                "Enter Expiry Date",
                "13 Nov 2025",
                help="Format: DD MMM YYYY (e.g., 13 Nov 2025)"
            )
            
            # Validate format
            if validate_expiry_format(expiry):
                st.success("✅ Valid date format")
            else:
                st.warning("⚠️ Format should be: DD MMM YYYY (e.g., 13 Nov 2025)")
        
        st.info("💡 **Tip:** Switch to manual entry if your desired expiry date is not in the dropdown")
        
        st.divider()
        
        # Manual Spot Price Override
        manual_spot = st.checkbox("Override Spot Price", value=False)
        custom_spot = st.number_input("Enter Spot Price", min_value=0.0, value=50000.0, 
                                      step=100.0, disabled=not manual_spot)
        
        st.divider()
        
        # Auto Refresh
        auto_refresh = st.checkbox("Auto Refresh", value=False)
        refresh_rate = st.slider("Refresh Interval (sec)", 10, 120, 30, disabled=not auto_refresh)
        
        st.divider()
        
        # Advanced Settings
        show_advanced = st.checkbox("Show Advanced Analytics", value=True)
        num_levels = st.slider("Support/Resistance Levels", 3, 10, 5)
        
        st.divider()
        st.caption(f"🕐 Last Updated: {datetime.now().strftime('%H:%M:%S')}")
        
        # Refresh Button
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    # Fetch Data
    with st.spinner("🔄 Fetching option chain data..."):
        df, spot_price, error, day_high, day_low = fetch_bse_option_chain(expiry, scrip_cd)
    
    if error:
        st.error(error)
        st.info("""
        💡 **Troubleshooting Tips:**
        - Check your internet connection
        - Verify the expiry date is valid and has trading data
        - Ensure scrip code is correct
        - Try switching between dropdown and manual entry
        - Some expiry dates may not have data available yet
        """)
        st.stop()
    
    if df is None or df.empty:
        st.warning(f"⚠️ No data available for expiry: {expiry}")
        st.info("Try a different expiry date or check if this expiry has active options trading.")
        st.stop()
    
    # Use custom spot price if provided
    if manual_spot and custom_spot:
        original_spot = spot_price
        spot_price = custom_spot
        st.info(f"ℹ️ Using manually set spot price: {format_currency(spot_price)} (API: {format_currency(original_spot)})")
    
    # Calculate Analysis
    pcr_data = analyzer.calculate_pcr_analysis(df)
    supports, resistances, nearest_support, nearest_resistance = analyzer.find_support_resistance(
        df, spot_price, num_levels
    )
    trading_signals = analyzer.generate_trading_signals(
        df, spot_price, pcr_data, nearest_support, nearest_resistance
    )
    
    # Success message
    st.success(f"✅ Data loaded successfully for {expiry} | {len(df)} active strikes")
    
    # Day Range Display
    render_day_range(spot_price, day_high, day_low)
    
    # Key Metrics Row
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("💰 Spot Price", format_currency(spot_price))
    
    with col2:
        if day_high:
            delta_high = day_high - spot_price
            st.metric("📈 Day High", format_currency(day_high), 
                     f"-{delta_high:.2f}" if delta_high > 0 else "At High")
        else:
            st.metric("📈 Day High", "N/A")
    
    with col3:
        if day_low:
            delta_low = spot_price - day_low
            st.metric("📉 Day Low", format_currency(day_low), 
                     f"+{delta_low:.2f}" if delta_low > 0 else "At Low")
        else:
            st.metric("📉 Day Low", "N/A")
    
    with col4:
        st.metric("📊 PCR (OI)", pcr_data["pcr_oi"])
    
    with col5:
        if nearest_support:
            delta_support = spot_price - nearest_support
            st.metric("🟢 Support", format_currency(nearest_support), f"-{delta_support:.0f}")
        else:
            st.metric("🟢 Support", "N/A")
    
    with col6:
        if nearest_resistance:
            delta_resistance = nearest_resistance - spot_price
            st.metric("🔴 Resistance", format_currency(nearest_resistance), f"+{delta_resistance:.0f}")
        else:
            st.metric("🔴 Resistance", "N/A")
    
    # Market Sentiment
    st.markdown(f"""
    <div style='padding: 15px; background-color: {pcr_data['color']}22; border-left: 5px solid {pcr_data['color']}; 
                border-radius: 5px; margin: 20px 0;'>
        <h3 style='color: {pcr_data['color']}; margin: 0;'>{pcr_data['sentiment']}</h3>
        <p style='margin: 5px 0 0 0;'>{pcr_data['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Trading Signals
    st.header("🎯 Trading Signals")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Call Buy", "📉 Put Buy", "💰 Put Sell", "💰 Call Sell"])
    
    with tab1:
        if trading_signals["call_buy"]:
            for signal in trading_signals["call_buy"]:
                render_trading_signal(signal, "call_buy")
        else:
            st.warning("No call buying opportunities")
    
    with tab2:
        if trading_signals["put_buy"]:
            for signal in trading_signals["put_buy"]:
                render_trading_signal(signal, "put_buy")
        else:
            st.warning("No put buying opportunities")
    
    with tab3:
        if trading_signals["put_sell"]:
            for signal in trading_signals["put_sell"]:
                render_trading_signal(signal, "put_sell")
        else:
            st.info("No put selling opportunities")
    
    with tab4:
        if trading_signals["call_sell"]:
            for signal in trading_signals["call_sell"]:
                render_trading_signal(signal, "call_sell")
        else:
            st.info("No call selling opportunities")
    
    st.divider()
    
    # ===== MIDDLE AD (Between sections) =====
    with st.container():
        st.caption("Advertisement")
        render_google_ad(ad_slot="2222222222", ad_format="horizontal", height=90)
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
        
        tab1, tab2, tab3 = st.tabs(["Max Pain", "IV Smile", "Full Chain"])
        
        with tab1:
            max_pain, pain_df = analyzer.analyze_max_pain(df)
            
            if max_pain:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("🎯 Max Pain", format_currency(max_pain))
                with col2:
                    if not pain_df.empty:
                        st.plotly_chart(chart_gen.create_pain_chart(pain_df, spot_price), 
                                       use_container_width=True)
        
        with tab2:
            iv_chart = chart_gen.create_iv_chart(df)
            st.plotly_chart(iv_chart, use_container_width=True)
        
        with tab3:
            st.dataframe(df, hide_index=True, use_container_width=True, height=400)
    
    # Footer
    st.divider()
    
    # ===== BOTTOM AD =====
    with st.container():
        st.caption("Advertisement")
        render_google_ad(ad_slot="3333333333", ad_format="horizontal", height=90)
        st.divider()
    
    # Privacy Policy & Disclaimer
    col1, col2 = st.columns(2)
    with col1:
        st.caption("⚠️ **Disclaimer:** Educational purposes only. Trading involves risk. Consult a financial advisor before making trading decisions.")
    with col2:
        render_privacy_footer()
    
    # Auto Refresh
    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()


if __name__ == "__main__":
    main()
