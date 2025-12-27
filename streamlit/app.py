import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple, Dict, Optional
import logging

# -------------------------
# Configuration
# -------------------------
st.set_page_config(
    page_title="📊 BSE Option Chain Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_EXPIRY = "13 Nov 2025"
DEFAULT_SCRIP = "1"
API_BASE_URL = "https://api.bseindia.com/BseIndiaAPI/api/DerivOptionChain_IV/w"
CACHE_TTL = 30  # seconds
NEARBY_RANGE = 500  # points

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


def calculate_percentage_change(current: float, reference: float) -> float:
    """Calculate percentage change"""
    if reference == 0:
        return 0.0
    return ((current - reference) / reference) * 100


# -------------------------
# Data Fetching with Improved Error Handling
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
        # Create session and make request
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
        
        # Process DataFrame
        df = pd.DataFrame(table)
        df = _process_option_chain_df(df)
        
        # Extract market data
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


def _process_option_chain_df(df: pd.DataFrame) -> pd.DataFrame:
    """Process and clean option chain dataframe"""
    # Rename columns
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
    
    # Select required columns
    cols = ["Strike Price", "CE OI", "CE LTP", "CE Volume", "CE IV",
            "PE OI", "PE LTP", "PE Volume", "PE IV"]
    df = df[cols]
    
    # Clean and convert numeric data
    for col in cols:
        df[col] = df[col].astype(str).str.replace(",", "").replace(["", " ", "None"], "0")
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).round(2)
    
    # Filter out rows with no OI
    df = df[(df["CE OI"] > 0) | (df["PE OI"] > 0)]
    df = df.sort_values(by="Strike Price").reset_index(drop=True)
    
    return df


def _extract_spot_price(data: dict, table: list, df: pd.DataFrame) -> float:
    """Extract spot price from API response"""
    spot_keys = ["UlaValue", "UnderlyingValue", "underlyingValue", "Underlying_Value", 
                 "spotPrice", "SpotPrice", "IndexValue", "indexValue"]
    
    # Try from main data
    for key in spot_keys:
        if key in data and data[key]:
            spot = safe_float_conversion(data[key])
            if spot > 0:
                return spot
    
    # Try from table
    if table:
        for key in spot_keys:
            if key in table[0]:
                spot = safe_float_conversion(table[0][key])
                if spot > 0:
                    return spot
    
    # Fallback to median strike
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
# Analysis Functions (Improved)
# -------------------------
class OptionAnalyzer:
    """Class to encapsulate all option analysis functions"""
    
    @staticmethod
    def calculate_pcr_analysis(df: pd.DataFrame) -> Dict:
        """Calculate PCR and market sentiment"""
        total_call_oi = df["CE OI"].sum()
        total_put_oi = df["PE OI"].sum()
        total_call_vol = df["CE Volume"].sum()
        total_put_vol = df["PE Volume"].sum()
        
        pcr_oi = round(total_put_oi / total_call_oi, 2) if total_call_oi > 0 else 0
        pcr_vol = round(total_put_vol / total_call_vol, 2) if total_call_vol > 0 else 0
        
        # Determine sentiment
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
        """Identify key support and resistance levels"""
        # Support levels (high PE OI below spot)
        support_df = df[df["Strike Price"] <= spot_price].nlargest(num_levels, "PE OI")
        supports = support_df[["Strike Price", "PE OI", "PE LTP"]].copy()
        
        # Resistance levels (high CE OI above spot)
        resistance_df = df[df["Strike Price"] >= spot_price].nlargest(num_levels, "CE OI")
        resistances = resistance_df[["Strike Price", "CE OI", "CE LTP"]].copy()
        
        # Find nearest levels
        nearest_support = support_df["Strike Price"].max() if not support_df.empty else None
        nearest_resistance = resistance_df["Strike Price"].min() if not resistance_df.empty else None
        
        return supports, resistances, nearest_support, nearest_resistance
    
    @staticmethod
    def analyze_max_pain(df: pd.DataFrame) -> Tuple[Optional[float], pd.DataFrame]:
        """Calculate Max Pain strike price"""
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
        """Analyze OI changes near spot price"""
        nearby = df[
            (df["Strike Price"] >= spot_price - range_points) & 
            (df["Strike Price"] <= spot_price + range_points)
        ].copy()
        
        # Calculate metrics
        nearby["OI Diff"] = nearby["PE OI"] - nearby["CE OI"]
        nearby["OI Ratio"] = nearby.apply(
            lambda row: round(row["PE OI"] / row["CE OI"], 2) if row["CE OI"] > 0 else 0,
            axis=1
        )
        
        # Add signals
        median_ce_oi = nearby["CE OI"].median()
        median_pe_oi = nearby["PE OI"].median()
        
        nearby["CE Signal"] = nearby["CE OI"].apply(lambda x: "🔴" if x > median_ce_oi else "")
        nearby["PE Signal"] = nearby["PE OI"].apply(lambda x: "🟢" if x > median_pe_oi else "")
        
        return nearby.sort_values("Strike Price")
    
    @staticmethod
    def generate_trading_signals(df: pd.DataFrame, spot_price: float, pcr_data: Dict, 
                                 nearest_support: Optional[float], nearest_resistance: Optional[float]) -> Dict:
        """Generate actionable trading signals"""
        pcr = pcr_data["pcr_oi"]
        
        signals = {
            "call_buy": [],
            "put_buy": [],
            "call_sell": [],
            "put_sell": [],
            "market_bias": "",
            "strategy": ""
        }
        
        # Determine market bias
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
        
        # Get ATM strike
        atm_idx = (df['Strike Price'] - spot_price).abs().argsort()[0]
        atm_strike = df.iloc[atm_idx]['Strike Price']
        
        # Bullish signals
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
        
        # Bearish signals
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
        
        # Premium collection strategies
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
# Visualization Functions (Improved)
# -------------------------
class ChartGenerator:
    """Class to encapsulate all chart generation functions"""
    
    @staticmethod
    def create_oi_chart(df: pd.DataFrame, spot_price: float) -> go.Figure:
        """Create interactive OI distribution chart"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Open Interest Distribution", "Volume Distribution"),
            vertical_spacing=0.12,
            row_heights=[0.6, 0.4]
        )
        
        # OI Chart
        fig.add_trace(
            go.Bar(name="Call OI", x=df["Strike Price"], y=df["CE OI"], 
                   marker_color='rgba(255, 99, 71, 0.7)',
                   hovertemplate='Strike: %{x}<br>Call OI: %{y:,.0f}<extra></extra>'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name="Put OI", x=df["Strike Price"], y=df["PE OI"], 
                   marker_color='rgba(60, 179, 113, 0.7)',
                   hovertemplate='Strike: %{x}<br>Put OI: %{y:,.0f}<extra></extra>'),
            row=1, col=1
        )
        
        # Volume Chart
        fig.add_trace(
            go.Bar(name="Call Vol", x=df["Strike Price"], y=df["CE Volume"], 
                   marker_color='rgba(255, 140, 0, 0.7)', showlegend=False,
                   hovertemplate='Strike: %{x}<br>Call Vol: %{y:,.0f}<extra></extra>'),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(name="Put Vol", x=df["Strike Price"], y=df["PE Volume"], 
                   marker_color='rgba(30, 144, 255, 0.7)', showlegend=False,
                   hovertemplate='Strike: %{x}<br>Put Vol: %{y:,.0f}<extra></extra>'),
            row=2, col=1
        )
        
        # Add spot price line
        fig.add_vline(x=spot_price, line_dash="dash", line_color="yellow", 
                      annotation_text=f"Spot: {spot_price:.0f}", row=1, col=1)
        fig.add_vline(x=spot_price, line_dash="dash", line_color="yellow", row=2, col=1)
        
        fig.update_layout(
            height=600,
            showlegend=True,
            hovermode='x unified',
            barmode='group',
            template='plotly_dark'
        )
        
        return fig
    
    @staticmethod
    def create_iv_chart(df: pd.DataFrame) -> go.Figure:
        """Create Implied Volatility smile chart"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df["Strike Price"], y=df["CE IV"],
            mode='lines+markers', name='Call IV',
            line=dict(color='red', width=2),
            hovertemplate='Strike: %{x}<br>Call IV: %{y:.2f}%<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=df["Strike Price"], y=df["PE IV"],
            mode='lines+markers', name='Put IV',
            line=dict(color='green', width=2),
            hovertemplate='Strike: %{x}<br>Put IV: %{y:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title="Implied Volatility Smile",
            xaxis_title="Strike Price",
            yaxis_title="Implied Volatility (%)",
            height=400,
            hovermode='x unified',
            template='plotly_dark'
        )
        
        return fig
    
    @staticmethod
    def create_pain_chart(pain_df: pd.DataFrame, spot_price: float) -> go.Figure:
        """Create Max Pain distribution chart"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=pain_df["Strike"], y=pain_df["Pain"],
            mode='lines', fill='tozeroy', name='Total Pain',
            line=dict(color='purple', width=2),
            hovertemplate='Strike: %{x}<br>Pain: %{y:,.0f}<extra></extra>'
        ))
        
        fig.add_vline(x=spot_price, line_dash="dash", 
                      annotation_text="Spot", line_color="yellow")
        
        fig.update_layout(
            title="Max Pain Distribution",
            xaxis_title="Strike Price",
            yaxis_title="Total Pain Value",
            height=300,
            template='plotly_dark'
        )
        
        return fig


# -------------------------
# UI Components
# -------------------------
def render_day_range(spot_price: float, day_high: Optional[float], day_low: Optional[float]):
    """Render day range visualization"""
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
            <div>From Low: +{format_currency(spot_price - day_low)}</div>
            <div>To High: {format_currency(day_high - spot_price)}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_trading_signal(signal: Dict, signal_type: str):
    """Render a single trading signal"""
    colors = {
        "call_buy": ("green", "🟢"),
        "put_buy": ("red", "🔴"),
        "put_sell": ("darkgreen", "💰"),
        "call_sell": ("darkred", "💰")
    }
    
    color, icon = colors.get(signal_type, ("gray", "•"))
    
    st.markdown(f"""
    <div style='padding: 15px; background-color: rgba({color}, 0.1); border-left: 4px solid {color}; 
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
    
    # Initialize analyzer
    analyzer = OptionAnalyzer()
    chart_gen = ChartGenerator()
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        expiry = st.text_input("Expiry Date", DEFAULT_EXPIRY, help="Format: DD MMM YYYY")
        scrip_cd = st.text_input("Scrip Code", DEFAULT_SCRIP, help="BSE scrip code")
        
        st.divider()
        
        manual_spot = st.checkbox("Override Spot Price", value=False)
        custom_spot = st.number_input("Enter Spot Price", min_value=0.0, value=50000.0, 
                                      step=100.0, disabled=not manual_spot)
        
        st.divider()
        
        auto_refresh = st.checkbox("Auto Refresh", value=False)
        refresh_rate = st.slider("Refresh Interval (sec)", 10, 120, 30, disabled=not auto_refresh)
        
        st.divider()
        
        show_advanced = st.checkbox("Show Advanced Analytics", value=True)
        num_levels = st.slider("Support/Resistance Levels", 3, 10, 5)
        
        st.divider()
        st.caption(f"🕐 Last Updated: {datetime.now().strftime('%H:%M:%S')}")
        
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    # Fetch Data
    with st.spinner("🔄 Fetching option chain data..."):
        df, spot_price, error, day_high, day_low = fetch_bse_option_chain(expiry, scrip_cd)
    
    if error:
        st.error(error)
        st.info("💡 **Troubleshooting Tips:**\n- Check your internet connection\n- Verify expiry date format (DD MMM YYYY)\n- Ensure scrip code is correct")
        st.stop()
    
    if df is None or df.empty:
        st.warning("No data available. Please check your inputs.")
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
        st.metric("📊 PCR (OI)", pcr_data["pcr_oi"], help="Put-Call Ratio based on Open Interest")
    
    with col5:
        if nearest_support:
            delta_support = spot_price - nearest_support
            st.metric("🟢 Nearest Support", format_currency(nearest_support), f"-{delta_support:.0f}")
        else:
            st.metric("🟢 Nearest Support", "N/A")
    
    with col6:
        if nearest_resistance:
            delta_resistance = nearest_resistance - spot_price
            st.metric("🔴 Nearest Resistance", format_currency(nearest_resistance), f"+{delta_resistance:.0f}")
        else:
            st.metric("🔴 Nearest Resistance", "N/A")
    
    # Market Sentiment
    st.markdown(f"""
    <div style='padding: 15px; background-color: {pcr_data['color']}22; border-left: 5px solid {pcr_data['color']}; 
                border-radius: 5px; margin: 20px 0;'>
        <h3 style='color: {pcr_data['color']}; margin: 0;'>{pcr_data['sentiment']}</h3>
        <p style='margin: 5px 0 0 0;'>{pcr_data['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Trading Signals Section
    st.header("🎯 Trading Signals & Recommendations")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(f"""
        <div style='padding: 20px; background: linear-gradient(135deg, {pcr_data['color']}44, {pcr_data['color']}88); 
                    border-radius: 10px; text-align: center;'>
            <h3 style='margin: 0;'>Market Bias</h3>
            <h2 style='color: {pcr_data['color']}; margin: 10px 0;'>{trading_signals['market_bias']}</h2>
            <p style='margin: 0; font-size: 14px;'>{trading_signals['strategy']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.info("""
        **📌 How to Use These Signals:**
        - **Green (Bullish)**: Consider buying Calls or selling Puts
        - **Red (Bearish)**: Consider buying Puts or selling Calls
        - **Target**: Expected price level to book profit
        - **Stop Loss**: Exit level if trade goes against you
        - Always manage risk and use proper position sizing
        """)
    
    # Tabbed view for different trade types
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Call Buying", "📉 Put Buying", "💰 Put Selling", "💰 Call Selling"])
    
    with tab1:
        st.subheader("Call Buying Opportunities")
        if trading_signals["call_buy"]:
            for signal in trading_signals["call_buy"]:
                render_trading_signal(signal, "call_buy")
        else:
            st.warning("No Call buying opportunities. Market bias is not bullish enough.")
    
    with tab2:
        st.subheader("Put Buying Opportunities")
        if trading_signals["put_buy"]:
            for signal in trading_signals["put_buy"]:
                render_trading_signal(signal, "put_buy")
        else:
            st.warning("No Put buying opportunities. Market bias is not bearish enough.")
    
    with tab3:
        st.subheader("Put Selling (Credit) Opportunities")
        st.caption("⚠️ Advanced Strategy: Requires margin.")
        if trading_signals["put_sell"]:
            for signal in trading_signals["put_sell"]:
                render_trading_signal(signal, "put_sell")
        else:
            st.info("No Put selling opportunities. Market needs stronger bullish bias.")
    
    with tab4:
        st.subheader("Call Selling (Credit) Opportunities")
        st.caption("⚠️ Advanced Strategy: Requires margin.")
        if trading_signals["call_sell"]:
            for signal in trading_signals["call_sell"]:
                render_trading_signal(signal, "call_sell")
        else:
            st.info("No Call selling opportunities. Market needs stronger bearish bias.")
    
    st.divider()
    
    # Support & Resistance Tables
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🟢 Key Support Levels")
        
        def style_support(row):
            if nearest_support and abs(row['Strike Price'] - nearest_support) < 1:
                return ['background-color: rgba(0, 255, 0, 0.4); font-weight: bold'] * len(row)
            elif row['PE OI'] > supports['PE OI'].quantile(0.75):
                return ['background-color: rgba(0, 200, 0, 0.2)'] * len(row)
            return [''] * len(row)
        
        styled_supports = supports.style.apply(style_support, axis=1).format({
            "Strike Price": lambda x: format_currency(x),
            "PE OI": lambda x: f"{x:,.0f}",
            "PE LTP": lambda x: format_currency(x)
        })
        
        st.dataframe(styled_supports, hide_index=True, use_container_width=True)
        if nearest_support:
            st.success(f"**Nearest Support:** {format_currency(nearest_support)}")
    
    with col2:
        st.subheader("🔴 Key Resistance Levels")
        
        def style_resistance(row):
            if nearest_resistance and abs(row['Strike Price'] - nearest_resistance) < 1:
                return ['background-color: rgba(255, 0, 0, 0.4); font-weight: bold'] * len(row)
            elif row['CE OI'] > resistances['CE OI'].quantile(0.75):
                return ['background-color: rgba(255, 100, 100, 0.2)'] * len(row)
            return [''] * len(row)
        
        styled_resistances = resistances.style.apply(style_resistance, axis=1).format({
            "Strike Price": lambda x: format_currency(x),
            "CE OI": lambda x: f"{x:,.0f}",
            "CE LTP": lambda x: format_currency(x)
        })
        
        st.dataframe(styled_resistances, hide_index=True, use_container_width=True)
        if nearest_resistance:
            st.error(f"**Nearest Resistance:** {format_currency(nearest_resistance)}")
    
    st.divider()
    
    # Nearby Strike Activity
    st.subheader(f"⚡ Nearby Strike Activity (±{NEARBY_RANGE} Points)")
    
    try:
        nearby = analyzer.get_nearby_strikes(df, spot_price, NEARBY_RANGE)
        
        if not nearby.empty:
            display_cols = ["Strike Price", "CE Signal", "CE OI", "PE OI", "PE Signal", 
                           "OI Diff", "OI Ratio", "CE LTP", "PE LTP"]
            display_df = nearby[display_cols].copy()
            
            atm_strike = display_df.iloc[(display_df['Strike Price'] - spot_price).abs().argsort()[:1]]['Strike Price'].values[0]
            
            def highlight_row(row):
                styles = [''] * len(row)
                
                if abs(row['Strike Price'] - atm_strike) < 1:
                    styles = ['background-color: rgba(255, 215, 0, 0.3); font-weight: bold'] * len(row)
                
                oi_diff_idx = display_cols.index('OI Diff')
                if row['OI Diff'] > 0:
                    styles[oi_diff_idx] = 'background-color: rgba(0, 255, 0, 0.25); color: darkgreen; font-weight: bold'
                elif row['OI Diff'] < 0:
                    styles[oi_diff_idx] = 'background-color: rgba(255, 0, 0, 0.25); color: darkred; font-weight: bold'
                
                return styles
            
            styled_df = display_df.style.apply(highlight_row, axis=1).format({
                "Strike Price": lambda x: format_currency(x),
                "CE OI": lambda x: f"{x:,.0f}",
                "PE OI": lambda x: f"{x:,.0f}",
                "OI Diff": lambda x: f"{x:,.0f}",
                "OI Ratio": lambda x: f"{x:.2f}",
                "CE LTP": lambda x: format_currency(x),
                "PE LTP": lambda x: format_currency(x)
            })
            
            st.dataframe(styled_df, hide_index=True, use_container_width=True, height=400)
            
            # Legend
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption("🟡 **Yellow** = ATM Strike")
            with col2:
                st.caption("🟢 **Green** = Put Buildup (Bullish)")
            with col3:
                st.caption("🔴 **Red** = Call Buildup (Bearish)")
        else:
            st.info(f"No significant activity in ±{NEARBY_RANGE} points range.")
    except Exception as e:
        st.warning(f"Unable to display nearby activity: {str(e)}")
    
    st.divider()
    
    # Charts
    st.subheader("📊 Open Interest & Volume Analysis")
    oi_chart = chart_gen.create_oi_chart(df, spot_price)
    st.plotly_chart(oi_chart, use_container_width=True)
    
    # Advanced Analytics
    if show_advanced:
        st.divider()
        st.subheader("🎯 Advanced Analytics")
        
        tab1, tab2, tab3 = st.tabs(["Max Pain Analysis", "IV Smile", "Full Option Chain"])
        
        with tab1:
            max_pain, pain_df = analyzer.analyze_max_pain(df)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if max_pain:
                    st.metric("🎯 Max Pain Strike", format_currency(max_pain))
                    pain_distance = abs(spot_price - max_pain)
                    pain_direction = "above" if spot_price > max_pain else "below"
                    st.info(f"Spot is {format_currency(pain_distance)} {pain_direction} Max Pain")
                else:
                    st.metric("🎯 Max Pain Strike", "N/A")
            
            with col2:
                if not pain_df.empty:
                    pain_chart = chart_gen.create_pain_chart(pain_df, spot_price)
                    st.plotly_chart(pain_chart, use_container_width=True)
        
        with tab2:
            iv_chart = chart_gen.create_iv_chart(df)
            st.plotly_chart(iv_chart, use_container_width=True)
            
            avg_call_iv = df["CE IV"].mean()
            avg_put_iv = df["PE IV"].mean()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Avg Call IV", f"{avg_call_iv:.2f}%")
            col2.metric("Avg Put IV", f"{avg_put_iv:.2f}%")
            col3.metric("IV Difference", f"{abs(avg_call_iv - avg_put_iv):.2f}%")
        
        with tab3:
            st.dataframe(
                df,
                hide_index=True,
                use_container_width=True,
                height=400,
                column_config={
                    "Strike Price": st.column_config.NumberColumn("Strike", format="₹%.0f"),
                    "CE OI": st.column_config.NumberColumn("CE OI", format="%,.0f"),
                    "PE OI": st.column_config.NumberColumn("PE OI", format="%,.0f"),
                    "CE Volume": st.column_config.NumberColumn("CE Vol", format="%,.0f"),
                    "PE Volume": st.column_config.NumberColumn("PE Vol", format="%,.0f"),
                    "CE LTP": st.column_config.NumberColumn("CE LTP", format="₹%.2f"),
                    "PE LTP": st.column_config.NumberColumn("PE LTP", format="₹%.2f"),
                    "CE IV": st.column_config.NumberColumn("CE IV", format="%.2f%%"),
                    "PE IV": st.column_config.NumberColumn("PE IV", format="%.2f%%"),
                }
            )
    
    # Summary Stats
    st.divider()
    st.subheader("📈 Market Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Call OI", f"{pcr_data['total_call_oi']:,.0f}")
        st.metric("Total Call Volume", f"{pcr_data['total_call_vol']:,.0f}")
    
    with col2:
        st.metric("Total Put OI", f"{pcr_data['total_put_oi']:,.0f}")
        st.metric("Total Put Volume", f"{pcr_data['total_put_vol']:,.0f}")
    
    with col3:
        oi_diff = pcr_data['total_put_oi'] - pcr_data['total_call_oi']
        st.metric("Net OI (Put - Call)", f"{oi_diff:,.0f}",
                 delta="Bullish" if oi_diff > 0 else "Bearish")
    
    with col4:
        active_strikes = len(df[(df["CE OI"] > 0) | (df["PE OI"] > 0)])
        st.metric("Active Strikes", active_strikes)
    
    # Footer
    st.divider()
    st.caption("⚠️ **Disclaimer:** This dashboard is for educational purposes only. Trading in derivatives involves risk. Please consult a financial advisor before making trading decisions.")
    
    # Auto Refresh
    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()


if __name__ == "__main__":
    main()
