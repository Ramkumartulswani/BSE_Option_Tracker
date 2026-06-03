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

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="📊 BSE Option Chain Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_SCRIP  = "1"
API_BASE_URL   = "https://api.bseindia.com/BseIndiaAPI/api/DerivOptionChain_IV/w"
CACHE_TTL      = 30
NEARBY_RANGE   = 500

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
      padding: 14px 18px;
      border-radius: 10px;
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.12);
      text-align: center;
  }
  .metric-card .label { font-size: 12px; opacity: .7; margin-bottom: 4px; }
  .metric-card .value { font-size: 22px; font-weight: 700; }
  .signal-box {
      padding: 14px;
      border-radius: 8px;
      margin: 8px 0;
  }
  .tag {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 20px;
      font-size: 11px;
      font-weight: 600;
      margin: 2px;
  }
  .tag-itm   { background:#1a7a4a; color:#aff5c8; }
  .tag-atm   { background:#7a6a10; color:#fff0a0; }
  .tag-otm   { background:#1a3a6a; color:#a0cfff; }
  .tag-high  { background:#6a1a1a; color:#ffaaaa; }
  .tag-bull  { background:#0f5f2f; color:#90ff90; }
  .tag-bear  { background:#5f0f0f; color:#ff9090; }
  .tag-neut  { background:#2e2e2e; color:#cccccc; }
  .tag-surge { background:#5f3f00; color:#ffd080; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Expiry Helpers
# ─────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_expiry_dates(scrip_cd: str = "1") -> List[str]:
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
            f"https://api.bseindia.com/BseIndiaAPI/api/DerivExpiryDates/w?scripcode={scrip_cd}",
        ]
        expiry_dates = []
        for url in urls_to_try:
            try:
                r = session.get(url, headers=headers, timeout=10)
                if r.status_code == 200:
                    data = r.json()
                    for key in ['Table','expiry','expiryDate','ExpiryDates','Expiry','expDates','ExpiryList','expirylist','expiryDt']:
                        if key in data and isinstance(data[key], list):
                            for item in data[key]:
                                if isinstance(item, dict):
                                    for dk in ['expiry','Expiry','ExpiryDate','expiry_date','Expiry_Date','expiryDt','ExpiryDt']:
                                        if dk in item and item[dk]:
                                            d = str(item[dk]).strip()
                                            if d and d not in expiry_dates:
                                                expiry_dates.append(d)
                                elif isinstance(item, str) and item.strip() not in expiry_dates:
                                    expiry_dates.append(item.strip())
                    if expiry_dates:
                        break
            except Exception:
                continue
        if not expiry_dates:
            expiry_dates = generate_default_expiries()
        expiry_dates = list(set([e.strip() for e in expiry_dates if e and e.strip()]))
        try:
            expiry_dates.sort(key=lambda x: datetime.strptime(x, "%d %b %Y"))
        except Exception:
            pass
        return expiry_dates
    except Exception:
        return generate_default_expiries()


def generate_default_expiries() -> List[str]:
    current_date = datetime.now()
    expiry_dates = []
    for i in range(12):
        month = current_date.month + i
        year  = current_date.year + (month - 1) // 12
        month = ((month - 1) % 12) + 1
        last_day = (datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)) - timedelta(days=1)
        while last_day.weekday() != 3:
            last_day -= timedelta(days=1)
        expiry_dates.append(last_day.strftime("%d %b %Y"))
    return expiry_dates


# ─────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────
def safe_float(value, default=0.0) -> float:
    try:
        return float(str(value).replace(",", "").strip())
    except Exception:
        return default


def fmt(value: float) -> str:
    return f"₹{value:,.2f}"


def fmt_cr(value: float) -> str:
    """Format large numbers in Crores"""
    if value >= 1e7:
        return f"₹{value/1e7:.2f} Cr"
    elif value >= 1e5:
        return f"₹{value/1e5:.2f} L"
    return f"₹{value:,.0f}"


def validate_expiry_format(expiry: str) -> bool:
    for fmt_str in ("%d %b %Y", "%d-%b-%Y"):
        try:
            datetime.strptime(expiry, fmt_str)
            return True
        except Exception:
            pass
    return False


# ─────────────────────────────────────────────
# Data Fetching
# ─────────────────────────────────────────────
@st.cache_data(ttl=CACHE_TTL)
def fetch_bse_option_chain(expiry: str, scrip_cd: str, strprice: str = "0"):
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
        r = session.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return None, None, f"❌ HTTP {r.status_code}", None, None
        data = r.json()
        table = data.get("Table", [])
        if not table:
            return None, None, "⚠️ No data for given expiry.", None, None
        df         = _process_df(table)
        spot_price = _extract_spot(data, table, df)
        day_high   = _extract_val(data, ["High","high","DayHigh","dayHigh"])
        day_low    = _extract_val(data, ["Low","low","DayLow","dayLow"])
        return df, spot_price, None, day_high, day_low
    except requests.exceptions.Timeout:
        return None, None, "⏱️ Timeout.", None, None
    except Exception as e:
        return None, None, f"⚠️ {e}", None, None


def _process_df(table):
    df = pd.DataFrame(table)
    df = df.rename(columns={
        "Strike_Price1": "Strike Price",
        "Open_Interest": "PE OI",    "C_Open_Interest": "CE OI",
        "Vol_Traded":    "PE Volume", "C_Vol_Traded":    "CE Volume",
        "Last_Trd_Price":"PE LTP",   "C_Last_Trd_Price":"CE LTP",
        "IV":            "PE IV",    "C_IV":             "CE IV",
    })
    cols = ["Strike Price","CE OI","CE LTP","CE Volume","CE IV",
            "PE OI","PE LTP","PE Volume","PE IV"]
    df = df[cols]
    for col in cols:
        df[col] = df[col].astype(str).str.replace(",","").replace(["","None"," "],"0")
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).round(2)
    df = df[(df["CE OI"] > 0) | (df["PE OI"] > 0)]
    return df.sort_values("Strike Price").reset_index(drop=True)


def _extract_spot(data, table, df):
    keys = ["UlaValue","UnderlyingValue","underlyingValue","Underlying_Value","spotPrice","SpotPrice","IndexValue","indexValue"]
    for k in keys:
        if k in data and data[k]:
            v = safe_float(data[k])
            if v > 0: return v
    if table:
        for k in keys:
            if k in table[0]:
                v = safe_float(table[0][k])
                if v > 0: return v
    return float(df["Strike Price"].median())


def _extract_val(data, keys):
    for k in keys:
        if k in data and data[k]:
            v = safe_float(data[k])
            if v > 0: return v
    return None


# ─────────────────────────────────────────────
# NEW: Total OI & Premium Summary
# ─────────────────────────────────────────────
def compute_totals(df: pd.DataFrame, spot_price: float) -> Dict:
    total_ce_oi      = df["CE OI"].sum()
    total_pe_oi      = df["PE OI"].sum()
    total_oi         = total_ce_oi + total_pe_oi
    total_ce_vol     = df["CE Volume"].sum()
    total_pe_vol     = df["PE Volume"].sum()
    total_vol        = total_ce_vol + total_pe_vol

    # Total Premium (LTP × OI as proxy for notional premium)
    df["CE Premium"] = df["CE LTP"] * df["CE OI"]
    df["PE Premium"] = df["PE LTP"] * df["PE OI"]
    total_ce_premium = df["CE Premium"].sum()
    total_pe_premium = df["PE Premium"].sum()
    total_premium    = total_ce_premium + total_pe_premium

    # ATM Straddle price
    atm_idx    = (df["Strike Price"] - spot_price).abs().idxmin()
    atm_strike = df.loc[atm_idx, "Strike Price"]
    atm_ce_ltp = df.loc[atm_idx, "CE LTP"]
    atm_pe_ltp = df.loc[atm_idx, "PE LTP"]
    straddle_price = atm_ce_ltp + atm_pe_ltp

    # Upper/lower breakeven
    breakeven_up   = atm_strike + straddle_price
    breakeven_down = atm_strike - straddle_price

    # OI skew %
    oi_skew = ((total_pe_oi - total_ce_oi) / total_oi * 100) if total_oi > 0 else 0

    # Premium skew
    premium_skew = ((total_pe_premium - total_ce_premium) / total_premium * 100) if total_premium > 0 else 0

    pcr_oi  = round(total_pe_oi / total_ce_oi, 3)   if total_ce_oi  > 0 else 0
    pcr_vol = round(total_pe_vol / total_ce_vol, 3)  if total_ce_vol > 0 else 0

    # Max OI strikes
    max_ce_strike = df.loc[df["CE OI"].idxmax(), "Strike Price"] if total_ce_oi > 0 else None
    max_pe_strike = df.loc[df["PE OI"].idxmax(), "Strike Price"] if total_pe_oi > 0 else None

    return {
        "total_ce_oi": total_ce_oi, "total_pe_oi": total_pe_oi, "total_oi": total_oi,
        "total_ce_vol": total_ce_vol, "total_pe_vol": total_pe_vol, "total_vol": total_vol,
        "total_ce_premium": total_ce_premium, "total_pe_premium": total_pe_premium, "total_premium": total_premium,
        "atm_strike": atm_strike, "atm_ce_ltp": atm_ce_ltp, "atm_pe_ltp": atm_pe_ltp,
        "straddle_price": straddle_price, "breakeven_up": breakeven_up, "breakeven_down": breakeven_down,
        "oi_skew": oi_skew, "premium_skew": premium_skew,
        "pcr_oi": pcr_oi, "pcr_vol": pcr_vol,
        "max_ce_strike": max_ce_strike, "max_pe_strike": max_pe_strike,
    }


# ─────────────────────────────────────────────
# NEW: Price-Based Directional Bias Engine
# ─────────────────────────────────────────────
def compute_price_bias(
    spot_price: float,
    totals: Dict,
    nearest_support: Optional[float],
    nearest_resistance: Optional[float],
    max_pain: Optional[float],
    day_high: Optional[float],
    day_low: Optional[float],
    pcr_oi: float,
) -> Dict:
    """
    Score market direction using PRICE-LEVEL evidence only.
    Each factor votes +1 (bullish) / -1 (bearish) / 0 (neutral).
    Final score → verdict + recommended side.
    """
    factors = []

    # ── 1. Spot vs Support / Resistance proximity ──────────────────────
    if nearest_support and nearest_resistance:
        dist_sup = spot_price - nearest_support
        dist_res = nearest_resistance - spot_price
        proximity_ratio = dist_sup / (dist_sup + dist_res) if (dist_sup + dist_res) > 0 else 0.5
        if proximity_ratio < 0.35:          # closer to support → bounce likely
            vote, reason = +1, f"Spot ₹{spot_price:,.0f} is only {dist_sup:.0f} pts above support {nearest_support:,.0f} → bounce zone"
        elif proximity_ratio > 0.65:        # closer to resistance → rejection likely
            vote, reason = -1, f"Spot ₹{spot_price:,.0f} is only {dist_res:.0f} pts below resistance {nearest_resistance:,.0f} → rejection zone"
        else:
            vote, reason = 0, f"Spot is mid-range between support ({nearest_support:,.0f}) and resistance ({nearest_resistance:,.0f})"
        factors.append({"name": "Support/Resistance Proximity", "vote": vote, "reason": reason,
                         "bull_val": f"{nearest_support:,.0f}", "bear_val": f"{nearest_resistance:,.0f}"})
    else:
        factors.append({"name": "Support/Resistance Proximity", "vote": 0,
                         "reason": "Not enough data", "bull_val": "—", "bear_val": "—"})

    # ── 2. Spot vs Max Pain ───────────────────────────────────────────
    if max_pain:
        mp_diff = spot_price - max_pain
        mp_pct  = mp_diff / max_pain * 100
        if mp_diff > 0:
            vote   = -1
            reason = f"Spot ({spot_price:,.0f}) is {mp_diff:.0f} pts ABOVE max pain ({max_pain:,.0f}) → gravity pull DOWN (sell calls / buy puts)"
        elif mp_diff < 0:
            vote   = +1
            reason = f"Spot ({spot_price:,.0f}) is {abs(mp_diff):.0f} pts BELOW max pain ({max_pain:,.0f}) → gravity pull UP (sell puts / buy calls)"
        else:
            vote, reason = 0, f"Spot is AT max pain ({max_pain:,.0f}) → pinning, avoid directional"
        factors.append({"name": "Max Pain Gravity", "vote": vote, "reason": reason,
                         "bull_val": f"{max_pain:,.0f} (pull up)" if mp_diff < 0 else "—",
                         "bear_val": f"{max_pain:,.0f} (pull down)" if mp_diff > 0 else "—"})
    else:
        factors.append({"name": "Max Pain Gravity", "vote": 0, "reason": "Max pain not calculated",
                         "bull_val": "—", "bear_val": "—"})

    # ── 3. Straddle Breakeven Zone ────────────────────────────────────
    be_up   = totals["breakeven_up"]
    be_down = totals["breakeven_down"]
    if be_up and be_down:
        if spot_price > be_up:
            vote   = +1
            reason = f"Spot ({spot_price:,.0f}) has broken ABOVE upper breakeven ({be_up:,.0f}) → trending up strongly"
        elif spot_price < be_down:
            vote   = -1
            reason = f"Spot ({spot_price:,.0f}) has broken BELOW lower breakeven ({be_down:,.0f}) → trending down strongly"
        else:
            mid    = (be_up + be_down) / 2
            vote   = +1 if spot_price > mid else -1
            side   = "upper half" if spot_price > mid else "lower half"
            reason = f"Spot is inside straddle zone ({be_down:,.0f}–{be_up:,.0f}), in {side} → mild directional lean"
        factors.append({"name": "Straddle Breakeven Zone", "vote": vote, "reason": reason,
                         "bull_val": f">{be_up:,.0f}", "bear_val": f"<{be_down:,.0f}"})

    # ── 4. Day Range Position ─────────────────────────────────────────
    if day_high and day_low and (day_high - day_low) > 0:
        day_range  = day_high - day_low
        pos        = (spot_price - day_low) / day_range   # 0–1
        if pos >= 0.70:
            vote   = -1
            reason = f"Spot is in TOP {pos*100:.0f}% of day range → overextended, mean-reversion risk"
        elif pos <= 0.30:
            vote   = +1
            reason = f"Spot is in BOTTOM {pos*100:.0f}% of day range → oversold intraday, bounce likely"
        else:
            vote   = +1 if pos > 0.5 else -1
            reason = f"Spot at {pos*100:.0f}% of day range → mild {'upper' if pos>0.5 else 'lower'} bias"
        factors.append({"name": "Day Range Position", "vote": vote, "reason": reason,
                         "bull_val": f"<30% ({day_low:,.0f}–{day_low+day_range*0.3:,.0f})",
                         "bear_val": f">70% ({day_low+day_range*0.7:,.0f}–{day_high:,.0f})"})

    # ── 5. Max CE OI (Resistance) vs Max PE OI (Support) distance ────
    mc_strike = totals.get("max_ce_strike")
    mp_strike = totals.get("max_pe_strike")
    if mc_strike and mp_strike:
        dist_to_res = mc_strike - spot_price
        dist_to_sup = spot_price - mp_strike
        if dist_to_sup < dist_to_res * 0.5:
            vote   = +1
            reason = f"Max PE OI wall ({mp_strike:,.0f}) very close below → strong floor, lean BULLISH"
        elif dist_to_res < dist_to_sup * 0.5:
            vote   = -1
            reason = f"Max CE OI wall ({mc_strike:,.0f}) very close above → strong ceiling, lean BEARISH"
        else:
            vote   = 0
            reason = f"Max CE OI at {mc_strike:,.0f} (+{dist_to_res:.0f}), Max PE OI at {mp_strike:,.0f} (-{dist_to_sup:.0f}) — balanced"
        factors.append({"name": "Max OI Wall Distance", "vote": vote, "reason": reason,
                         "bull_val": f"PE wall {mp_strike:,.0f}", "bear_val": f"CE wall {mc_strike:,.0f}"})

    # ── 6. PCR cross-check ────────────────────────────────────────────
    if pcr_oi >= 1.2:
        vote, reason = +1, f"PCR {pcr_oi:.2f} ≥ 1.2 → strong put writing, bullish confirmation"
    elif pcr_oi <= 0.8:
        vote, reason = -1, f"PCR {pcr_oi:.2f} ≤ 0.8 → strong call writing, bearish confirmation"
    else:
        vote, reason = 0, f"PCR {pcr_oi:.2f} is neutral (0.8–1.2)"
    factors.append({"name": "PCR Confirmation", "vote": vote, "reason": reason,
                     "bull_val": "≥1.2", "bear_val": "≤0.8"})

    # ── Score ─────────────────────────────────────────────────────────
    score      = sum(f["vote"] for f in factors)
    max_score  = len(factors)
    bull_count = sum(1 for f in factors if f["vote"] == +1)
    bear_count = sum(1 for f in factors if f["vote"] == -1)
    neut_count = sum(1 for f in factors if f["vote"] ==  0)

    pct = score / max_score * 100  # -100 to +100

    if   pct >= 50:  verdict, color, action, emoji = "STRONG BUY CALLS",  "#00cc44", "BUY CALLS / SELL PUTS",  "🚀"
    elif pct >= 20:  verdict, color, action, emoji = "MILD BUY CALLS",    "#66dd88", "Consider CALL buying",    "📈"
    elif pct <= -50: verdict, color, action, emoji = "STRONG BUY PUTS",   "#cc2200", "BUY PUTS / SELL CALLS",   "🔻"
    elif pct <= -20: verdict, color, action, emoji = "MILD BUY PUTS",     "#dd6666", "Consider PUT buying",     "📉"
    else:            verdict, color, action, emoji = "RANGE / NEUTRAL",   "#aaaaaa", "Straddle / Iron Condor",  "⚖️"

    return {
        "factors":    factors,
        "score":      score,
        "max_score":  max_score,
        "pct":        pct,
        "verdict":    verdict,
        "color":      color,
        "action":     action,
        "emoji":      emoji,
        "bull_count": bull_count,
        "bear_count": bear_count,
        "neut_count": neut_count,
    }


def render_price_direction_panel(bias: Dict, spot_price: float, totals: Dict,
                                  nearest_support: Optional[float],
                                  nearest_resistance: Optional[float]):
    """Render the Price-Based Direction Panel"""

    pct   = bias["pct"]
    # Gauge fill: map -100..+100 → 0..100% width, with center at 50%
    gauge_fill = 50 + pct / 2          # 0% = full bear, 50% = neutral, 100% = full bull
    gauge_fill = max(2, min(98, gauge_fill))

    # Color gradient: red (bear) → grey (neutral) → green (bull)
    bar_color = bias["color"]

    vote_icons = {+1: "🟢", -1: "🔴", 0: "⚪"}

    # Factor rows HTML
    rows_html = ""
    for f in bias["factors"]:
        icon  = vote_icons[f["vote"]]
        label = "BULLISH" if f["vote"] == +1 else ("BEARISH" if f["vote"] == -1 else "NEUTRAL")
        lc    = "#66dd88" if f["vote"] == +1 else ("#dd6666" if f["vote"] == -1 else "#aaaaaa")
        rows_html += f"""
        <tr>
          <td style='padding:7px 10px;font-weight:600;color:#ddd'>{f['name']}</td>
          <td style='padding:7px 10px;text-align:center'>{icon}
              <span style='font-size:11px;color:{lc};font-weight:700'> {label}</span></td>
          <td style='padding:7px 10px;font-size:12px;color:#ccc'>{f['reason']}</td>
        </tr>"""

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);
                border:2px solid {bar_color};border-radius:14px;padding:24px;margin:20px 0'>

      <div style='text-align:center;margin-bottom:20px'>
        <div style='font-size:13px;opacity:.7;letter-spacing:2px;text-transform:uppercase'>
          Price-Based Directional Verdict
        </div>
        <div style='font-size:36px;font-weight:900;color:{bar_color};margin:8px 0'>
          {bias['emoji']} {bias['verdict']}
        </div>
        <div style='font-size:16px;color:#FFD700;font-weight:600'>{bias['action']}</div>
      </div>

      <!-- Score Gauge -->
      <div style='margin:18px 0'>
        <div style='display:flex;justify-content:space-between;font-size:12px;opacity:.7;margin-bottom:4px'>
          <span>🔴 STRONG BEAR</span><span>⚪ NEUTRAL</span><span>🟢 STRONG BULL</span>
        </div>
        <div style='background:#333;border-radius:20px;height:28px;position:relative;overflow:hidden'>
          <!-- neutral centre line -->
          <div style='position:absolute;left:50%;top:0;width:2px;height:100%;background:rgba(255,255,255,.3)'></div>
          <!-- fill bar, always from centre outward -->
          {"<div style='position:absolute;left:50%;top:0;width:" + str(abs(gauge_fill-50)) + "%;height:100%;background:" + bar_color + ";opacity:.85;" + ("" if pct>=0 else "right:50%;left:auto;") + "'></div>"
           if pct >= 0 else
           "<div style='position:absolute;right:50%;top:0;width:" + str(abs(gauge_fill-50)) + "%;height:100%;background:" + bar_color + ";opacity:.85;'></div>"
          }
          <div style='position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);
                      font-size:13px;font-weight:700;color:white;text-shadow:0 0 6px #000'>
            Score: {bias['score']:+d} / {bias['max_score']} &nbsp;|&nbsp;
            🟢 {bias['bull_count']} &nbsp; 🔴 {bias['bear_count']} &nbsp; ⚪ {bias['neut_count']}
          </div>
        </div>
      </div>

      <!-- Factor Table -->
      <table style='width:100%;border-collapse:collapse;margin-top:16px'>
        <thead>
          <tr style='border-bottom:1px solid #444'>
            <th style='padding:7px 10px;text-align:left;color:#aaa;font-size:12px'>FACTOR</th>
            <th style='padding:7px 10px;text-align:center;color:#aaa;font-size:12px'>VOTE</th>
            <th style='padding:7px 10px;text-align:left;color:#aaa;font-size:12px'>PRICE EVIDENCE</th>
          </tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>

    </div>
    """, unsafe_allow_html=True)

    # ── Recommended Strikes box ──
    atm    = totals["atm_strike"]
    st_px  = totals["straddle_price"]
    be_up  = totals["breakeven_up"]
    be_dn  = totals["breakeven_down"]

    if bias["pct"] >= 20:           # bullish
        rec_buy  = f"BUY CALL @ {atm:,.0f} (ATM)"
        rec_sell = f"SELL PUT @ {nearest_support:,.0f} (support)" if nearest_support else "SELL OTM PUT"
        tgt      = fmt(nearest_resistance) if nearest_resistance else "next resistance"
        sl       = fmt(nearest_support - 100) if nearest_support else "below support"
        box_color = "#0f4f2f"
        border    = "#00cc44"
    elif bias["pct"] <= -20:        # bearish
        rec_buy  = f"BUY PUT @ {atm:,.0f} (ATM)"
        rec_sell = f"SELL CALL @ {nearest_resistance:,.0f} (resistance)" if nearest_resistance else "SELL OTM CALL"
        tgt      = fmt(nearest_support) if nearest_support else "next support"
        sl       = fmt(nearest_resistance + 100) if nearest_resistance else "above resistance"
        box_color = "#4f0f0f"
        border    = "#cc2200"
    else:                            # neutral
        rec_buy  = f"BUY STRADDLE @ {atm:,.0f}"
        rec_sell = f"SELL STRANGLE: CE {nearest_resistance:,.0f} / PE {nearest_support:,.0f}" \
                   if nearest_resistance and nearest_support else "IRON CONDOR"
        tgt      = f"Collect premium within {fmt(be_dn)} – {fmt(be_up)}"
        sl       = "Exit if spot breaks outside straddle zone"
        box_color = "#2a2a2a"
        border    = "#888888"

    st.markdown(f"""
    <div style='background:{box_color};border-left:5px solid {border};
                border-radius:10px;padding:18px;margin:10px 0'>
      <div style='font-size:15px;font-weight:700;color:{border};margin-bottom:10px'>
        🎯 Recommended Trade Setup (Price-Based)
      </div>
      <div style='display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;font-size:13px'>
        <div><div style='opacity:.7;font-size:11px'>PRIMARY</div><div style='font-weight:700;color:#FFD700'>{rec_buy}</div></div>
        <div><div style='opacity:.7;font-size:11px'>HEDGE / PREMIUM</div><div style='font-weight:700;color:#ccc'>{rec_sell}</div></div>
        <div><div style='opacity:.7;font-size:11px'>TARGET</div><div style='font-weight:700;color:#66ff99'>{tgt}</div></div>
        <div><div style='opacity:.7;font-size:11px'>STOP LOSS (SPOT)</div><div style='font-weight:700;color:#ff6666'>{sl}</div></div>
      </div>
      <div style='margin-top:12px;font-size:11px;opacity:.6'>
        ATM: {fmt(atm)} · Straddle: {fmt(st_px)} · Breakevens: {fmt(be_dn)} ↔ {fmt(be_up)}
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# NEW: Full Chain Enrichment (decision signals)
# ─────────────────────────────────────────────
def enrich_chain(df: pd.DataFrame, spot_price: float) -> pd.DataFrame:
    out = df.copy()

    # — Moneyness tag
    atm_strike = (out["Strike Price"] - spot_price).abs().idxmin()
    atm_val    = out.loc[atm_strike, "Strike Price"]

    def moneyness(s):
        if abs(s - atm_val) <= (atm_val * 0.003):  # within 0.3%
            return "ATM"
        elif s < spot_price:
            return "ITM-CE / OTM-PE"
        else:
            return "OTM-CE / ITM-PE"
    out["Moneyness"] = out["Strike Price"].apply(moneyness)

    # — OI Concentration (top 20% by each side)
    ce_q80 = out["CE OI"].quantile(0.80)
    pe_q80 = out["PE OI"].quantile(0.80)
    out["CE Concentration"] = out["CE OI"].apply(lambda x: "🔴 HIGH" if x >= ce_q80 else "")
    out["PE Concentration"] = out["PE OI"].apply(lambda x: "🟢 HIGH" if x >= pe_q80 else "")

    # — Volume surge (vol > 2× median)
    ce_med_vol = out["CE Volume"].median()
    pe_med_vol = out["PE Volume"].median()
    out["CE Vol Surge"] = out["CE Volume"].apply(lambda x: "⚡ SURGE" if x > 2 * ce_med_vol and ce_med_vol > 0 else "")
    out["PE Vol Surge"] = out["PE Volume"].apply(lambda x: "⚡ SURGE" if x > 2 * pe_med_vol and pe_med_vol > 0 else "")

    # — OI Ratio per strike (PE OI / CE OI)
    out["Strike PCR"] = out.apply(
        lambda r: round(r["PE OI"] / r["CE OI"], 2) if r["CE OI"] > 0 else np.nan, axis=1
    )

    # — Straddle price per strike
    out["Straddle"] = (out["CE LTP"] + out["PE LTP"]).round(2)

    # — Notional Premium (₹)
    out["CE Notional"] = (out["CE LTP"] * out["CE OI"]).round(0)
    out["PE Notional"] = (out["PE LTP"] * out["PE OI"]).round(0)

    # — Per-strike bias signal
    def strike_bias(row):
        pcr = row["Strike PCR"]
        if pd.isna(pcr):
            return "—"
        if pcr >= 1.5:
            return "🟢 Strong Support"
        elif pcr >= 1.1:
            return "🟡 Mild Support"
        elif pcr <= 0.6:
            return "🔴 Strong Resistance"
        elif pcr <= 0.9:
            return "🟠 Mild Resistance"
        else:
            return "⚪ Neutral"
    out["Strike Bias"] = out.apply(strike_bias, axis=1)

    # — IV Skew signal (PE IV vs CE IV)
    def iv_signal(row):
        if row["CE IV"] == 0 or row["PE IV"] == 0:
            return "—"
        ratio = row["PE IV"] / row["CE IV"]
        if ratio > 1.3:
            return "⬇️ Fear (High PE IV)"
        elif ratio < 0.7:
            return "⬆️ Greed (High CE IV)"
        else:
            return "➡️ Balanced"
    out["IV Signal"] = out.apply(iv_signal, axis=1)

    # — Distance from spot
    out["Dist from Spot"] = (out["Strike Price"] - spot_price).round(1)
    out["Dist %"] = ((out["Strike Price"] - spot_price) / spot_price * 100).round(2)

    return out


# ─────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────
class OptionAnalyzer:
    @staticmethod
    def calculate_pcr_analysis(df: pd.DataFrame) -> Dict:
        total_ce_oi  = df["CE OI"].sum()
        total_pe_oi  = df["PE OI"].sum()
        total_ce_vol = df["CE Volume"].sum()
        total_pe_vol = df["PE Volume"].sum()
        pcr_oi  = round(total_pe_oi  / total_ce_oi,  2) if total_ce_oi  > 0 else 0
        pcr_vol = round(total_pe_vol / total_ce_vol, 2) if total_ce_vol > 0 else 0
        if pcr_oi > 1.2:
            sentiment, description, color = "📈 Bullish", "Strong Put Writing – Support Building", "green"
        elif pcr_oi < 0.8:
            sentiment, description, color = "📉 Bearish", "Strong Call Writing – Resistance Building", "red"
        else:
            sentiment, description, color = "⚖️ Neutral", "Balanced Market Conditions", "orange"
        return {
            "pcr_oi": pcr_oi, "pcr_vol": pcr_vol,
            "sentiment": sentiment, "description": description, "color": color,
            "total_call_oi": total_ce_oi, "total_put_oi": total_pe_oi,
            "total_call_vol": total_ce_vol, "total_put_vol": total_pe_vol,
        }

    @staticmethod
    def find_support_resistance(df, spot_price, num_levels=5):
        sup_df = df[df["Strike Price"] <= spot_price].nlargest(num_levels, "PE OI")
        res_df = df[df["Strike Price"] >= spot_price].nlargest(num_levels, "CE OI")
        supports    = sup_df[["Strike Price","PE OI","PE LTP"]].copy()
        resistances = res_df[["Strike Price","CE OI","CE LTP"]].copy()
        nearest_support    = sup_df["Strike Price"].max() if not sup_df.empty else None
        nearest_resistance = res_df["Strike Price"].min() if not res_df.empty else None
        return supports, resistances, nearest_support, nearest_resistance

    @staticmethod
    def analyze_max_pain(df):
        strikes = df["Strike Price"].unique()
        pain_values = []
        for strike in strikes:
            call_pain = (df[df["Strike Price"] > strike]["CE OI"] *
                         (df[df["Strike Price"] > strike]["Strike Price"] - strike)).sum()
            put_pain  = (df[df["Strike Price"] < strike]["PE OI"] *
                         (strike - df[df["Strike Price"] < strike]["Strike Price"])).sum()
            pain_values.append({"Strike": strike, "Pain": call_pain + put_pain})
        pain_df = pd.DataFrame(pain_values)
        max_pain = pain_df.loc[pain_df["Pain"].idxmin(), "Strike"] if not pain_df.empty else None
        return max_pain, pain_df

    @staticmethod
    def get_nearby_strikes(df, spot_price, range_points=NEARBY_RANGE):
        nearby = df[(df["Strike Price"] >= spot_price - range_points) &
                    (df["Strike Price"] <= spot_price + range_points)].copy()
        nearby["OI Diff"]  = nearby["PE OI"] - nearby["CE OI"]
        nearby["OI Ratio"] = nearby.apply(
            lambda r: round(r["PE OI"] / r["CE OI"], 2) if r["CE OI"] > 0 else 0, axis=1)
        med_ce = nearby["CE OI"].median()
        med_pe = nearby["PE OI"].median()
        nearby["CE Signal"] = nearby["CE OI"].apply(lambda x: "🔴" if x > med_ce else "")
        nearby["PE Signal"] = nearby["PE OI"].apply(lambda x: "🟢" if x > med_pe else "")
        return nearby.sort_values("Strike Price")

    @staticmethod
    def generate_trading_signals(df, spot_price, pcr_data, nearest_support, nearest_resistance):
        pcr = pcr_data["pcr_oi"]
        signals = {"call_buy":[], "put_buy":[], "call_sell":[], "put_sell":[], "market_bias":"", "strategy":""}
        if   pcr > 1.3: signals["market_bias"], signals["strategy"] = "Strongly Bullish",    "Buy Calls or Sell Puts"
        elif pcr > 1.0: signals["market_bias"], signals["strategy"] = "Moderately Bullish",  "Buy ATM/OTM Calls"
        elif pcr < 0.7: signals["market_bias"], signals["strategy"] = "Strongly Bearish",    "Buy Puts or Sell Calls"
        elif pcr < 0.9: signals["market_bias"], signals["strategy"] = "Moderately Bearish",  "Buy ATM/OTM Puts"
        else:           signals["market_bias"], signals["strategy"] = "Neutral/Rangebound",  "Iron Condor or Straddle"

        atm_idx    = (df['Strike Price'] - spot_price).abs().idxmin()
        atm_strike = df.loc[atm_idx, 'Strike Price']

        if pcr >= 1.0:
            signals["call_buy"].append({
                "strike": atm_strike, "type": "ATM Call",
                "target": nearest_resistance or spot_price + 500,
                "stop_loss": nearest_support or spot_price - 200,
                "reason": "ATM call for bullish move"})
            otm_c = df[df['Strike Price'] > spot_price].nsmallest(2, 'Strike Price')
            if not otm_c.empty:
                signals["call_buy"].append({
                    "strike": otm_c.iloc[0]['Strike Price'], "type": "OTM Call",
                    "target": nearest_resistance or spot_price + 700,
                    "stop_loss": spot_price - 100, "reason": "OTM call – aggressive bullish"})

        if pcr <= 0.9:
            signals["put_buy"].append({
                "strike": atm_strike, "type": "ATM Put",
                "target": nearest_support or spot_price - 500,
                "stop_loss": nearest_resistance or spot_price + 200,
                "reason": "ATM put for bearish move"})
            otm_p = df[df['Strike Price'] < spot_price].nlargest(2, 'Strike Price')
            if not otm_p.empty:
                signals["put_buy"].append({
                    "strike": otm_p.iloc[0]['Strike Price'], "type": "OTM Put",
                    "target": nearest_support or spot_price - 700,
                    "stop_loss": spot_price + 100, "reason": "OTM put – aggressive bearish"})

        if pcr >= 1.2:
            strong_sup = df[df["Strike Price"] < spot_price].nlargest(3, "PE OI")
            if not strong_sup.empty:
                ss = strong_sup.iloc[0]['Strike Price']
                signals["put_sell"].append({
                    "strike": ss, "type": "OTM Put Sell",
                    "target": "Premium collection", "stop_loss": ss - 200,
                    "reason": f"Strong support at {ss:,.0f} – high PE OI"})

        if pcr <= 0.8:
            strong_res = df[df["Strike Price"] > spot_price].nlargest(3, "CE OI")
            if not strong_res.empty:
                sr = strong_res.iloc[0]['Strike Price']
                signals["call_sell"].append({
                    "strike": sr, "type": "OTM Call Sell",
                    "target": "Premium collection", "stop_loss": sr + 200,
                    "reason": f"Strong resistance at {sr:,.0f} – high CE OI"})
        return signals


# ─────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────
class ChartGenerator:
    @staticmethod
    def create_oi_chart(df, spot_price):
        fig = make_subplots(rows=2, cols=1,
            subplot_titles=("Open Interest Distribution", "Volume Distribution"),
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
        fig.update_layout(height=600, hovermode='x unified', barmode='group', template='plotly_dark')
        return fig

    @staticmethod
    def create_iv_chart(df):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Strike Price"], y=df["CE IV"],
            mode='lines+markers', name='Call IV', line=dict(color='red', width=2)))
        fig.add_trace(go.Scatter(x=df["Strike Price"], y=df["PE IV"],
            mode='lines+markers', name='Put IV', line=dict(color='green', width=2)))
        fig.update_layout(title="Implied Volatility Smile",
            xaxis_title="Strike Price", yaxis_title="IV (%)",
            height=400, hovermode='x unified', template='plotly_dark')
        return fig

    @staticmethod
    def create_pain_chart(pain_df, spot_price):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pain_df["Strike"], y=pain_df["Pain"],
            mode='lines', fill='tozeroy', name='Total Pain',
            line=dict(color='purple', width=2)))
        fig.add_vline(x=spot_price, line_dash="dash", annotation_text="Spot", line_color="yellow")
        fig.update_layout(title="Max Pain Distribution",
            xaxis_title="Strike Price", yaxis_title="Pain Value",
            height=300, template='plotly_dark')
        return fig

    @staticmethod
    def create_total_oi_donut(totals: Dict) -> go.Figure:
        fig = make_subplots(rows=1, cols=2,
            subplot_titles=("OI Distribution", "Premium Distribution"),
            specs=[[{"type":"domain"},{"type":"domain"}]])
        fig.add_trace(go.Pie(
            labels=["Call OI", "Put OI"],
            values=[totals["total_ce_oi"], totals["total_pe_oi"]],
            hole=0.55, marker_colors=["#ff6347","#3cb371"],
            textinfo="label+percent"), row=1, col=1)
        fig.add_trace(go.Pie(
            labels=["Call Premium", "Put Premium"],
            values=[totals["total_ce_premium"], totals["total_pe_premium"]],
            hole=0.55, marker_colors=["#ffa07a","#90ee90"],
            textinfo="label+percent"), row=1, col=2)
        fig.update_layout(height=320, template='plotly_dark', showlegend=True)
        return fig

    @staticmethod
    def create_notional_bar(enriched: pd.DataFrame, spot_price: float) -> go.Figure:
        nearby = enriched[
            (enriched["Strike Price"] >= spot_price - NEARBY_RANGE) &
            (enriched["Strike Price"] <= spot_price + NEARBY_RANGE)
        ]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="CE Notional", x=nearby["Strike Price"], y=nearby["CE Notional"],
            marker_color='rgba(255,80,50,0.75)'))
        fig.add_trace(go.Bar(name="PE Notional", x=nearby["Strike Price"], y=nearby["PE Notional"],
            marker_color='rgba(50,200,100,0.75)'))
        fig.add_vline(x=spot_price, line_dash="dash", line_color="yellow",
            annotation_text=f"Spot {spot_price:.0f}")
        fig.update_layout(title="Notional Premium (LTP × OI) – Nearby Strikes",
            height=350, barmode='group', hovermode='x unified', template='plotly_dark',
            yaxis_title="Notional ₹")
        return fig

    @staticmethod
    def create_straddle_curve(enriched: pd.DataFrame, spot_price: float) -> go.Figure:
        nearby = enriched[
            (enriched["Strike Price"] >= spot_price - NEARBY_RANGE) &
            (enriched["Strike Price"] <= spot_price + NEARBY_RANGE)
        ]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=nearby["Strike Price"], y=nearby["Straddle"],
            mode='lines+markers', name='Straddle Price',
            fill='tozeroy', line=dict(color='#ff9f40', width=2)))
        fig.add_vline(x=spot_price, line_dash="dash", line_color="yellow",
            annotation_text=f"ATM {spot_price:.0f}")
        fig.update_layout(title="Straddle (CE LTP + PE LTP) across Strikes",
            height=300, hovermode='x unified', template='plotly_dark',
            yaxis_title="Straddle ₹")
        return fig

    @staticmethod
    def create_pcr_heatmap(enriched: pd.DataFrame, spot_price: float) -> go.Figure:
        nearby = enriched[
            (enriched["Strike Price"] >= spot_price - NEARBY_RANGE) &
            (enriched["Strike Price"] <= spot_price + NEARBY_RANGE)
        ].copy()
        nearby["Strike PCR"] = nearby["Strike PCR"].fillna(0)
        fig = go.Figure(go.Bar(
            x=nearby["Strike Price"],
            y=nearby["Strike PCR"],
            marker=dict(
                color=nearby["Strike PCR"],
                colorscale=[
                    [0.0, "#ff4040"], [0.4, "#ff9933"],
                    [0.5, "#888888"], [0.6, "#33cc66"],
                    [1.0, "#00aa00"]
                ],
                cmin=0, cmax=2,
                colorbar=dict(title="PCR", x=1.01)
            ),
            name="Strike PCR"
        ))
        fig.add_hline(y=1.0, line_dash="dash", line_color="white", annotation_text="PCR=1")
        fig.add_vline(x=spot_price, line_dash="dash", line_color="yellow")
        fig.update_layout(title="Per-Strike PCR (Put/Call OI Ratio)",
            height=320, hovermode='x unified', template='plotly_dark',
            yaxis_title="PCR")
        return fig


# ─────────────────────────────────────────────
# UI Helpers
# ─────────────────────────────────────────────
def render_day_range(spot_price, day_high, day_low):
    if not (day_high and day_low): return
    day_range  = day_high - day_low
    range_pct  = (day_range / day_low) * 100
    pos_in_rng = ((spot_price - day_low) / day_range * 100) if day_range > 0 else 50
    st.markdown(f"""
    <div style='padding:20px;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
                border-radius:10px;margin:15px 0;color:white;'>
        <h3 style='margin:0 0 15px 0;text-align:center;'>📊 Index Day Range</h3>
        <div style='display:flex;justify-content:space-around;text-align:center;'>
            <div><div style='font-size:13px;opacity:.85'>Day Low</div>
                 <div style='font-size:22px;font-weight:700'>{fmt(day_low)}</div></div>
            <div><div style='font-size:13px;opacity:.85'>Current (Spot)</div>
                 <div style='font-size:26px;font-weight:700;color:#FFD700'>{fmt(spot_price)}</div>
                 <div style='font-size:11px;margin-top:4px'>{pos_in_rng:.1f}% in range</div></div>
            <div><div style='font-size:13px;opacity:.85'>Day High</div>
                 <div style='font-size:22px;font-weight:700'>{fmt(day_high)}</div></div>
        </div>
        <div style='margin-top:18px;background:rgba(255,255,255,.2);border-radius:10px;padding:3px'>
            <div style='width:{min(pos_in_rng,100)}%;background:linear-gradient(90deg,#00ff00,#ffd700,#ff0000);
                        height:22px;border-radius:8px'></div>
        </div>
        <div style='display:flex;justify-content:space-between;margin-top:10px;font-size:13px'>
            <div>Range: {fmt(day_range)}</div><div>Movement: {range_pct:.2f}%</div>
        </div>
    </div>""", unsafe_allow_html=True)


def render_totals_panel(totals: Dict, spot_price: float):
    """Big summary panel: Total OI, Volume, Premium, Straddle, Breakeven"""
    st.markdown("""
    <h3 style='margin:0 0 10px 0'>📦 Total OI · Volume · Premium Summary</h3>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='label'>📊 Total Open Interest</div>
          <div class='value'>{totals['total_oi']:,.0f}</div>
          <div style='display:flex;justify-content:space-around;margin-top:10px;font-size:13px'>
            <span style='color:#ff6347'>🔴 CE: {totals['total_ce_oi']:,.0f}</span>
            <span style='color:#3cb371'>🟢 PE: {totals['total_pe_oi']:,.0f}</span>
          </div>
          <div style='font-size:12px;margin-top:6px;opacity:.7'>
            OI Skew (PE-CE): {"+" if totals['oi_skew']>=0 else ""}{totals['oi_skew']:.1f}%
          </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='label'>🔄 Total Volume</div>
          <div class='value'>{totals['total_vol']:,.0f}</div>
          <div style='display:flex;justify-content:space-around;margin-top:10px;font-size:13px'>
            <span style='color:#ff6347'>🔴 CE: {totals['total_ce_vol']:,.0f}</span>
            <span style='color:#3cb371'>🟢 PE: {totals['total_pe_vol']:,.0f}</span>
          </div>
          <div style='font-size:12px;margin-top:6px;opacity:.7'>
            Vol PCR: {totals['pcr_vol']:.2f}
          </div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='label'>💰 Total Notional Premium</div>
          <div class='value'>{fmt_cr(totals['total_premium'])}</div>
          <div style='display:flex;justify-content:space-around;margin-top:10px;font-size:13px'>
            <span style='color:#ff6347'>🔴 CE: {fmt_cr(totals['total_ce_premium'])}</span>
            <span style='color:#3cb371'>🟢 PE: {fmt_cr(totals['total_pe_premium'])}</span>
          </div>
          <div style='font-size:12px;margin-top:6px;opacity:.7'>
            Prem Skew (PE-CE): {"+" if totals['premium_skew']>=0 else ""}{totals['premium_skew']:.1f}%
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c4, c5, c6, c7 = st.columns(4)

    with c4:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='label'>🎯 ATM Strike</div>
          <div class='value'>{fmt(totals['atm_strike'])}</div>
          <div style='font-size:12px;margin-top:6px;opacity:.7'>CE LTP: {fmt(totals['atm_ce_ltp'])} | PE LTP: {fmt(totals['atm_pe_ltp'])}</div>
        </div>""", unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='label'>📐 ATM Straddle Price</div>
          <div class='value' style='color:#ffd700'>{fmt(totals['straddle_price'])}</div>
          <div style='font-size:12px;margin-top:6px;opacity:.7'>
            {(totals['straddle_price']/totals['atm_strike']*100):.2f}% of spot
          </div>
        </div>""", unsafe_allow_html=True)

    with c6:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='label'>⬆️ Breakeven Up</div>
          <div class='value' style='color:#3cb371'>{fmt(totals['breakeven_up'])}</div>
          <div style='font-size:12px;margin-top:6px;opacity:.7'>+{fmt(totals['breakeven_up']-spot_price)} from spot</div>
        </div>""", unsafe_allow_html=True)

    with c7:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='label'>⬇️ Breakeven Down</div>
          <div class='value' style='color:#ff6347'>{fmt(totals['breakeven_down'])}</div>
          <div style='font-size:12px;margin-top:6px;opacity:.7'>-{fmt(spot_price-totals['breakeven_down'])} from spot</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    ce_c, pe_c = st.columns(2)
    with ce_c:
        st.info(f"🔴 **Max Call OI at:** {fmt(totals['max_ce_strike'])} (Primary Resistance)")
    with pe_c:
        st.info(f"🟢 **Max Put OI at:** {fmt(totals['max_pe_strike'])} (Primary Support)")


def render_trading_signal(signal, signal_type):
    colors = {
        "call_buy":  ("green",    "🟢"),
        "put_buy":   ("red",      "🔴"),
        "put_sell":  ("darkgreen","💰"),
        "call_sell": ("darkred",  "💰"),
    }
    color, icon = colors.get(signal_type, ("gray","•"))
    tgt = fmt(signal['target']) if isinstance(signal['target'], (int,float)) else signal['target']
    st.markdown(f"""
    <div style='padding:14px;background-color:rgba(0,255,0,0.08);border-left:4px solid {color};border-radius:5px;margin:8px 0'>
        <h4 style='margin:0 0 8px 0;color:{color}'>{icon} {signal['type']} – Strike: {fmt(signal['strike'])}</h4>
        <p style='margin:4px 0'><strong>Target:</strong> {tgt}</p>
        <p style='margin:4px 0'><strong>Stop Loss (Spot):</strong> {fmt(signal['stop_loss'])}</p>
        <p style='margin:4px 0'><strong>Reason:</strong> {signal['reason']}</p>
    </div>""", unsafe_allow_html=True)


def _style_chain(df: pd.DataFrame) -> pd.DataFrame.style:
    """Color-code the enriched chain dataframe"""
    def color_bias(val):
        if "Strong Support"     in str(val): return "background-color:#0f4f2f;color:#90ff90"
        if "Mild Support"       in str(val): return "background-color:#1a3a1a;color:#ccffcc"
        if "Strong Resistance"  in str(val): return "background-color:#4f0f0f;color:#ff9090"
        if "Mild Resistance"    in str(val): return "background-color:#3a1a1a;color:#ffcccc"
        return ""

    def color_conc(val):
        if "HIGH" in str(val): return "font-weight:700;color:#ffd700"
        return ""

    def color_surge(val):
        if "SURGE" in str(val): return "background-color:#4a3000;color:#ffd080;font-weight:700"
        return ""

    def color_iv(val):
        if "Fear"   in str(val): return "color:#ff6666"
        if "Greed"  in str(val): return "color:#66ff66"
        return ""

    styled = df.style \
        .applymap(color_bias,   subset=["Strike Bias"]) \
        .applymap(color_conc,   subset=["CE Concentration","PE Concentration"]) \
        .applymap(color_surge,  subset=["CE Vol Surge","PE Vol Surge"]) \
        .applymap(color_iv,     subset=["IV Signal"]) \
        .background_gradient(subset=["Strike PCR"], cmap="RdYlGn", vmin=0, vmax=2) \
        .format({
            "Strike Price": lambda x: f"{x:,.0f}",
            "CE OI":        lambda x: f"{x:,.0f}",
            "PE OI":        lambda x: f"{x:,.0f}",
            "CE LTP":       lambda x: f"₹{x:,.2f}",
            "PE LTP":       lambda x: f"₹{x:,.2f}",
            "CE Volume":    lambda x: f"{x:,.0f}",
            "PE Volume":    lambda x: f"{x:,.0f}",
            "CE Notional":  lambda x: fmt_cr(x),
            "PE Notional":  lambda x: fmt_cr(x),
            "Straddle":     lambda x: f"₹{x:,.2f}",
            "Strike PCR":   lambda x: f"{x:.2f}" if not pd.isna(x) else "—",
            "Dist %":       lambda x: f"{x:+.2f}%",
        }, na_rep="—")
    return styled


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    st.title("📈 BSE Option Chain Pro Dashboard")
    st.markdown("**Real-time OI · Volume · Premium · Decision Analytics**")

    analyzer  = OptionAnalyzer()
    chart_gen = ChartGenerator()

    # ── Sidebar ──────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")
        scrip_cd = st.text_input("Scrip Code", DEFAULT_SCRIP, help="BSE scrip code")
        st.divider()

        st.subheader("📅 Expiry Date")
        expiry_input_method = st.radio("Input method:", ["Dropdown (Auto-loaded)", "Manual Entry"])

        if expiry_input_method == "Dropdown (Auto-loaded)":
            with st.spinner("Loading expiry dates..."):
                expiry_dates = fetch_expiry_dates(scrip_cd)
            if expiry_dates:
                expiry = st.selectbox("Select Expiry", options=expiry_dates, index=0)
                st.success(f"✅ {len(expiry_dates)} dates loaded")
            else:
                st.error("Failed to load dates")
                expiry = st.text_input("Enter Expiry", "13 Nov 2025")
        else:
            expiry = st.text_input("Enter Expiry", "13 Nov 2025", help="DD MMM YYYY")
            if validate_expiry_format(expiry): st.success("✅ Valid format")
            else: st.warning("⚠️ Use DD MMM YYYY")

        st.divider()
        manual_spot  = st.checkbox("Override Spot Price", value=False)
        custom_spot  = st.number_input("Enter Spot Price", min_value=0.0, value=50000.0,
                                       step=100.0, disabled=not manual_spot)
        st.divider()
        auto_refresh  = st.checkbox("Auto Refresh", value=False)
        refresh_rate  = st.slider("Refresh Interval (sec)", 10, 120, 30, disabled=not auto_refresh)
        st.divider()
        show_advanced = st.checkbox("Show Advanced Analytics", value=True)
        num_levels    = st.slider("Support/Resistance Levels", 3, 10, 5)
        nearby_range  = st.slider("Nearby Range (pts)", 100, 2000, NEARBY_RANGE, step=100)
        st.divider()
        st.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')}")
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()

    # ── Fetch ─────────────────────────────────
    with st.spinner("Fetching option chain data..."):
        df, spot_price, error, day_high, day_low = fetch_bse_option_chain(expiry, scrip_cd)

    if error:
        st.error(error)
        st.stop()
    if df is None or df.empty:
        st.warning(f"No data for expiry: {expiry}")
        st.stop()

    if manual_spot and custom_spot:
        st.info(f"ℹ️ Manual spot: {fmt(custom_spot)} (API: {fmt(spot_price)})")
        spot_price = custom_spot

    # ── Compute ───────────────────────────────
    totals   = compute_totals(df, spot_price)
    enriched = enrich_chain(df, spot_price)
    pcr_data = analyzer.calculate_pcr_analysis(df)
    supports, resistances, nearest_support, nearest_resistance = \
        analyzer.find_support_resistance(df, spot_price, num_levels)
    trading_signals = analyzer.generate_trading_signals(
        df, spot_price, pcr_data, nearest_support, nearest_resistance)
    max_pain, pain_df = analyzer.analyze_max_pain(df)
    price_bias = compute_price_bias(
        spot_price, totals, nearest_support, nearest_resistance,
        max_pain, day_high, day_low, totals["pcr_oi"]
    )

    st.success(f"✅ {expiry} · {len(df)} active strikes loaded")

    # ── Day Range ─────────────────────────────
    render_day_range(spot_price, day_high, day_low)

    # ── Key Metrics row ───────────────────────
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: st.metric("💰 Spot",          fmt(spot_price))
    with c2: st.metric("📈 Day High",       fmt(day_high) if day_high else "N/A")
    with c3: st.metric("📉 Day Low",        fmt(day_low) if day_low else "N/A")
    with c4: st.metric("📊 PCR (OI)",       totals["pcr_oi"])
    with c5: st.metric("🟢 Support",        fmt(nearest_support) if nearest_support else "N/A",
                        f"-{spot_price-nearest_support:.0f}" if nearest_support else "")
    with c6: st.metric("🔴 Resistance",     fmt(nearest_resistance) if nearest_resistance else "N/A",
                        f"+{nearest_resistance-spot_price:.0f}" if nearest_resistance else "")

    # ── Sentiment bar ─────────────────────────
    st.markdown(f"""
    <div style='padding:14px;background-color:{pcr_data['color']}22;
                border-left:5px solid {pcr_data['color']};border-radius:5px;margin:18px 0'>
        <h3 style='color:{pcr_data['color']};margin:0'>{pcr_data['sentiment']}</h3>
        <p style='margin:5px 0 0 0'>{pcr_data['description']}</p>
    </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Price-Based Direction Panel ───────────
    st.header("🧭 Price-Based Directional Analysis")
    render_price_direction_panel(
        price_bias, spot_price, totals, nearest_support, nearest_resistance
    )

    st.divider()

    # ── Total OI / Volume / Premium Panel ─────
    render_totals_panel(totals, spot_price)

    # ── OI / Premium donut charts ─────────────
    st.plotly_chart(chart_gen.create_total_oi_donut(totals), use_container_width=True)

    st.divider()

    # ── Trading Signals ───────────────────────
    st.header("🎯 Trading Signals")
    tab_cb, tab_pb, tab_ps, tab_cs = st.tabs(["📈 Call Buy","📉 Put Buy","💰 Put Sell","💰 Call Sell"])
    with tab_cb:
        if trading_signals["call_buy"]:
            for s in trading_signals["call_buy"]:
                render_trading_signal(s, "call_buy")
        else:
            st.warning("No call buy setups at current market conditions.")
    with tab_pb:
        if trading_signals["put_buy"]:
            for s in trading_signals["put_buy"]:
                render_trading_signal(s, "put_buy")
        else:
            st.warning("No put buy setups at current market conditions.")
    with tab_ps:
        if trading_signals["put_sell"]:
            for s in trading_signals["put_sell"]:
                render_trading_signal(s, "put_sell")
        else:
            st.info("No put sell setups at current market conditions.")
    with tab_cs:
        if trading_signals["call_sell"]:
            for s in trading_signals["call_sell"]:
                render_trading_signal(s, "call_sell")
        else:
            st.info("No call sell setups at current market conditions.")

    st.divider()

    # ── Support & Resistance ──────────────────
    s_col, r_col = st.columns(2)
    with s_col:
        st.subheader("🟢 Support Levels")
        st.dataframe(supports.style.format({
            "Strike Price": lambda x: fmt(x),
            "PE OI": lambda x: f"{x:,.0f}",
            "PE LTP": lambda x: fmt(x)
        }), hide_index=True, use_container_width=True)
    with r_col:
        st.subheader("🔴 Resistance Levels")
        st.dataframe(resistances.style.format({
            "Strike Price": lambda x: fmt(x),
            "CE OI": lambda x: f"{x:,.0f}",
            "CE LTP": lambda x: fmt(x)
        }), hide_index=True, use_container_width=True)

    st.divider()

    # ── OI & Volume Charts ────────────────────
    st.subheader("📊 OI & Volume Analysis")
    st.plotly_chart(chart_gen.create_oi_chart(df, spot_price), use_container_width=True)

    # ── Notional Premium Chart ────────────────
    st.subheader("💰 Notional Premium by Strike")
    st.plotly_chart(chart_gen.create_notional_bar(enriched, spot_price), use_container_width=True)

    # ── Straddle Curve ────────────────────────
    st.subheader("📐 Straddle Price Curve")
    st.plotly_chart(chart_gen.create_straddle_curve(enriched, spot_price), use_container_width=True)

    # ── Per-Strike PCR Heatmap ────────────────
    st.subheader("🌡️ Per-Strike PCR Heatmap")
    st.plotly_chart(chart_gen.create_pcr_heatmap(enriched, spot_price), use_container_width=True)

    # ── Advanced Analytics ────────────────────
    if show_advanced:
        st.divider()
        st.subheader("🎯 Advanced Analytics")
        adv1, adv2, adv3, adv4 = st.tabs(["Max Pain","IV Smile","Full Chain (Enriched)","Nearby Strikes"])

        with adv1:
            if max_pain:
                mc1, mc2 = st.columns([1,2])
                with mc1:
                    st.metric("🎯 Max Pain", fmt(max_pain))
                    delta = spot_price - max_pain
                    st.metric("Δ Spot vs Max Pain", f"{delta:+.0f}", delta_color="inverse")
                with mc2:
                    if not pain_df.empty:
                        st.plotly_chart(chart_gen.create_pain_chart(pain_df, spot_price),
                                        use_container_width=True)

        with adv2:
            st.plotly_chart(chart_gen.create_iv_chart(df), use_container_width=True)

        with adv3:
            st.markdown("""
            **Decision columns:**
            - **Strike Bias** — per-strike PCR interpretation  
            - **CE / PE Concentration** — OI in top 20% bucket  
            - **CE / PE Vol Surge** — volume > 2× median  
            - **IV Signal** — PE IV vs CE IV skew  
            - **Straddle** — CE LTP + PE LTP  
            - **CE / PE Notional** — LTP × OI (money deployed)  
            - **Strike PCR** — color-graded Put/Call OI ratio  
            """)

            display_cols = [
                "Strike Price","Dist from Spot","Dist %","Moneyness",
                "CE OI","CE LTP","CE Volume","CE IV","CE Concentration","CE Vol Surge","CE Notional",
                "PE OI","PE LTP","PE Volume","PE IV","PE Concentration","PE Vol Surge","PE Notional",
                "Strike PCR","Straddle","Strike Bias","IV Signal"
            ]
            existing_cols = [c for c in display_cols if c in enriched.columns]

            # Filter to nearby by default for readability
            show_all = st.checkbox("Show all strikes (may be slow)", value=False)
            chain_df = enriched if show_all else enriched[
                (enriched["Strike Price"] >= spot_price - nearby_range) &
                (enriched["Strike Price"] <= spot_price + nearby_range)
            ]

            try:
                styled = _style_chain(chain_df[existing_cols])
                st.dataframe(styled, hide_index=True, use_container_width=True, height=520)
            except Exception:
                st.dataframe(chain_df[existing_cols], hide_index=True, use_container_width=True, height=520)

        with adv4:
            nearby = analyzer.get_nearby_strikes(df, spot_price, nearby_range)
            st.dataframe(nearby, hide_index=True, use_container_width=True, height=400)

    # ── Footer ────────────────────────────────
    st.divider()
    st.caption("⚠️ **Disclaimer:** For educational purposes only. Trading involves significant risk.")

    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()


if __name__ == "__main__":
    main()
