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
import calendar

st.set_page_config(page_title="📊 Market Journal", layout="wide", initial_sidebar_state="expanded")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_SCRIP  = "1"
API_BASE_URL   = "https://api.bseindia.com/BseIndiaAPI/api/DerivOptionChain_IV/w"
CACHE_TTL      = 30
NEARBY_RANGE   = 500

st.markdown("""
<style>
  .main > div { padding: 0 1rem; }
  .stApp { background: #0f0f1a; }
  .block-container { max-width: 100%; padding: 0.5rem 1rem; }
  h1, h2, h3 { margin-top: 0 !important; }

  /* Config bar */
  .config-bar { display:flex; align-items:center; gap:10px; padding:10px 14px; background:#1a1a2e; border-radius:12px; margin:8px 0; flex-wrap:wrap; }
  .config-item { display:flex; flex-direction:column; }
  .config-label { font-size:9px; color:#64748B; font-weight:700; letter-spacing:.5px; text-transform:uppercase; }
  .config-value { font-size:13px; color:#fff; font-weight:700; }
  .config-btn { padding:8px 16px; background:#2a2a4e; border:none; border-radius:8px; color:#fff; cursor:pointer; font-size:14px; }

  /* Disclaimer */
  .disclaimer { padding:10px; background:#FEF3C7; border:1px solid #FDE047; border-radius:8px; margin:8px 0; text-align:center; font-size:11px; font-weight:700; color:#92400E; }

  /* Day Range */
  .day-range { padding:20px; background:linear-gradient(135deg,#667eea,#764ba2); border-radius:16px; margin:8px 0; color:white; text-align:center; }
  .day-range-row { display:flex; justify-content:space-around; align-items:center; }
  .day-range-label { font-size:11px; opacity:.7; font-weight:700; }
  .day-range-val { font-size:20px; font-weight:900; }
  .day-range-spot { font-size:24px; font-weight:900; color:#FFD700; }
  .day-range-bar { margin-top:14px; background:rgba(255,255,255,.2); border-radius:8px; height:18px; overflow:hidden; }
  .day-range-fill { height:100%; background:linear-gradient(90deg,#00ff00,#ffd700,#ff0000); border-radius:8px; }
  .day-range-footer { display:flex; justify-content:space-between; margin-top:8px; font-size:11px; opacity:.8; }

  /* Metric card */
  .m-card { padding:10px; background:#1a1a2e; border-radius:10px; text-align:center; }
  .m-label { font-size:9px; color:#64748B; font-weight:700; letter-spacing:.5px; }
  .m-value { font-size:15px; color:#fff; font-weight:900; margin-top:2px; }
  .m-delta { font-size:10px; color:#94A3B8; font-weight:700; }

  /* Sentiment */
  .sent-bar { padding:14px; border-radius:10px; border-left:5px solid; margin:8px 0; }
  .sent-title { font-size:18px; font-weight:900; margin:0; }
  .sent-desc { font-size:12px; color:#ccc; margin:4px 0 0 0; }
  .sent-badges { display:flex; gap:8px; margin-top:8px; }
  .sent-badge { padding:3px 10px; border-radius:6px; font-size:11px; font-weight:800; }

  /* Direction card */
  .dir-card { padding:18px; background:#1a1a2e; border:2px solid; border-radius:16px; margin:12px 0; }
  .dir-title { font-size:12px; color:#aaa; font-weight:700; letter-spacing:1px; text-align:center; margin-bottom:8px; }
  .dir-verdict { font-size:30px; font-weight:900; text-align:center; }
  .dir-action { font-size:15px; color:#FFD700; font-weight:700; text-align:center; margin-bottom:12px; }

  /* Gauge */
  .gauge-wrap { margin:12px 0; }
  .gauge-labels { display:flex; justify-content:space-between; margin-bottom:4px; }
  .gauge-labels span { font-size:9px; color:#888; font-weight:600; }
  .gauge-bar { background:#333; border-radius:14px; height:28px; position:relative; overflow:hidden; }
  .gauge-center { position:absolute; left:50%; top:0; width:2px; height:100%; background:rgba(255,255,255,.3); }
  .gauge-fill { height:100%; opacity:.85; }
  .gauge-score { position:absolute; left:50%; top:50%; transform:translate(-50%,-50%); font-size:12px; font-weight:700; color:#fff; text-shadow:0 0 6px #000; }
  .gauge-counts { text-align:center; margin-top:6px; font-size:12px; color:#aaa; font-weight:700; }

  /* Factor row */
  .factor-row { display:flex; gap:8px; padding:8px 0; border-bottom:1px solid #2a2a2a; }
  .factor-vote { font-size:14px; }
  .factor-name { font-size:12px; font-weight:700; color:#ddd; }
  .factor-reason { font-size:10px; color:#888; margin-top:2px; }

  /* Trade setup */
  .trade-box { padding:14px; border-radius:10px; border-left:5px solid; margin-top:12px; }
  .trade-title { font-size:14px; font-weight:700; margin-bottom:10px; }
  .trade-grid { display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:10px; font-size:12px; }
  .trade-label { font-size:9px; opacity:.5; font-weight:700; letter-spacing:.5px; }
  .trade-val { font-weight:700; }

  /* Card */
  .card { padding:16px; background:#1a1a2e; border-radius:16px; margin:8px 0; }
  .card-title { font-size:16px; font-weight:900; color:#fff; margin-bottom:12px; }

  /* Tabs */
  .tab-row { display:flex; gap:4px; background:#16213e; border-radius:10px; padding:3px; margin-bottom:12px; flex-wrap:wrap; }
  .tab-btn { padding:6px 12px; border-radius:8px; border:none; background:transparent; color:#64748B; font-size:11px; font-weight:700; cursor:pointer; }
  .tab-btn-active { background:#2a2a4e; color:#fff; }

  /* Signal card */
  .sig-card { padding:12px; background:#16213e; border-radius:10px; margin-bottom:8px; border-left:4px solid; }
  .sig-type { font-size:13px; font-weight:800; margin-bottom:6px; }
  .sig-detail { font-size:11px; color:#ccc; font-weight:600; }
  .sig-reason { font-size:10px; color:#888; margin-top:4px; font-style:italic; }

  /* SR */
  .sr-col { flex:1; }
  .sr-title { font-size:13px; font-weight:800; margin-bottom:8px; }
  .sr-item { padding:6px 0; border-bottom:1px solid #2a2a2a; }
  .sr-strike { font-size:16px; font-weight:900; color:#fff; }
  .sr-detail { font-size:10px; color:#888; font-weight:600; }

  /* Max Pain */
  .mp-box { padding:16px; background:#16213e; border-radius:12px; text-align:center; }
  .mp-label { font-size:11px; color:#64748B; font-weight:700; }
  .mp-value { font-size:28px; color:#FFD700; font-weight:900; }
  .mp-delta { font-size:13px; color:#ccc; font-weight:700; margin-top:4px; }

  /* Summary counts */
  .sum-counts { display:flex; gap:12px; flex-wrap:wrap; margin-top:8px; }
  .sum-count { font-size:11px; color:#aaa; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Expiry Helpers
# ─────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_expiry_dates(scrip_cd: str = "1") -> List[str]:
    headers = {"accept":"application/json, text/plain, */*","origin":"https://www.bseindia.com","referer":"https://www.bseindia.com/","user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        session = requests.Session()
        session.get("https://www.bseindia.com", headers=headers, timeout=5)
        urls = [
            f"https://api.bseindia.com/BseIndiaAPI/api/DDlExpiry/w?flag=0&scripcode={scrip_cd}",
            f"https://api.bseindia.com/BseIndiaAPI/api/DefaultData/w?scripcode={scrip_cd}",
            f"https://api.bseindia.com/BseIndiaAPI/api/DerivExpiryDates/w?scripcode={scrip_cd}",
        ]
        expiry_dates = []
        for url in urls:
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
            expiry_dates = _default_expiries()
        expiry_dates = list(set([e.strip() for e in expiry_dates if e and e.strip()]))
        try:
            expiry_dates.sort(key=lambda x: datetime.strptime(x, "%d %b %Y"))
        except Exception:
            pass
        return expiry_dates
    except Exception:
        return _default_expiries()

def _default_expiries() -> List[str]:
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
    except:
        return default

def fmt(v: float) -> str:
    return f"₹{v:,.2f}"

def fmt_cr(v: float) -> str:
    if v >= 1e7:
        return f"₹{v/1e7:.2f} Cr"
    elif v >= 1e5:
        return f"₹{v/1e5:.2f} L"
    return f"₹{v:,.0f}"

# ─────────────────────────────────────────────
# Data Fetching
# ─────────────────────────────────────────────
@st.cache_data(ttl=CACHE_TTL)
def fetch_bse_option_chain(expiry: str, scrip_cd: str, strprice: str = "0"):
    url = f"{API_BASE_URL}?Expiry={expiry.replace(' ', '+')}&scrip_cd={scrip_cd}&strprice={strprice}"
    headers = {"accept":"application/json, text/plain, */*","origin":"https://www.bseindia.com","referer":"https://www.bseindia.com/","user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
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
        df = _process_df(table)
        spot = _extract_spot(data, table, df)
        dh = _extract_val(data, ["High","high","DayHigh","dayHigh"])
        dl = _extract_val(data, ["Low","low","DayLow","dayLow"])
        return df, spot, None, dh, dl
    except requests.exceptions.Timeout:
        return None, None, "⏱️ Timeout.", None, None
    except Exception as e:
        return None, None, f"⚠️ {e}", None, None

def _process_df(table):
    df = pd.DataFrame(table)
    df = df.rename(columns={
        "Strike_Price1":"Strike Price","Open_Interest":"PE OI","C_Open_Interest":"CE OI",
        "Vol_Traded":"PE Volume","C_Vol_Traded":"CE Volume","Last_Trd_Price":"PE LTP",
        "C_Last_Trd_Price":"CE LTP","IV":"PE IV","C_IV":"CE IV"})
    cols = ["Strike Price","CE OI","CE LTP","CE Volume","CE IV","PE OI","PE LTP","PE Volume","PE IV"]
    df = df[cols]
    for col in cols:
        df[col] = df[col].astype(str).str.replace(",","").replace(["","None"," "],"0")
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).round(2)
    df = df[(df["CE OI"]>0)|(df["PE OI"]>0)]
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
# Analysis
# ─────────────────────────────────────────────
def compute_totals(df, spot_price):
    tc = df["CE OI"].sum(); tp = df["PE OI"].sum(); to = tc+tp
    tcv = df["CE Volume"].sum(); tpv = df["PE Volume"].sum(); tv = tcv+tpv
    tcp = (df["CE LTP"]*df["CE OI"]).sum(); tpp = (df["PE LTP"]*df["PE OI"]).sum(); tpr = tcp+tpp
    ai = (df["Strike Price"]-spot_price).abs().idxmin()
    as_ = df.loc[ai,"Strike Price"]; ace = df.loc[ai,"CE LTP"]; ape = df.loc[ai,"PE LTP"]
    sp = ace+ape; bu = as_+sp; bd = as_-sp
    return {"total_ce_oi":tc,"total_pe_oi":tp,"total_oi":to,"total_ce_vol":tcv,"total_pe_vol":tpv,"total_vol":tv,
        "total_ce_premium":tcp,"total_pe_premium":tpp,"total_premium":tpr,"atm_strike":as_,"atm_ce_ltp":ace,"atm_pe_ltp":ape,
        "straddle_price":sp,"breakeven_up":bu,"breakeven_down":bd,"oi_skew":((tp-tc)/to*100) if to>0 else 0,
        "premium_skew":((tpp-tcp)/tpr*100) if tpr>0 else 0,"pcr_oi":round(tp/tc,3) if tc>0 else 0,"pcr_vol":round(tpv/tcv,3) if tcv>0 else 0,
        "max_ce_strike":df.loc[df["CE OI"].idxmax(),"Strike Price"] if tc>0 else None,
        "max_pe_strike":df.loc[df["PE OI"].idxmax(),"Strike Price"] if tp>0 else None}

def calc_pcr(df):
    tc=df["CE OI"].sum(); tp=df["PE OI"].sum(); tcv=df["CE Volume"].sum(); tpv=df["PE Volume"].sum()
    pcr_oi=round(tp/tc,2) if tc>0 else 0; pcr_vol=round(tpv/tcv,2) if tcv>0 else 0
    if pcr_oi>1.2: s,d,c = "📈 Bullish", "Strong Put Writing – Support Building", "green"
    elif pcr_oi<0.8: s,d,c = "📉 Bearish", "Strong Call Writing – Resistance Building", "red"
    else: s,d,c = "⚖️ Neutral", "Balanced Market Conditions", "orange"
    return {"pcr_oi":pcr_oi,"pcr_vol":pcr_vol,"sentiment":s,"description":d,"color":c,
        "total_call_oi":tc,"total_put_oi":tp,"total_call_vol":tcv,"total_put_vol":tpv}

def find_sr(df, spot, n=5):
    sdf=df[df["Strike Price"]<=spot].nlargest(n,"PE OI")
    rdf=df[df["Strike Price"]>=spot].nlargest(n,"CE OI")
    supports = sdf[["Strike Price","PE OI","PE LTP"]].copy()
    resistances = rdf[["Strike Price","CE OI","CE LTP"]].copy()
    ns = sdf["Strike Price"].max() if not sdf.empty else None
    nr = rdf["Strike Price"].min() if not rdf.empty else None
    return supports, resistances, ns, nr

def compute_max_pain(df):
    strikes=df["Strike Price"].unique()
    pain_vals = []
    for s in strikes:
        call_pain = (df[df["Strike Price"]>s]["CE OI"]*(df[df["Strike Price"]>s]["Strike Price"]-s)).sum()
        put_pain = (df[df["Strike Price"]<s]["PE OI"]*(s-df[df["Strike Price"]<s]["Strike Price"])).sum()
        pain_vals.append({"Strike":s,"Pain":call_pain+put_pain})
    pdf=pd.DataFrame(pain_vals)
    mp = pdf.loc[pdf["Pain"].idxmin(),"Strike"] if not pdf.empty else None
    return mp, pdf

def compute_price_bias(spot, t, ns, nr, mp, dh, dl, pcr):
    f = []
    if ns and nr:
        ds=spot-ns; dr=nr-spot; pr=ds/(ds+dr) if (ds+dr)>0 else .5
        if pr<.35: f.append({"name":"S/R Proximity","vote":1,"reason":f"Spot {fmt(spot)} is {ds:.0f} pts above support {fmt(ns)} → bounce zone","bull_val":f"{ns:,.0f}","bear_val":f"{nr:,.0f}"})
        elif pr>.65: f.append({"name":"S/R Proximity","vote":-1,"reason":f"Spot {fmt(spot)} is {dr:.0f} pts below resistance {fmt(nr)} → rejection zone","bull_val":f"{ns:,.0f}","bear_val":f"{nr:,.0f}"})
        else: f.append({"name":"S/R Proximity","vote":0,"reason":f"Mid-range between {fmt(ns)}–{fmt(nr)}","bull_val":f"{ns:,.0f}","bear_val":f"{nr:,.0f}"})
    else: f.append({"name":"S/R Proximity","vote":0,"reason":"No data","bull_val":"—","bear_val":"—"})

    if mp:
        d=spot-mp
        if d>0: f.append({"name":"Max Pain Gravity","vote":-1,"reason":f"Spot ABOVE max pain {fmt(mp)} → gravity pull DOWN","bull_val":"—","bear_val":f"{mp:,.0f}"})
        elif d<0: f.append({"name":"Max Pain Gravity","vote":1,"reason":f"Spot BELOW max pain {fmt(mp)} → gravity pull UP","bull_val":f"{mp:,.0f}","bear_val":"—"})
        else: f.append({"name":"Max Pain Gravity","vote":0,"reason":f"AT max pain {fmt(mp)}","bull_val":"—","bear_val":"—"})
    else: f.append({"name":"Max Pain Gravity","vote":0,"reason":"N/A","bull_val":"—","bear_val":"—"})

    bu=t["breakeven_up"]; bd=t["breakeven_down"]
    if spot>bu: f.append({"name":"Straddle BE","vote":1,"reason":f"Above upper BE {fmt(bu)} → trending up","bull_val":f">{bu:,.0f}","bear_val":f"<{bd:,.0f}"})
    elif spot<bd: f.append({"name":"Straddle BE","vote":-1,"reason":f"Below lower BE {fmt(bd)} → trending down","bull_val":f">{bu:,.0f}","bear_val":f"<{bd:,.0f}"})
    else:
        v=1 if spot>(bu+bd)/2 else -1; s="upper" if spot>(bu+bd)/2 else "lower"
        f.append({"name":"Straddle BE","vote":v,"reason":f"Inside straddle zone, {s} half","bull_val":f">{bu:,.0f}","bear_val":f"<{bd:,.0f}"})

    if dh and dl and dh-dl>0:
        dr=dh-dl; p=(spot-dl)/dr
        if p>=.7: f.append({"name":"Day Range","vote":-1,"reason":f"Top {p*100:.0f}% → overextended","bull_val":"<30%","bear_val":">70%"})
        elif p<=.3: f.append({"name":"Day Range","vote":1,"reason":f"Bottom {p*100:.0f}% → bounce likely","bull_val":"<30%","bear_val":">70%"})
        else: v=1 if p>.5 else -1; f.append({"name":"Day Range","vote":v,"reason":f"{p*100:.0f}% → mild {'upper' if p>.5 else 'lower'}","bull_val":"<30%","bear_val":">70%"})

    mc=t.get("max_ce_strike"); mp2=t.get("max_pe_strike")
    if mc and mp2:
        dr2=mc-spot; ds2=spot-mp2
        if ds2<dr2*.5: f.append({"name":"Max OI Wall","vote":1,"reason":f"PE wall {fmt(mp2)} close → strong floor","bull_val":f"PE {mp2:,.0f}","bear_val":f"CE {mc:,.0f}"})
        elif dr2<ds2*.5: f.append({"name":"Max OI Wall","vote":-1,"reason":f"CE wall {fmt(mc)} close → strong ceiling","bull_val":f"PE {mp2:,.0f}","bear_val":f"CE {mc:,.0f}"})
        else: f.append({"name":"Max OI Wall","vote":0,"reason":f"Balanced CE={fmt(mc)} PE={fmt(mp2)}","bull_val":f"PE {mp2:,.0f}","bear_val":f"CE {mc:,.0f}"})

    if pcr>=1.2: f.append({"name":"PCR Confirmation","vote":1,"reason":f"PCR {pcr:.2f}≥1.2 → bullish","bull_val":"≥1.2","bear_val":"≤0.8"})
    elif pcr<=.8: f.append({"name":"PCR Confirmation","vote":-1,"reason":f"PCR {pcr:.2f}≤0.8 → bearish","bull_val":"≥1.2","bear_val":"≤0.8"})
    else: f.append({"name":"PCR Confirmation","vote":0,"reason":f"PCR {pcr:.2f} neutral","bull_val":"≥1.2","bear_val":"≤0.8"})

    score=sum(x["vote"] for x in f); ms=len(f); bc=sum(1 for x in f if x["vote"]>0); brc=sum(1 for x in f if x["vote"]<0); nc=sum(1 for x in f if x["vote"]==0)
    pct=score/ms*100
    if pct>=50: vd,cl,ac,em="STRONG BUY CALLS","#00cc44","BUY CALLS / SELL PUTS","🚀"
    elif pct>=20: vd,cl,ac,em="MILD BUY CALLS","#66dd88","Consider CALL buying","📈"
    elif pct<=-50: vd,cl,ac,em="STRONG BUY PUTS","#cc2200","BUY PUTS / SELL CALLS","🔻"
    elif pct<=-20: vd,cl,ac,em="MILD BUY PUTS","#dd6666","Consider PUT buying","📉"
    else: vd,cl,ac,em="RANGE / NEUTRAL","#aaaaaa","Straddle / Iron Condor","⚖️"
    return {"factors":f,"score":score,"max_score":ms,"pct":pct,"verdict":vd,"color":cl,"action":ac,"emoji":em,"bull_count":bc,"bear_count":brc,"neut_count":nc}

def enrich_chain(df, spot):
    out=df.copy()
    ai=(out["Strike Price"]-spot).abs().idxmin(); av=out.loc[ai,"Strike Price"]
    def mn(s):
        if abs(s-av)<=av*.003: return "ATM"
        return "ITM-CE/OTM-PE" if s<spot else "OTM-CE/ITM-PE"
    out["Moneyness"]=out["Strike Price"].apply(mn)
    ceq=out["CE OI"].quantile(.8); peq=out["PE OI"].quantile(.8)
    out["CE Conc"]=out["CE OI"].apply(lambda x:"🔴HIGH" if x>=ceq else "")
    out["PE Conc"]=out["PE OI"].apply(lambda x:"🟢HIGH" if x>=peq else "")
    cmv=out["CE Volume"].median(); pmv=out["PE Volume"].median()
    out["CE Surge"]=out["CE Volume"].apply(lambda x:"⚡SURGE" if x>2*cmv and cmv>0 else "")
    out["PE Surge"]=out["PE Volume"].apply(lambda x:"⚡SURGE" if x>2*pmv and pmv>0 else "")
    out["Strike PCR"]=out.apply(lambda r:round(r["PE OI"]/r["CE OI"],2) if r["CE OI"]>0 else np.nan, axis=1)
    out["Straddle"]=(out["CE LTP"]+out["PE LTP"]).round(2)
    out["CE Notional"]=(out["CE LTP"]*out["CE OI"]).round(0)
    out["PE Notional"]=(out["PE LTP"]*out["PE OI"]).round(0)
    def sb(r):
        p=r["Strike PCR"]
        if pd.isna(p): return "—"
        if p>=1.5: return "🟢 Strong Support"
        if p>=1.1: return "🟡 Mild Support"
        if p<=.6: return "🔴 Strong Resistance"
        if p<=.9: return "🟠 Mild Resistance"
        return "⚪ Neutral"
    out["Strike Bias"]=out.apply(sb,axis=1)
    def ivs(r):
        if r["CE IV"]==0 or r["PE IV"]==0: return "—"
        rat=r["PE IV"]/r["CE IV"]
        if rat>1.3: return "⬇️ Fear (High PE IV)"
        if rat<.7: return "⬆️ Greed (High CE IV)"
        return "➡️ Balanced"
    out["IV Signal"]=out.apply(ivs,axis=1)
    out["Dist Spot"]=(out["Strike Price"]-spot).round(1)
    out["Dist %"]=((out["Strike Price"]-spot)/spot*100).round(2)
    return out

def gen_signals(df, spot, pcr, ns, nr):
    s={"call_buy":[],"put_buy":[],"call_sell":[],"put_sell":[],"market_bias":"","strategy":""}
    p=pcr["pcr_oi"]
    if p>1.3: s["market_bias"],s["strategy"]="Strongly Bullish","Buy Calls/Sell Puts"
    elif p>1.0: s["market_bias"],s["strategy"]="Moderately Bullish","Buy ATM/OTM Calls"
    elif p<.7: s["market_bias"],s["strategy"]="Strongly Bearish","Buy Puts/Sell Calls"
    elif p<.9: s["market_bias"],s["strategy"]="Moderately Bearish","Buy ATM/OTM Puts"
    else: s["market_bias"],s["strategy"]="Neutral","Iron Condor/Straddle"
    ai=(df["Strike Price"]-spot).abs().idxmin(); a=df.loc[ai,"Strike Price"]
    if p>=1.0:
        s["call_buy"].append({"strike":a,"type":"ATM Call","target":nr or spot+500,"stop_loss":ns or spot-200,"reason":"ATM call for bullish move"})
        otm=df[df["Strike Price"]>spot].nsmallest(2,"Strike Price")
        if not otm.empty: s["call_buy"].append({"strike":otm.iloc[0]["Strike Price"],"type":"OTM Call","target":nr or spot+700,"stop_loss":spot-100,"reason":"OTM call – aggressive bullish"})
    if p<=.9:
        s["put_buy"].append({"strike":a,"type":"ATM Put","target":ns or spot-500,"stop_loss":nr or spot+200,"reason":"ATM put for bearish move"})
        otm=df[df["Strike Price"]<spot].nlargest(2,"Strike Price")
        if not otm.empty: s["put_buy"].append({"strike":otm.iloc[0]["Strike Price"],"type":"OTM Put","target":ns or spot-700,"stop_loss":spot+100,"reason":"OTM put – aggressive bearish"})
    if p>=1.2:
        ss=df[df["Strike Price"]<spot].nlargest(3,"PE OI")
        if not ss.empty: s["put_sell"].append({"strike":ss.iloc[0]["Strike Price"],"type":"OTM Put Sell","target":"Premium collection","stop_loss":ss.iloc[0]["Strike Price"]-200,"reason":f"Strong support at {ss.iloc[0]['Strike Price']:,.0f} – high PE OI"})
    if p<=.8:
        sr=df[df["Strike Price"]>spot].nlargest(3,"CE OI")
        if not sr.empty: s["call_sell"].append({"strike":sr.iloc[0]["Strike Price"],"type":"OTM Call Sell","target":"Premium collection","stop_loss":sr.iloc[0]["Strike Price"]+200,"reason":f"Strong resistance at {sr.iloc[0]['Strike Price']:,.0f} – high CE OI"})
    return s

# ─────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────
def create_oi_chart(df, spot):
    fig=make_subplots(rows=2,cols=1,subplot_titles=("Open Interest","Volume"),vertical_spacing=.12,row_heights=[.6,.4])
    fig.add_trace(go.Bar(name="Call OI",x=df["Strike Price"],y=df["CE OI"],marker_color="rgba(255,99,71,0.7)"),row=1,col=1)
    fig.add_trace(go.Bar(name="Put OI",x=df["Strike Price"],y=df["PE OI"],marker_color="rgba(60,179,113,0.7)"),row=1,col=1)
    fig.add_trace(go.Bar(name="Call Vol",x=df["Strike Price"],y=df["CE Volume"],marker_color="rgba(255,140,0,0.7)",showlegend=False),row=2,col=1)
    fig.add_trace(go.Bar(name="Put Vol",x=df["Strike Price"],y=df["PE Volume"],marker_color="rgba(30,144,255,0.7)",showlegend=False),row=2,col=1)
    fig.add_vline(x=spot,line_dash="dash",line_color="yellow",annotation_text=f"Spot {spot:.0f}",row=1,col=1)
    fig.add_vline(x=spot,line_dash="dash",line_color="yellow",row=2,col=1)
    fig.update_layout(height=500,hovermode="x unified",barmode="group",template="plotly_dark")
    return fig

def create_iv_chart(df):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df["Strike Price"],y=df["CE IV"],mode="lines+markers",name="Call IV",line=dict(color="red",width=2)))
    fig.add_trace(go.Scatter(x=df["Strike Price"],y=df["PE IV"],mode="lines+markers",name="Put IV",line=dict(color="green",width=2)))
    fig.update_layout(title="IV Smile",xaxis_title="Strike",yaxis_title="IV (%)",height=350,hovermode="x unified",template="plotly_dark")
    return fig

def create_pain_chart(pdf, spot):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=pdf["Strike"],y=pdf["Pain"],mode="lines",fill="tozeroy",name="Pain",line=dict(color="purple",width=2)))
    fig.add_vline(x=spot,line_dash="dash",line_color="yellow",annotation_text="Spot")
    fig.update_layout(title="Max Pain Distribution",xaxis_title="Strike",yaxis_title="Pain",height=300,template="plotly_dark")
    return fig

def create_donut(t):
    fig=make_subplots(rows=1,cols=2,subplot_titles=("OI Distribution","Premium Distribution"),specs=[[{"type":"domain"},{"type":"domain"}]])
    fig.add_trace(go.Pie(labels=["Call OI","Put OI"],values=[t["total_ce_oi"],t["total_pe_oi"]],hole=.55,marker_colors=["#ff6347","#3cb371"],textinfo="label+percent"),row=1,col=1)
    fig.add_trace(go.Pie(labels=["Call Premium","Put Premium"],values=[t["total_ce_premium"],t["total_pe_premium"]],hole=.55,marker_colors=["#ffa07a","#90ee90"],textinfo="label+percent"),row=1,col=2)
    fig.update_layout(height=300,template="plotly_dark",showlegend=True)
    return fig

def create_straddle_curve(enr, spot):
    nb=enr[(enr["Strike Price"]>=spot-NEARBY_RANGE)&(enr["Strike Price"]<=spot+NEARBY_RANGE)]
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=nb["Strike Price"],y=nb["Straddle"],mode="lines+markers",name="Straddle",fill="tozeroy",line=dict(color="#ff9f40",width=2)))
    fig.add_vline(x=spot,line_dash="dash",line_color="yellow",annotation_text=f"ATM {spot:.0f}")
    fig.update_layout(title="Straddle Price Curve",height=300,hovermode="x unified",template="plotly_dark")
    return fig

def create_pcr_heatmap(enr, spot):
    nb=enr[(enr["Strike Price"]>=spot-NEARBY_RANGE)&(enr["Strike Price"]<=spot+NEARBY_RANGE)].copy()
    nb["Strike PCR"]=nb["Strike PCR"].fillna(0)
    fig=go.Figure(go.Bar(x=nb["Strike Price"],y=nb["Strike PCR"],marker=dict(color=nb["Strike PCR"],colorscale=[[0,"#ff4040"],[.4,"#ff9933"],[.5,"#888888"],[.6,"#33cc66"],[1,"#00aa00"]],cmin=0,cmax=2,colorbar=dict(title="PCR")),name="PCR"))
    fig.add_hline(y=1,line_dash="dash",line_color="white",annotation_text="PCR=1")
    fig.add_vline(x=spot,line_dash="dash",line_color="yellow")
    fig.update_layout(title="Per-Strike PCR",height=300,hovermode="x unified",template="plotly_dark")
    return fig

# ─────────────────────────────────────────────
# Market Journal UI Renderers
# ─────────────────────────────────────────────
def render_config_bar(expiry_dates, scrip_cd, expiry):
    st.markdown(f"""
    <div class="config-bar">
      <div class="config-item">
        <span class="config-label">Expiry</span>
        <span class="config-value">{expiry}</span>
      </div>
      <div class="config-item">
        <span class="config-label">Scrip</span>
        <span class="config-value">{scrip_cd}</span>
      </div>
      <div style="margin-left:auto">
        <button class="config-btn" onclick="window.location.reload()">🔄</button>
      </div>
    </div>
    """, unsafe_allow_html=True)

def render_disclaimer():
    st.markdown('<div class="disclaimer">⚠️ Educational Only. Not financial advice.</div>', unsafe_allow_html=True)

def render_day_range(spot, dh, dl):
    if not dh or not dl: return
    dr=dh-dl; rp=dr/dl*100; pr=((spot-dl)/dr*100) if dr>0 else 50
    st.markdown(f"""
    <div class="day-range">
      <div class="day-range-row">
        <div><div class="day-range-label">Day Low</div><div class="day-range-val">{fmt(dl)}</div></div>
        <div><div class="day-range-label">Spot</div><div class="day-range-spot">{fmt(spot)}</div><div style="font-size:10px;opacity:.6">{pr:.1f}% in range</div></div>
        <div><div class="day-range-label">Day High</div><div class="day-range-val">{fmt(dh)}</div></div>
      </div>
      <div class="day-range-bar"><div class="day-range-fill" style="width:{min(pr,100)}%"></div></div>
      <div class="day-range-footer"><span>Range: {fmt(dr)}</span><span>Movement: {rp:.2f}%</span></div>
    </div>
    """, unsafe_allow_html=True)

def render_metrics(spot, dh, dl, pcr_oi, ns, nr):
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: st.markdown(f'<div class="m-card"><div class="m-label">Spot</div><div class="m-value">{fmt(spot)}</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="m-card"><div class="m-label">Day High</div><div class="m-value">{fmt(dh) if dh else "N/A"}</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="m-card"><div class="m-label">Day Low</div><div class="m-value">{fmt(dl) if dl else "N/A"}</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="m-card"><div class="m-label">PCR OI</div><div class="m-value">{pcr_oi}</div></div>', unsafe_allow_html=True)
    with c5:
        delta = f"-{spot-ns:.0f}" if ns else ""
        st.markdown(f'<div class="m-card"><div class="m-label">Support</div><div class="m-value">{fmt(ns) if ns else "N/A"}</div><div class="m-delta">{delta}</div></div>', unsafe_allow_html=True)
    with c6:
        delta = f"+{nr-spot:.0f}" if nr else ""
        st.markdown(f'<div class="m-card"><div class="m-label">Resistance</div><div class="m-value">{fmt(nr) if nr else "N/A"}</div><div class="m-delta">{delta}</div></div>', unsafe_allow_html=True)

def render_sentiment(pcr_data):
    clr = {"green":"#10B981","red":"#EF4444","orange":"#F59E0B"}.get(pcr_data["color"],"#888")
    st.markdown(f"""
    <div class="sent-bar" style="border-left-color:{clr};background-color:{clr}15">
      <div class="sent-title" style="color:{clr}">{pcr_data["sentiment"]}</div>
      <div class="sent-desc">{pcr_data["description"]}</div>
      <div class="sent-badges">
        <div class="sent-badge" style="background:#DC262620;color:#DC2626">CE: {pcr_data["total_call_oi"]:,.0f}</div>
        <div class="sent-badge" style="background:#10B98120;color:#10B981">PE: {pcr_data["total_put_oi"]:,.0f}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

def render_direction_panel(bias, spot, t, ns, nr):
    gf = max(2, min(98, 50 + bias["pct"]/2))
    is_bull = bias["pct"] >= 0
    rows_html = ""
    for f in bias["factors"]:
        icon = {1:"🟢",-1:"🔴",0:"⚪"}[f["vote"]]
        label = {1:"BULLISH",-1:"BEARISH",0:"NEUTRAL"}[f["vote"]]
        lc = {1:"#66dd88",-1:"#dd6666",0:"#aaaaaa"}[f["vote"]]
        rows_html += f'<div class="factor-row"><div class="factor-vote">{icon}</div><div style="flex:1"><div class="factor-name">{f["name"]} <span style="font-size:10px;color:{lc}">({label})</span></div><div class="factor-reason">{f["reason"]}</div></div></div>'

    fill_style = ""
    if bias["pct"] >= 0:
        fill_style = f'<div class="gauge-fill" style="position:absolute;left:50%;width:{abs(gf-50)}%;background:{bias["color"]}"></div>'
    else:
        fill_style = f'<div class="gauge-fill" style="position:absolute;right:50%;width:{abs(gf-50)}%;background:{bias["color"]}"></div>'

    st.markdown(f"""
    <div class="dir-card" style="border-color:{bias['color']}">
      <div class="dir-title">🧭 Price-Based Directional Analysis</div>
      <div class="dir-verdict" style="color:{bias['color']}">{bias['emoji']} {bias['verdict']}</div>
      <div class="dir-action">{bias['action']}</div>
      <div class="gauge-wrap">
        <div class="gauge-labels"><span>🔴 BEAR</span><span>⚪ NEUTRAL</span><span>🟢 BULL</span></div>
        <div class="gauge-bar">
          <div class="gauge-center"></div>
          {fill_style}
          <div class="gauge-score">Score: {bias['score']:+d}/{bias['max_score']}</div>
        </div>
        <div class="gauge-counts">🟢 {bias['bull_count']} &nbsp;🔴 {bias['bear_count']} &nbsp;⚪ {bias['neut_count']}</div>
      </div>
      {rows_html}
    </div>
    """, unsafe_allow_html=True)

    a=t["atm_strike"]; sp=t["straddle_price"]; bu=t["breakeven_up"]; bd=t["breakeven_down"]
    if bias["pct"]>=20:
        rb=f"BUY CALL @ {a:,.0f} (ATM)"; rs=f"SELL PUT @ {nr:,.0f}" if nr else "SELL OTM PUT"
        tg=fmt(nr) if nr else "next resistance"; sl=fmt(ns-100) if ns else "below support"; bc="#0f4f2f"; bo="#00cc44"
    elif bias["pct"]<=-20:
        rb=f"BUY PUT @ {a:,.0f} (ATM)"; rs=f"SELL CALL @ {nr:,.0f}" if nr else "SELL OTM CALL"
        tg=fmt(ns) if ns else "next support"; sl=fmt(nr+100) if nr else "above resistance"; bc="#4f0f0f"; bo="#cc2200"
    else:
        rb=f"BUY STRADDLE @ {a:,.0f}"; rs=f"SELL STRANGLE CE {nr:,.0f}/PE {ns:,.0f}" if nr and ns else "IRON CONDOR"
        tg=f"Premium within {fmt(bd)}–{fmt(bu)}"; sl="Exit outside straddle zone"; bc="#2a2a2a"; bo="#888888"

    st.markdown(f"""
    <div class="trade-box" style="background:{bc};border-left-color:{bo}">
      <div class="trade-title" style="color:{bo}">🎯 Recommended Trade Setup</div>
      <div class="trade-grid">
        <div><div class="trade-label">PRIMARY</div><div class="trade-val" style="color:#FFD700">{rb}</div></div>
        <div><div class="trade-label">HEDGE</div><div class="trade-val" style="color:#ccc">{rs}</div></div>
        <div><div class="trade-label">TARGET</div><div class="trade-val" style="color:#66ff99">{tg}</div></div>
        <div><div class="trade-label">STOP LOSS</div><div class="trade-val" style="color:#ff6666">{sl}</div></div>
      </div>
      <div style="margin-top:8px;font-size:10px;opacity:.6">ATM: {fmt(a)} · Straddle: {fmt(sp)} · BE: {fmt(bd)} ↔ {fmt(bu)}</div>
    </div>
    """, unsafe_allow_html=True)

def render_totals(t, spot):
    st.markdown('<div class="card"><div class="card-title">📦 OI · Volume · Premium Summary</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="m-card"><div class="m-label">Total OI</div><div class="m-value">{t["total_oi"]:,.0f}</div><div style="display:flex;justify-content:space-around;margin-top:6px;font-size:11px;font-weight:700"><span style="color:#ff6347">CE: {t["total_ce_oi"]:,.0f}</span><span style="color:#3cb371">PE: {t["total_pe_oi"]:,.0f}</span></div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="m-card"><div class="m-label">Total Volume</div><div class="m-value">{t["total_vol"]:,.0f}</div><div style="display:flex;justify-content:space-around;margin-top:6px;font-size:11px;font-weight:700"><span style="color:#ff6347">CE: {t["total_ce_vol"]:,.0f}</span><span style="color:#3cb371">PE: {t["total_pe_vol"]:,.0f}</span></div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="m-card"><div class="m-label">Total Premium</div><div class="m-value">{fmt_cr(t["total_premium"])}</div><div style="display:flex;justify-content:space-around;margin-top:6px;font-size:11px;font-weight:700"><span style="color:#ff6347">CE: {fmt_cr(t["total_ce_premium"])}</span><span style="color:#3cb371">PE: {fmt_cr(t["total_pe_premium"])}</span></div></div>', unsafe_allow_html=True)
    c4,c5,c6,c7 = st.columns(4)
    with c4: st.markdown(f'<div class="m-card"><div class="m-label">ATM Strike</div><div class="m-value" style="font-size:14px">{fmt(t["atm_strike"])}</div><div style="font-size:10px;opacity:.7">CE: {fmt(t["atm_ce_ltp"])} | PE: {fmt(t["atm_pe_ltp"])}</div></div>', unsafe_allow_html=True)
    with c5: st.markdown(f'<div class="m-card"><div class="m-label">Straddle</div><div class="m-value" style="font-size:14px;color:#ffd700">{fmt(t["straddle_price"])}</div><div style="font-size:10px;opacity:.7">{(t["straddle_price"]/t["atm_strike"]*100):.2f}% of spot</div></div>', unsafe_allow_html=True)
    with c6: st.markdown(f'<div class="m-card"><div class="m-label">BE Up</div><div class="m-value" style="font-size:14px;color:#3cb371">{fmt(t["breakeven_up"])}</div><div style="font-size:10px;opacity:.7">+{fmt(t["breakeven_up"]-spot)}</div></div>', unsafe_allow_html=True)
    with c7: st.markdown(f'<div class="m-card"><div class="m-label">BE Down</div><div class="m-value" style="font-size:14px;color:#ff6347">{fmt(t["breakeven_down"])}</div><div style="font-size:10px;opacity:.7">-{fmt(spot-t["breakeven_down"])}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="display:flex;gap:8px;margin-top:8px"><div style="flex:1;background:#16213e;padding:10px;border-radius:8px"><span style="font-size:12px;font-weight:700;color:#ff6347">🔴 Max CE OI: {fmt(t["max_ce_strike"])}</span></div><div style="flex:1;background:#16213e;padding:10px;border-radius:8px"><span style="font-size:12px;font-weight:700;color:#3cb371">🟢 Max PE OI: {fmt(t["max_pe_strike"])}</span></div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def render_signals(signals, signal_tab, set_signal_tab):
    st.markdown('<div class="card"><div class="card-title">🎯 Trading Signals</div>', unsafe_allow_html=True)
    tabs = [("CB","📈 Call Buy"),("PB","📉 Put Buy"),("PS","💰 Put Sell"),("CS","💰 Call Sell")]
    cols = st.columns(len(tabs))
    for i, (key, label) in enumerate(tabs):
        with cols[i]:
            active = "tab-btn-active" if signal_tab == key else ""
            if st.button(label, key=f"sig_{key}", use_container_width=True, type="secondary" if signal_tab!=key else "primary"):
                set_signal_tab(key)

    sig_map = {"CB":signals["call_buy"],"PB":signals["put_buy"],"PS":signals["put_sell"],"CS":signals["call_sell"]}
    color_map = {"CB":"#059669","PB":"#DC2626","PS":"#065F46","CS":"#7F1D1D"}
    items = sig_map[signal_tab]
    if not items:
        st.warning("No setups at current market conditions.")
    else:
        for s in items:
            tgt = fmt(s["target"]) if isinstance(s["target"],(int,float)) else s["target"]
            st.markdown(f"""
            <div class="sig-card" style="border-left-color:{color_map[signal_tab]}">
              <div class="sig-type" style="color:{color_map[signal_tab]}">{s['type']} – Strike: {fmt(s['strike'])}</div>
              <div class="sig-detail">🎯 Target: {tgt}</div>
              <div class="sig-detail">🛑 Stop Loss: {fmt(s['stop_loss'])}</div>
              <div class="sig-reason">{s['reason']}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def render_sr(supports, resistances, ns, nr):
    st.markdown('<div class="card"><div class="card-title">🔴🟢 Support & Resistance</div>', unsafe_allow_html=True)
    sc, rc = st.columns(2)
    with sc:
        st.markdown('<div class="sr-title" style="color:#10B981">🟢 Support Levels</div>', unsafe_allow_html=True)
        for _, r in supports.iterrows():
            st.markdown(f'<div class="sr-item"><div class="sr-strike">{fmt(r["Strike Price"])}</div><div class="sr-detail">PE OI: {r["PE OI"]:,.0f} · PE LTP: {fmt(r["PE LTP"])}</div></div>', unsafe_allow_html=True)
        if not ns: st.markdown('<div class="sr-detail">No support</div>', unsafe_allow_html=True)
    with rc:
        st.markdown('<div class="sr-title" style="color:#EF4444">🔴 Resistance Levels</div>', unsafe_allow_html=True)
        for _, r in resistances.iterrows():
            st.markdown(f'<div class="sr-item"><div class="sr-strike">{fmt(r["Strike Price"])}</div><div class="sr-detail">CE OI: {r["CE OI"]:,.0f} · CE LTP: {fmt(r["CE LTP"])}</div></div>', unsafe_allow_html=True)
        if not nr: st.markdown('<div class="sr-detail">No resistance</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def render_advanced(spot, df, enriched, signals, bias, mp, pain_df, adv_tab, set_adv_tab):
    st.markdown('<div class="card"><div class="card-title">🎯 Advanced Analytics</div>', unsafe_allow_html=True)
    tabs = [("signals","🎯 Signals"),("pain","🎯 Max Pain"),("iv","📈 IV Smile"),("chain","📋 Full Chain"),("nearby","📍 Nearby")]
    cols = st.columns(len(tabs))
    for i, (key, label) in enumerate(tabs):
        with cols[i]:
            if st.button(label, key=f"adv_{key}", use_container_width=True, type="primary" if adv_tab==key else "secondary"):
                set_adv_tab(key)

    if adv_tab == "pain":
        if mp:
            mc1, mc2 = st.columns([1,2])
            with mc1:
                st.markdown(f'<div class="mp-box"><div class="mp-label">🎯 Max Pain</div><div class="mp-value">{fmt(mp)}</div><div class="mp-delta">{"+" if spot-mp>=0 else ""}{(spot-mp):.0f} vs spot</div></div>', unsafe_allow_html=True)
            with mc2:
                if pain_df is not None and not pain_df.empty:
                    st.plotly_chart(create_pain_chart(pain_df, spot), use_container_width=True)
        else:
            st.info("Max pain not available")

    elif adv_tab == "iv":
        st.plotly_chart(create_iv_chart(df), use_container_width=True)

    elif adv_tab == "chain":
        display_cols = ["Strike Price","Dist Spot","Dist %","Moneyness","CE OI","CE LTP","CE Volume","CE IV","CE Conc","CE Surge","CE Notional",
            "PE OI","PE LTP","PE Volume","PE IV","PE Conc","PE Surge","PE Notional","Strike PCR","Straddle","Strike Bias","IV Signal"]
        existing = [c for c in display_cols if c in enriched.columns]
        show_all = st.checkbox("Show all strikes", False)
        chain = enriched if show_all else enriched[(enriched["Strike Price"]>=spot-NEARBY_RANGE)&(enriched["Strike Price"]<=spot+NEARBY_RANGE)]
        try:
            styled = chain[existing].style.format({
                "Strike Price":lambda x:f"{x:,.0f}","CE OI":lambda x:f"{x:,.0f}","PE OI":lambda x:f"{x:,.0f}",
                "CE LTP":lambda x:f"₹{x:.2f}","PE LTP":lambda x:f"₹{x:.2f}",
                "CE Volume":lambda x:f"{x:,.0f}","PE Volume":lambda x:f"{x:,.0f}",
                "CE Notional":fmt_cr,"PE Notional":fmt_cr,"Straddle":lambda x:f"₹{x:.2f}",
                "Strike PCR":lambda x:f"{x:.2f}" if not pd.isna(x) else "—","Dist %":lambda x:f"{x:+.2f}%",
            }, na_rep="—")
            st.dataframe(styled, hide_index=True, use_container_width=True, height=500)
        except:
            st.dataframe(chain[existing], hide_index=True, use_container_width=True, height=500)

    elif adv_tab == "nearby":
        nearby = df[(df["Strike Price"]>=spot-NEARBY_RANGE)&(df["Strike Price"]<=spot+NEARBY_RANGE)].copy()
        nearby["OI Diff"] = nearby["PE OI"]-nearby["CE OI"]
        nearby["OI Ratio"] = nearby.apply(lambda r:round(r["PE OI"]/r["CE OI"],2) if r["CE OI"]>0 else 0, axis=1)
        st.dataframe(nearby, hide_index=True, use_container_width=True, height=400)

    elif adv_tab == "signals":
        st.markdown(f'<div style="background:#16213e;padding:12px;border-radius:8px;margin-bottom:6px"><div style="font-size:10px;color:#64748B;font-weight:700">Market Bias</div><div style="font-size:16px;color:{bias["color"]};font-weight:900">{signals["market_bias"]}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div style="background:#16213e;padding:12px;border-radius:8px;margin-bottom:6px"><div style="font-size:10px;color:#64748B;font-weight:700">Strategy</div><div style="font-size:16px;color:#fff;font-weight:900">{signals["strategy"]}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sum-counts"><span class="sum-count">📈 Call Buys: {len(signals["call_buy"])}</span><span class="sum-count">📉 Put Buys: {len(signals["put_buy"])}</span><span class="sum-count">💰 Put Sells: {len(signals["put_sell"])}</span><span class="sum-count">💰 Call Sells: {len(signals["call_sell"])}</span></div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    st.title("📊 Market Journal")
    st.markdown("**Real-time OI · Volume · Premium · Decision Analytics**")

    # ── Sidebar ──
    with st.sidebar:
        st.header("⚙️ Configuration")
        scrip_cd = st.text_input("Scrip Code", DEFAULT_SCRIP)
        st.divider()
        st.subheader("📅 Expiry Date")
        mode = st.radio("Input:", ["Dropdown", "Manual"])
        if mode == "Dropdown":
            with st.spinner("Loading..."):
                dates = fetch_expiry_dates(scrip_cd)
            if dates:
                expiry = st.selectbox("Select Expiry", dates, index=0)
            else:
                st.error("No dates loaded")
                expiry = st.text_input("Enter manually", datetime.now().strftime("%d %b %Y"))
        else:
            expiry = st.text_input("DD MMM YYYY", datetime.now().strftime("%d %b %Y"))
        st.divider()
        manual_spot = st.checkbox("Override Spot")
        custom_spot = st.number_input("Spot Price", min_value=0.0, value=50000.0, step=100.0, disabled=not manual_spot)
        st.divider()
        show_adv = st.checkbox("Advanced Analytics", True)
        sr_levels = st.slider("S/R Levels", 3, 10, 5)
        nearby_rng = st.slider("Nearby Range", 100, 2000, NEARBY_RANGE, step=100)
        st.divider()
        st.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')}")
        if st.button("🔄 Refresh"):
            st.cache_data.clear()
            st.rerun()

    # ── Fetch Data ──
    with st.spinner("Fetching option chain..."):
        df, spot_price, error, day_high, day_low = fetch_bse_option_chain(expiry, scrip_cd)

    if error:
        st.error(f"⚠️ {error}")
        st.stop()
    if df is None or df.empty:
        st.warning(f"No data for {expiry}")
        st.stop()

    if manual_spot and custom_spot:
        spot_price = custom_spot

    # ── Compute Analysis ──
    totals = compute_totals(df, spot_price)
    enriched = enrich_chain(df, spot_price)
    pcr_data = calc_pcr(df)
    supports, resistances, ns, nr = find_sr(df, spot_price, sr_levels)
    max_pain, pain_df = compute_max_pain(df)
    price_bias = compute_price_bias(spot_price, totals, ns, nr, max_pain, day_high, day_low, totals["pcr_oi"])
    signals = gen_signals(df, spot_price, pcr_data, ns, nr)

    # ⚠️ Disclaimer
    render_disclaimer()

    # 📊 Day Range
    render_day_range(spot_price, day_high, day_low)

    # 💰 Key Metrics
    render_metrics(spot_price, day_high, day_low, totals["pcr_oi"], ns, nr)

    # 📈 Sentiment
    render_sentiment(pcr_data)

    st.divider()

    # 🧭 Direction
    render_direction_panel(price_bias, spot_price, totals, ns, nr)

    st.divider()

    # 📦 Totals + Donuts
    render_totals(totals, spot_price)
    st.plotly_chart(create_donut(totals), use_container_width=True)

    st.divider()

    # 📊 OI & Volume
    st.subheader("📊 OI & Volume")
    st.plotly_chart(create_oi_chart(df, spot_price), use_container_width=True)

    st.subheader("📐 Straddle Curve")
    st.plotly_chart(create_straddle_curve(enriched, spot_price), use_container_width=True)

    st.subheader("🌡️ PCR Heatmap")
    st.plotly_chart(create_pcr_heatmap(enriched, spot_price), use_container_width=True)

    st.divider()

    # 🎯 Trading Signals (sidebar controls render_signals with session state)
    if 'signal_tab' not in st.session_state:
        st.session_state.signal_tab = "CB"
    render_signals(signals, st.session_state.signal_tab, lambda k: st.session_state.__setitem__("signal_tab", k))

    st.divider()

    # 🔴🟢 SR
    render_sr(supports, resistances, ns, nr)

    st.divider()

    # 🎯 Advanced
    if show_adv:
        if 'adv_tab' not in st.session_state:
            st.session_state.adv_tab = "signals"
        render_advanced(spot_price, df, enriched, signals, price_bias, max_pain, pain_df,
                        st.session_state.adv_tab, lambda k: st.session_state.__setitem__("adv_tab", k))

    st.divider()
    st.caption("⚠️ **Disclaimer:** Educational purposes only. Trading involves significant risk.")

if __name__ == "__main__":
    main()
