"""
QuantEdge PRO — Dashboard Cuantitativo
=======================================
Software de análisis cuantitativo para traders retail.
El cliente introduce su propia Gemini API Key.

Fuentes de datos:
  • Yahoo Finance — precios H4
  • FRED (Federal Reserve) — datos macroeconómicos gratuitos, sin key
  • Gemini AI — análisis IA (key del cliente)

Pestañas:
  ⬡  RESUMEN EJECUTIVO
  ①  MERCADO — Z-Diff, Markov, Monte Carlo
  ②  VOLATILIDAD — ATR, cono, regímenes
  ③  VOLUMEN — Perfil, VWAP, Delta
  ④  MACRO CUANTITATIVA — tipos reales, curva, spreads
  ⑤  ANÁLISIS IA — Gemini + contexto completo
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from datetime import datetime, timedelta
import requests, json, re, warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuantEdge PRO",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── THEME ────────────────────────────────────────────────────────────────────
BG     = "#04070d"
S0     = "#060b13"
S1     = "#0a1019"
S2     = "#0e1520"
BORDER = "#1a2d40"
GREEN  = "#00e676"
RED    = "#ff1744"
BLUE   = "#0090ff"
YELLOW = "#ffd600"
ORANGE = "#ff9100"
CYAN   = "#00e5ff"
PURPLE = "#d500f9"
MUTED  = "#4a6080"
TEXT   = "#cdd9e5"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&family=Rajdhani:wght@600;700&display=swap');
html,body,[class*="css"]{{font-family:'Inter',sans-serif;background:{BG};color:{TEXT};}}
.stTabs [data-baseweb="tab-list"]{{gap:2px;background:{S1};padding:4px;border-radius:6px;border:1px solid {BORDER};}}
.stTabs [data-baseweb="tab"]{{background:transparent;color:{MUTED};font-family:'Rajdhani',sans-serif;font-size:13px;font-weight:600;letter-spacing:1.5px;padding:8px 18px;border-radius:4px;transition:all .2s;}}
.stTabs [aria-selected="true"]{{background:{BG};color:{CYAN};border-bottom:2px solid {CYAN};}}
.stTabs [data-baseweb="tab"]:hover{{color:{TEXT};}}
.kpi{{background:{S1};border:1px solid {BORDER};border-radius:6px;padding:16px 18px;margin-bottom:8px;transition:border-color .2s;}}
.kpi:hover{{border-color:{MUTED};}}
.kpi-lbl{{font-size:9px;letter-spacing:3px;text-transform:uppercase;color:{MUTED};margin-bottom:6px;font-family:'JetBrains Mono',monospace;}}
.kpi-val{{font-family:'Rajdhani',sans-serif;font-size:26px;font-weight:700;line-height:1;}}
.kpi-sub{{font-size:10px;color:{MUTED};margin-top:4px;font-family:'JetBrains Mono',monospace;}}
.card{{background:{S1};border:1px solid {BORDER};border-radius:6px;padding:20px 24px;margin-bottom:12px;}}
.signal-box{{border-radius:6px;padding:16px 20px;margin-bottom:10px;}}
.entry-box{{background:{S0};border:1px solid {BORDER};border-left:4px solid {CYAN};border-radius:6px;padding:16px 20px;margin:10px 0;}}
.metric-row{{display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid {BORDER};}}
.metric-row:last-child{{border-bottom:none;}}
div[data-testid="stSidebar"]{{background:{S0};border-right:1px solid {BORDER};}}
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
APP_NAME    = "QuantEdge PRO"
APP_VERSION = "1.0"
H4_PER_DAY  = 6
TRADING_DAYS= 252
FRED_BASE   = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="

QUICK_MAP = {
    "EUR/USD":       ("EURUSD=X", "forex"),
    "GBP/USD":       ("GBPUSD=X", "forex"),
    "USD/JPY":       ("USDJPY=X", "forex"),
    "XAU/USD 🥇":    ("GC=F",     "commodity"),
    "S&P 500 🔄":    ("ES=F",     "index"),
    "NASDAQ 🔄":     ("NQ=F",     "index"),
    "DOW JONES 🔄":  ("YM=F",     "index"),
    "DAX 🔄":        ("FDAX=F",   "index"),
    "CRUDE OIL 🔄":  ("CL=F",     "commodity"),
    "BTC/USD":       ("BTC-USD",  "crypto"),
    "— Manual —":    ("",         "forex"),
}

FUTURES_NOTE = {
    "ES=F":"S&P 500 E-mini","NQ=F":"Nasdaq 100 E-mini",
    "YM=F":"Dow Jones E-mini","FDAX=F":"DAX Futures","CL=F":"WTI Crude Oil",
}

# ─── SESSION STATE ────────────────────────────────────────────────────────────
for k in ["df","results","macro_data","ai_analysis"]:
    if k not in st.session_state:
        st.session_state[k] = None

# ══════════════════════════════════════════════════════════════════════════════
#  QUANTITATIVE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def calc_order_flow(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    df["tp"]      = (df["High"] + df["Low"] + df["Close"]) / 3
    df["tp_prev"] = df["tp"].shift(1)
    vol = df["Volume"].fillna(0)
    if vol.sum() == 0 or vol.nunique() <= 3:
        rng     = df["High"] - df["Low"]
        eff_vol = (rng / rng.mean() * df["tp"].mean() * 10000).fillna(1.0)
    else:
        eff_vol = vol.replace(0, np.nan).ffill().fillna(1.0)
    df["raw_mf"] = np.where(
        df["tp"] > df["tp_prev"],  df["tp"] * eff_vol,
        np.where(df["tp"] < df["tp_prev"], -df["tp"] * eff_vol, 0)
    )
    df["rmf"]    = df["raw_mf"].rolling(period, min_periods=1).sum()
    mu           = df["rmf"].rolling(period, min_periods=2).mean()
    sigma        = df["rmf"].rolling(period, min_periods=2).std()
    df["z_diff"] = ((df["rmf"] - mu) / sigma.replace(0, np.nan)).fillna(0)
    return df


def calc_markov(df: pd.DataFrame) -> dict:
    returns = df["Close"].pct_change().dropna()
    b0, b1  = returns.quantile(0.33), returns.quantile(0.67)
    def lbl(r): return 0 if r <= b0 else 2 if r > b1 else 1
    states  = returns.apply(lbl).values
    T       = np.zeros((3, 3))
    for i in range(len(states) - 1):
        T[states[i], states[i+1]] += 1
    rs = T.sum(axis=1, keepdims=True); rs[rs == 0] = 1; T /= rs
    try:
        vals, vecs = np.linalg.eig(T.T)
        si   = np.argmin(np.abs(vals - 1))
        stat = np.abs(vecs[:, si].real); stat /= stat.sum()
    except Exception:
        stat = np.ones(3) / 3
    cur  = int(states[-1])
    dist = np.zeros(3); dist[cur] = 1.0
    nd   = dist @ np.linalg.matrix_power(T, H4_PER_DAY)
    n3d  = dist @ np.linalg.matrix_power(T, H4_PER_DAY * 3)
    return {"transition": T, "labels": ["BAJISTA","NEUTRAL","ALCISTA"],
            "current": cur, "current_label": ["BAJISTA","NEUTRAL","ALCISTA"][cur],
            "next_day": nd, "next_3day": n3d, "stationary": stat}


def run_mc(price, returns, sims=3000, steps=6, z_adj=0.0, vol_mult=1.0):
    mu    = returns.mean()
    sigma = returns.std() * vol_mult
    drift = mu + z_adj * sigma * 0.15
    eps   = np.random.default_rng(42).standard_normal((sims, steps))
    return price * np.exp(((drift - 0.5*sigma**2) + sigma*eps).cumsum(axis=1))


def calc_volatility(df: pd.DataFrame) -> dict:
    c = df["Close"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    r = np.diff(np.log(c))
    tr  = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    atr = {w: float(pd.Series(tr).rolling(w).mean().iloc[-1]) for w in [5,14,20,50] if w <= len(tr)}
    rv_s = pd.Series(r).rolling(14).std() * np.sqrt(TRADING_DAYS * H4_PER_DAY)
    rv_c = float(rv_s.iloc[-1]) if not np.isnan(rv_s.iloc[-1]) else float(r[-14:].std() * np.sqrt(TRADING_DAYS*H4_PER_DAY))
    hl   = np.log(h[1:]/l[1:])
    pk   = float(np.sqrt((hl**2 / (4*np.log(2)))[-14:].mean() * TRADING_DAYS*H4_PER_DAY))
    gk   = 0.5*hl**2 - (2*np.log(2)-1)*(np.log(c[1:]/c[:-1]))**2
    gkv  = float(np.sqrt(gk[-14:].mean() * TRADING_DAYS*H4_PER_DAY))
    rs   = float(pd.Series(r).rolling(5).std().iloc[-1]  * np.sqrt(TRADING_DAYS*H4_PER_DAY))
    rl   = float(pd.Series(r).rolling(20).std().iloc[-1] * np.sqrt(TRADING_DAYS*H4_PER_DAY)) if len(r)>=20 else rv_c
    reg, rc = ("EXPANSIÓN", ORANGE) if rs > rl*1.3 else ("COMPRESIÓN", CYAN) if rs < rl*0.7 else ("NORMAL", YELLOW)
    p    = float(c[-1]); ds = float(r[-14:].std())
    return {"atr":atr,"rv_current":rv_c,"rv_series":rv_s,"parkinson":pk,"garman_klass":gkv,
            "rv_short":rs,"rv_long":rl,"vol_regime":reg,"vol_color":rc,
            "sigma_1d":ds,"price_1s":p*ds*np.sqrt(H4_PER_DAY),"price_2s":p*ds*np.sqrt(H4_PER_DAY)*2,
            "tr_series":pd.Series(tr)}


def calc_volume_profile(df: pd.DataFrame, bins: int = 40) -> dict:
    c=df["Close"].values.astype(float); h=df["High"].values.astype(float)
    l=df["Low"].values.astype(float);   v=df["Volume"].fillna(0).values.astype(float)
    if v.sum()==0 or pd.Series(v).nunique()<=3:
        rng=h-l; tp=(h+l+c)/3; v=(rng/rng.mean()*tp.mean()*1000)
    mn,mx    = l.min(), h.max()
    edges    = np.linspace(mn, mx, bins+1)
    centers  = (edges[:-1]+edges[1:])/2
    vol_bins = np.zeros(bins)
    for i in range(len(df)):
        mask = (centers>=l[i])&(centers<=h[i]); n=mask.sum()
        if n>0: vol_bins[mask] += v[i]/n
    poc = float(centers[np.argmax(vol_bins)])
    tv  = vol_bins.sum(); tgt = tv*0.70
    si  = np.argsort(vol_bins)[::-1]; cv=0; vai=[]
    for idx in si:
        if cv>=tgt: break
        cv+=vol_bins[idx]; vai.append(idx)
    return {"centers":centers,"vol_bins":vol_bins,"poc":poc,
            "vah":float(centers[max(vai)]),"val":float(centers[min(vai)]),
            "total_vol":tv,"price_min":mn,"price_max":mx}


def calc_vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["High"]+df["Low"]+df["Close"])/3
    v  = df["Volume"].fillna(0)
    if v.sum()==0 or v.nunique()<=3:
        rng=df["High"]-df["Low"]; v=(rng/rng.mean()*tp.mean()*1000).fillna(1.0)
    return (tp*v).cumsum() / v.cumsum().replace(0, np.nan)


def calc_volume_delta(df: pd.DataFrame) -> pd.Series:
    v=df["Volume"].fillna(0).values.astype(float)
    c=df["Close"].values.astype(float); h=df["High"].values.astype(float); l=df["Low"].values.astype(float)
    if v.sum()>0 and pd.Series(v).nunique()>3:
        rng=h-l; rng[rng==0]=1e-10
        delta=((c-l)/rng - (h-c)/rng)*v
    else:
        d=pd.Series(c).diff().fillna(0).values; delta=d*np.abs(d)*1000
    return pd.Series(delta, index=df.index)


def calc_volume_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    v=df["Volume"].fillna(0).values.astype(float)
    c=df["Close"].values.astype(float); h=df["High"].values.astype(float); l=df["Low"].values.astype(float)
    if v.sum()==0 or pd.Series(v).nunique()<=3:
        rng=h-l; tp=(h+l+c)/3; v=(rng/rng.mean()*tp.mean()*1000)
    vm = pd.Series(v).rolling(20,min_periods=5).mean().values.copy()
    vs = pd.Series(v).rolling(20,min_periods=5).std().values.copy()
    vs = np.where((vs==0)|np.isnan(vs), 1.0, vs)
    vm = np.where(np.isnan(vm), np.nanmean(v), vm)
    body = np.abs(c - np.roll(c,1))
    rng_v = np.where((h-l)==0, 1e-10, h-l)
    bp = body/rng_v
    at,sc=[],[]
    for i in range(len(df)):
        z=(v[i]-vm[i])/vs[i]
        if z>2.5 and bp[i]<0.3:    at.append("ABSORCIÓN");    sc.append(min(abs(z),5))
        elif z>2.5:                  at.append("SPIKE VOL");    sc.append(min(abs(z),5))
        elif z>1.8 and bp[i]>0.7:   at.append("MOMENTUM");     sc.append(min(abs(z),4))
        elif z<-1.5 and bp[i]>0.6:  at.append("RUPTURA SECA"); sc.append(min(abs(z),3))
        else:                        at.append("NORMAL");        sc.append(0)
    df2=df.copy(); df2["vol_z"]=(v-vm)/vs; df2["anomaly"]=at; df2["anom_score"]=sc; df2["vol_eff"]=v
    return df2


def interpret_zdiff(z, df, macro=0) -> dict:
    c=df["Close"].values; h=df["High"].values; l=df["Low"].values; n=len(c)
    lb=min(14,n); rh=h[-lb:].max(); rl_v=l[-lb:].min()
    rspan=rh-rl_v if rh!=rl_v else 1e-10
    ppct=(c[-1]-rl_v)/rspan
    in_top=ppct>0.75; in_bot=ppct<0.25
    rising=c[-3:].mean()>c[-6:-3].mean() if n>=6 else True
    ph=h[-6:-1].max() if n>=6 else rh; pl=l[-6:-1].min() if n>=6 else rl_v
    bu=c[-1]>ph; bd=c[-1]<pl; az=abs(z)
    # Simplified contextual interpretation
    if az>2.2:
        if z>0:
            if in_top and not bu: sig,col,bull="DISTRIBUCIÓN EN TECHO",RED,False
            elif bu:               sig,col,bull="RUPTURA ALCISTA",GREEN,True
            elif in_bot:           sig,col,bull="ACUMULACIÓN OCULTA",GREEN,True
            else:                  sig,col,bull="AGOTAMIENTO",ORANGE,None
        else:
            if in_bot and not bd: sig,col,bull="CAPITULACIÓN",GREEN,True
            elif bd:               sig,col,bull="RUPTURA BAJISTA",RED,False
            elif in_top:           sig,col,bull="DISTRIBUCIÓN OCULTA",RED,False
            else:                  sig,col,bull="AGOTAMIENTO BAJISTA",ORANGE,None
    elif az>1.5:
        if z>0:
            if bu or (rising and in_top): sig,col,bull="COMPRA MOMENTUM",GREEN,True
            elif in_bot and rising:        sig,col,bull="REBOTE EN SOPORTE","#69f0ae",True
            elif in_top and not rising:    sig,col,bull="DIVERGENCIA BAJISTA",RED,False
            else:                          sig,col,bull="SESGO ALCISTA","#69f0ae",True
        else:
            if bd or (not rising and in_bot): sig,col,bull="VENTA MOMENTUM",RED,False
            elif in_top and not rising:        sig,col,bull="RECHAZO RESISTENCIA",RED,False
            elif in_bot and rising:            sig,col,bull="DIVERGENCIA ALCISTA",YELLOW,None
            else:                              sig,col,bull="SESGO BAJISTA","#ff6b6b",False
    elif az>0.5:
        sig,col,bull=("SESGO ALCISTA","#69f0ae",True) if z>0 else ("SESGO BAJISTA","#ff6b6b",False)
    else:
        sig,col,bull="NEUTRAL",YELLOW,None
    return {"signal":sig,"color":col,"bull":bull,"rising":rising,"abs_z":az,
            "extreme":az>2.2,"price_pct":ppct,"in_top":in_top,"in_bottom":in_bot,
            "breaking_up":bu,"breaking_down":bd}

# ══════════════════════════════════════════════════════════════════════════════
#  MACRO ENGINE — FRED (gratis, sin key)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)  # cache 1h
def fetch_fred(series_id: str, periods: int = 120) -> pd.Series:
    """Descarga una serie de FRED sin API key."""
    try:
        url  = f"{FRED_BASE}{series_id}"
        df   = pd.read_csv(url, index_col=0, parse_dates=True)
        df.columns = [series_id]
        df   = df[df[series_id] != "."]
        df[series_id] = pd.to_numeric(df[series_id], errors="coerce").dropna()
        return df[series_id].dropna().tail(periods)
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=3600)
def load_macro_data() -> dict:
    """Carga todos los datos macro necesarios desde FRED."""
    data = {}
    series = {
        # Tipos de interés
        "FEDFUNDS":  "Fed Funds Rate",
        "DFF":       "Fed Funds Efectivo (diario)",
        # Inflación
        "PCEPILFE":  "PCE Core (Fed target)",
        "CPIAUCSL":  "CPI (IPC EEUU)",
        "T10YIE":    "Breakeven Inflación 10Y",
        # Tipos reales
        "DFII10":    "Tipo Real 10Y (TIPS)",
        "DFII5":     "Tipo Real 5Y (TIPS)",
        "DFII2":     "Tipo Real 2Y (TIPS)",
        # Curva de tipos
        "DGS2":      "Treasury 2Y",
        "DGS5":      "Treasury 5Y",
        "DGS10":     "Treasury 10Y",
        "DGS30":     "Treasury 30Y",
        "T10Y2Y":    "Spread 10Y-2Y",
        "T10Y3M":    "Spread 10Y-3M",
        # Condiciones financieras
        "NFCI":      "Chicago Fed Financial Conditions",
        "VIXCLS":    "VIX",
        # Empleo / Macro
        "UNRATE":    "Desempleo EEUU",
        "PAYEMS":    "Nóminas no agrícolas",
    }
    for sid, name in series.items():
        s = fetch_fred(sid)
        if not s.empty:
            data[sid] = {"series": s, "name": name, "latest": float(s.iloc[-1])}
    return data


def calc_real_rates(macro: dict) -> dict:
    """
    Tipos reales = Tipos nominales - Inflación esperada (breakeven).
    También calcula expectativas de tipos implícitas en la curva.
    """
    out = {}
    # Tipos reales directos (TIPS)
    for sid in ["DFII10","DFII5","DFII2"]:
        if sid in macro:
            out[sid] = macro[sid]["latest"]
    # Tipo real aproximado = Fed Funds - PCE Core
    if "FEDFUNDS" in macro and "PCEPILFE" in macro:
        ff  = macro["FEDFUNDS"]["latest"]
        pce = macro["PCEPILFE"]["latest"]
        out["real_rate_approx"] = ff - pce
        out["fed_funds"]        = ff
        out["pce_core"]         = pce
    # Breakeven inflacion
    if "T10YIE" in macro:
        out["breakeven_10y"] = macro["T10YIE"]["latest"]
    # Spread curva
    if "T10Y2Y" in macro:
        out["spread_10y2y"] = macro["T10Y2Y"]["latest"]
        out["inverted"]     = out["spread_10y2y"] < 0
    return out


def build_summary_score(r: dict) -> dict:
    """Puntuación multi-modelo -100 a +100."""
    price    = r["price"]
    zctx     = r["zdiff_ctx"]
    vdata    = r["vol_data"]
    mk       = r["markov"]
    vp       = r["vol_profile"]
    vwap_now = float(r["vwap_series"].iloc[-1])
    delt     = r["delta_series"]
    adj_bull = r["adj_bull"]

    signals = []

    # 1. Monte Carlo (peso 25)
    mc_s = (adj_bull - 50) * 0.5
    mc_l = "Alcista" if adj_bull>=60 else "Bajista" if adj_bull<=40 else "Neutral"
    mc_c = GREEN if adj_bull>=60 else RED if adj_bull<=40 else YELLOW
    signals.append({"cat":"Monte Carlo","ico":"🎲","val":f"P={adj_bull:.1f}%",
                    "lbl":mc_l,"col":mc_c,"score":mc_s,"peso":25})

    # 2. Z-Diff (peso 25)
    z    = r["last_z"]; zb = zctx.get("bull")
    z_s  = (min(abs(z),2.5)/2.5*25) * (1 if zb is True else -1 if zb is False else 0)
    signals.append({"cat":"Z-Diff Order Flow","ico":"⚡","val":f"{z:.3f}",
                    "lbl":zctx.get("signal","Neutral"),"col":zctx.get("color",YELLOW),
                    "score":z_s,"peso":25})

    # 3. Markov (peso 15)
    nd   = mk["next_day"]
    mk_s = (float(nd[2]) - float(nd[0])) * 15
    mk_l = mk["labels"][int(np.argmax(nd))]
    mk_c = GREEN if nd[2]>nd[0] else RED if nd[0]>nd[2] else YELLOW
    signals.append({"cat":"Cadena de Markov","ico":"🔗",
                    "val":f"↑{nd[2]*100:.0f}% ↓{nd[0]*100:.0f}%",
                    "lbl":f"→ {mk_l}","col":mk_c,"score":mk_s,"peso":15})

    # 4. Volumen zonas (peso 20)
    poc,vah,val_v = vp["poc"],vp["vah"],vp["val"]
    cd   = float(delt.sum()); rd = float(delt.iloc[-6:].sum())
    in_va= val_v<=price<=vah; ab_va=price>vah; bl_va=price<val_v
    ab_vw= price>vwap_now; db = cd>0 and rd>0
    if ab_va and ab_vw and db:    vs,vl,vc=20,"Alcista — sobre VA+VWAP",GREEN
    elif ab_va and not db:         vs,vl,vc=8,"Alcista débil — delta mixto","#69f0ae"
    elif bl_va and not ab_vw and not db: vs,vl,vc=-20,"Bajista — bajo VA+VWAP",RED
    elif bl_va and db:             vs,vl,vc=-8,"Bajista débil — delta mixto","#ff6b6b"
    elif in_va and price>poc:      vs,vl,vc=8,"Neutro-alcista — sobre POC","#69f0ae"
    elif in_va:                    vs,vl,vc=-8,"Neutro-bajista — bajo POC","#ff6b6b"
    else:                          vs,vl,vc=0,"Neutral",YELLOW
    signals.append({"cat":"Volumen & Zonas","ico":"📦",
                    "val":f"POC {poc:.4g}","lbl":vl,"col":vc,"score":vs,"peso":20})

    # 5. Volatilidad régimen (peso 5)
    reg  = vdata["vol_regime"]
    v_s  = 0; v_l = reg; v_c = vdata["vol_color"]
    signals.append({"cat":"Régimen Volatilidad","ico":"📊",
                    "val":f"{vdata['rv_current']*100:.1f}%","lbl":v_l,"col":v_c,
                    "score":v_s,"peso":5})

    # 6. Macro FRED (peso 15) — sólo si disponible
    mac  = st.session_state.get("macro_data") or {}
    rr   = calc_real_rates(mac) if mac else {}
    m_s  = 0
    if rr:
        rra = rr.get("real_rate_approx", rr.get("DFII10", 0))
        sp  = rr.get("spread_10y2y", 0)
        if rra > 1.5:   m_s -= 8   # tipos reales altos = restrictivo
        elif rra < 0:   m_s += 8   # tipos reales negativos = expansivo
        if sp < -0.5:   m_s -= 5   # curva muy invertida = recesión
        elif sp > 0.5:  m_s += 3
    m_l  = "Restrictivo" if m_s<0 else "Expansivo" if m_s>0 else "Neutral"
    m_c  = RED if m_s<0 else GREEN if m_s>0 else YELLOW
    signals.append({"cat":"Macro FRED","ico":"🌐",
                    "val":f"RR={rr.get('real_rate_approx',0):+.1f}%" if rr else "Sin datos",
                    "lbl":m_l,"col":m_c,"score":m_s,"peso":15})

    total   = float(np.clip(sum(s["score"] for s in signals), -100, 100))
    bulls   = sum(1 for s in signals if s["score"]>2)
    bears   = sum(1 for s in signals if s["score"]<-2)
    neuts   = len(signals)-bulls-bears
    align   = max(bulls,bears)/len(signals)

    if abs(total)>=55 and align>=0.75:   conv,conv_c="ALTA",GREEN if total>0 else RED
    elif abs(total)>=35 and align>=0.5:  conv,conv_c="MEDIA",ORANGE
    else:                                 conv,conv_c="BAJA",MUTED

    if total>=40 and conv in("ALTA","MEDIA"):   verd,vc_="SESGO ALCISTA ▲",GREEN
    elif total<=-40 and conv in("ALTA","MEDIA"):verd,vc_="SESGO BAJISTA ▼",RED
    elif abs(total)>=25:                         verd,vc_="MONITORIZAR ◉",ORANGE
    else:                                         verd,vc_="SIN SEÑAL CLARA —",MUTED

    return {"signals":signals,"total":total,"verdict":verd,"verdict_color":vc_,
            "conviction":conv,"conviction_color":conv_c,
            "bulls":bulls,"bears":bears,"neuts":neuts,"align":align,
            "poc":poc,"vah":vah,"val":val_v,"vwap":vwap_now,"real_rates":rr}

# ══════════════════════════════════════════════════════════════════════════════
#  CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def chart_price_mc(df, mc_paths):
    price   = float(df["Close"].iloc[-1])
    last_ts = df.index[-1]; h4d = timedelta(hours=4)
    steps   = mc_paths.shape[1]
    fts     = [last_ts + h4d*(i+1) for i in range(steps)]
    p5,p25,p50,p75,p95 = [np.percentile(mc_paths,p,axis=0) for p in [5,25,50,75,95]]
    final   = mc_paths[:,-1]
    bull_pct= float(np.mean(final>price)*100)

    fig = make_subplots(rows=2, cols=2,
        column_widths=[0.78,0.22], row_heights=[0.70,0.30],
        shared_xaxes=False, shared_yaxes=False,
        horizontal_spacing=0.01, vertical_spacing=0.06,
        specs=[[{"type":"xy"},{"type":"bar","rowspan":2}],[{"type":"bar"},None]])

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing_fillcolor=GREEN, increasing_line_color=GREEN,
        decreasing_fillcolor=RED,   decreasing_line_color=RED,
        name="H4", showlegend=False), row=1, col=1)

    for (ya,yb),fc in [((p95,p5),"rgba(0,144,255,0.06)"),((p75,p25),"rgba(0,144,255,0.15)")]:
        fig.add_trace(go.Scatter(x=fts+fts[::-1],y=list(ya)+list(yb[::-1]),
            fill="toself",fillcolor=fc,line=dict(color="rgba(0,0,0,0)"),showlegend=False),row=1,col=1)
    fig.add_trace(go.Scatter(x=fts,y=p50,line=dict(color=CYAN,width=2,dash="dash"),name="Mediana MC"),row=1,col=1)

    rng2 = np.random.default_rng(7)
    for i in rng2.choice(len(mc_paths), size=min(40,len(mc_paths)), replace=False):
        c2 = "rgba(0,230,118,0.03)" if mc_paths[i,-1]>price else "rgba(255,23,68,0.03)"
        fig.add_trace(go.Scatter(x=fts,y=mc_paths[i],line=dict(color=c2,width=1),
            showlegend=False,hoverinfo="skip"),row=1,col=1)
    fig.add_hline(y=price,line_color="rgba(255,255,255,0.3)",line_dash="dot",row=1,col=1)

    zc_colors = df["z_diff"].apply(lambda z: GREEN if z>1.5 else "#69f0ae" if z>0.5
        else YELLOW if z>-0.5 else "#ff6b6b" if z>-1.5 else RED)
    fig.add_trace(go.Bar(x=df.index,y=df["z_diff"],marker_color=zc_colors,showlegend=False),row=2,col=1)
    for yv,clr in [(1.5,"rgba(0,230,118,.3)"),(-1.5,"rgba(255,23,68,.3)"),(0,"rgba(255,255,255,.1)")]:
        fig.add_hline(y=yv,line_color=clr,line_dash="dash",row=2,col=1)

    mn,mx    = final.min(), final.max()
    edges    = np.linspace(mn,mx,55)
    counts,_ = np.histogram(final,bins=edges)
    centers  = (edges[:-1]+edges[1:])/2
    fig.add_trace(go.Bar(x=counts,y=centers,orientation="h",
        marker_color=[GREEN if c>price else RED for c in centers],
        marker_line_width=0,opacity=0.85,showlegend=False),row=1,col=2)
    fig.add_hline(y=price,line_color="rgba(255,255,255,0.7)",line_dash="dot",row=1,col=2)
    fig.add_annotation(xref="x3 domain",yref="y3 domain",x=0.5,y=1.0,yanchor="top",
        text=f"▲ {bull_pct:.1f}%",showarrow=False,
        font=dict(size=18,color=GREEN if bull_pct>=50 else RED,family="Rajdhani"))

    fig.update_layout(template="plotly_dark",paper_bgcolor=BG,plot_bgcolor=S1,
        height=600,margin=dict(l=8,r=8,t=8,b=8),xaxis_rangeslider_visible=False,
        legend=dict(orientation="h",y=1.02,x=0,font=dict(size=10),bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor=BORDER); fig.update_yaxes(gridcolor=BORDER)
    fig.update_yaxes(title_text="Precio H4",row=1,col=1)
    fig.update_yaxes(title_text="Z-Diff",row=2,col=1)
    fig.update_yaxes(showticklabels=False,row=1,col=2)
    return fig, bull_pct


def chart_volatility(df, vd, mc_paths):
    price   = float(df["Close"].iloc[-1])
    last_ts = df.index[-1]; h4d=timedelta(hours=4)
    steps   = mc_paths.shape[1]; fts=[last_ts+h4d*(i+1) for i in range(steps)]
    sh4     = vd["sigma_1d"]/np.sqrt(H4_PER_DAY); ts=np.arange(1,steps+1)
    c1u=[price*np.exp( sh4*np.sqrt(t)) for t in ts]
    c1d=[price*np.exp(-sh4*np.sqrt(t)) for t in ts]
    c2u=[price*np.exp( 2*sh4*np.sqrt(t)) for t in ts]
    c2d=[price*np.exp(-2*sh4*np.sqrt(t)) for t in ts]

    fig=make_subplots(rows=2,cols=2,
        subplot_titles=["Cono de Volatilidad","Volatilidad Realizada Rolling","ATR Multi-Periodo (% precio)","Distribución de Retornos vs Normal"],
        vertical_spacing=0.14,horizontal_spacing=0.1)

    fig.add_trace(go.Scatter(x=df.index[-40:],y=df["Close"].values[-40:],
        line=dict(color=TEXT,width=1.5),showlegend=False,name="Precio"),row=1,col=1)
    for ya,yb,fc in [(c2u,c2d,"rgba(0,144,255,0.06)"),(c1u,c1d,"rgba(0,144,255,0.15)")]:
        fig.add_trace(go.Scatter(x=fts+fts[::-1],y=ya+yb[::-1],fill="toself",
            fillcolor=fc,line=dict(color="rgba(0,0,0,0)"),showlegend=False),row=1,col=1)
    for arr,clr in [(c1u,"rgba(0,144,255,.6)"),(c1d,"rgba(0,144,255,.6)"),
                    (c2u,"rgba(0,144,255,.3)"),(c2d,"rgba(0,144,255,.3)")]:
        fig.add_trace(go.Scatter(x=fts,y=arr,line=dict(color=clr,width=1,dash="dot"),
            showlegend=False,hoverinfo="skip"),row=1,col=1)

    rv = vd["rv_series"].dropna()*100
    fig.add_trace(go.Scatter(x=df.index[-len(rv):],y=rv,line=dict(color=ORANGE,width=1.5),
        fill="toself",fillcolor="rgba(255,145,0,0.08)",showlegend=False),row=1,col=2)
    fig.add_hline(y=float(rv.mean()),line_dash="dash",line_color="rgba(255,145,0,0.4)",row=1,col=2)
    fig.add_annotation(xref="x2 domain",yref="y2 domain",x=0.02,y=0.98,
        text=f"RV actual: {vd['rv_current']*100:.2f}%  |  Régimen: {vd['vol_regime']}",
        showarrow=False,font=dict(size=10,color=vd["vol_color"]),xanchor="left",yanchor="top")

    aw=list(vd["atr"].keys()); av=list(vd["atr"].values()); ap=[v/price*100 for v in av]
    dec=1 if price>1000 else 2 if price>100 else 5
    fig.add_trace(go.Bar(x=[f"{w}v" for w in aw],y=ap,
        marker_color=[CYAN,BLUE,ORANGE,PURPLE][:len(aw)],
        text=[f"{v:.{dec}f}" for v in av],textposition="outside",showlegend=False),row=2,col=1)

    rets=np.diff(np.log(df["Close"].values.astype(float)))*100
    fig.add_trace(go.Histogram(x=rets,nbinsx=40,
        marker_color="rgba(0,144,255,0.55)",showlegend=False),row=2,col=2)
    xf=np.linspace(rets.min(),rets.max(),100)
    yf=stats.norm.pdf(xf,rets.mean(),rets.std())*len(rets)*(xf[1]-xf[0])
    fig.add_trace(go.Scatter(x=xf,y=yf,line=dict(color=ORANGE,width=2),showlegend=False),row=2,col=2)
    kurt=float(stats.kurtosis(rets)); skew=float(stats.skew(rets))
    fig.add_annotation(xref="x4 domain",yref="y4 domain",x=0.98,y=0.95,
        text=f"Kurt: {kurt:.2f}  Skew: {skew:.2f}",showarrow=False,align="right",
        font=dict(size=10,color=ORANGE if abs(kurt)>1 else MUTED),xanchor="right",yanchor="top")

    fig.update_layout(template="plotly_dark",paper_bgcolor=BG,plot_bgcolor=S1,
        height=620,margin=dict(l=8,r=8,t=40,b=8),showlegend=False)
    fig.update_xaxes(gridcolor=BORDER); fig.update_yaxes(gridcolor=BORDER)
    return fig


def chart_volume_profile(df, vp, vwap, delta, danom):
    price=float(df["Close"].iloc[-1]); dec=1 if price>1000 else 2 if price>100 else 5
    fig=make_subplots(rows=2,cols=2,
        column_widths=[0.72,0.28],row_heights=[0.62,0.38],
        shared_xaxes=False,horizontal_spacing=0.02,vertical_spacing=0.08,
        specs=[[{"type":"xy"},{"type":"bar","rowspan":2}],[{"type":"bar"},None]],
        subplot_titles=["Velas + VWAP + Zonas","Perfil de Volumen","Volume Delta",""])

    fig.add_trace(go.Candlestick(x=df.index,open=df["Open"],high=df["High"],
        low=df["Low"],close=df["Close"],
        increasing_fillcolor=GREEN,increasing_line_color=GREEN,
        decreasing_fillcolor=RED,decreasing_line_color=RED,showlegend=False),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=vwap,
        line=dict(color=YELLOW,width=1.5,dash="dash"),name="VWAP"),row=1,col=1)
    fig.add_hline(y=vp["poc"],line_color=ORANGE,line_width=2,row=1,col=1,
        annotation_text=f"POC {vp['poc']:.{dec}f}",annotation_font_color=ORANGE,annotation_position="right")
    fig.add_hrect(y0=vp["val"],y1=vp["vah"],fillcolor="rgba(0,144,255,0.07)",line_width=0,row=1,col=1)
    fig.add_hline(y=vp["vah"],line_color="rgba(0,144,255,.5)",line_width=1,line_dash="dot",row=1,col=1,
        annotation_text=f"VAH {vp['vah']:.{dec}f}",annotation_font_color=BLUE,annotation_position="right")
    fig.add_hline(y=vp["val"],line_color="rgba(0,144,255,.5)",line_width=1,line_dash="dot",row=1,col=1,
        annotation_text=f"VAL {vp['val']:.{dec}f}",annotation_font_color=BLUE,annotation_position="right")

    for _,row_d in danom[danom["anomaly"]!="NORMAL"].iterrows():
        ac={"ABSORCIÓN":PURPLE,"SPIKE VOL":ORANGE,"MOMENTUM":CYAN,"RUPTURA SECA":YELLOW}.get(row_d["anomaly"],TEXT)
        fig.add_trace(go.Scatter(x=[row_d.name],y=[float(row_d["High"])*1.001],
            mode="markers+text",marker=dict(symbol="triangle-down",size=9,color=ac),
            text=[row_d["anomaly"][:3]],textposition="top center",
            textfont=dict(size=8,color=ac),showlegend=False),row=1,col=1)

    va_m=(vp["centers"]>=vp["val"])&(vp["centers"]<=vp["vah"])
    poc_w=(vp["price_max"]-vp["price_min"])/len(vp["centers"])
    bc=[ORANGE if abs(c-vp["poc"])<poc_w else BLUE if va_m[i] else MUTED
        for i,c in enumerate(vp["centers"])]
    fig.add_trace(go.Bar(x=vp["vol_bins"],y=vp["centers"],orientation="h",
        marker_color=bc,marker_line_width=0,opacity=0.85,showlegend=False),row=1,col=2)
    fig.add_hline(y=price,line_color="rgba(255,255,255,0.6)",line_width=1.5,line_dash="dot",row=1,col=2)

    fig.add_trace(go.Bar(x=df.index,y=delta.values,
        marker_color=[GREEN if d>=0 else RED for d in delta.values],
        marker_line_width=0,opacity=0.8,showlegend=False),row=2,col=1)
    fig.add_hline(y=0,line_color="rgba(255,255,255,.15)",line_dash="dot",row=2,col=1)

    fig.update_layout(template="plotly_dark",paper_bgcolor=BG,plot_bgcolor=S1,
        height=640,margin=dict(l=8,r=8,t=30,b=8),
        legend=dict(orientation="h",y=1.02,bgcolor="rgba(0,0,0,0)"),
        xaxis_rangeslider_visible=False)
    fig.update_xaxes(gridcolor=BORDER); fig.update_yaxes(gridcolor=BORDER)
    fig.update_yaxes(showticklabels=False,row=1,col=2)
    return fig


def chart_real_rates(macro: dict) -> go.Figure:
    """Gráfico tipos reales: Fed Funds, PCE Core, Tipo Real aproximado, TIPS."""
    fig = make_subplots(rows=2, cols=2,
        subplot_titles=[
            "Tipos Reales = Fed Funds − PCE Core",
            "Tipos TIPS (Reales directos del mercado)",
            "Curva de Tipos del Tesoro EEUU",
            "Spreads & Riesgo Recesión",
        ],
        vertical_spacing=0.14, horizontal_spacing=0.10)

    # ── Panel 1: Tipos reales aproximados ────────────────────────────────────
    if "FEDFUNDS" in macro and "PCEPILFE" in macro:
        ff  = macro["FEDFUNDS"]["series"]
        pce = macro["PCEPILFE"]["series"]
        ff_r, pce_r = ff.reindex(pce.index, method="ffill").dropna(), pce
        idx  = ff_r.index.intersection(pce_r.index)
        real = ff_r.loc[idx] - pce_r.loc[idx]

        fig.add_trace(go.Scatter(x=ff_r.index,y=ff_r.values,name="Fed Funds",
            line=dict(color=CYAN,width=2)),row=1,col=1)
        fig.add_trace(go.Scatter(x=pce_r.index,y=pce_r.values,name="PCE Core",
            line=dict(color=ORANGE,width=2)),row=1,col=1)
        fig.add_trace(go.Scatter(x=real.index,y=real.values,name="Tipo Real",
            line=dict(color=GREEN,width=2.5),fill="tozeroy",
            fillcolor="rgba(0,230,118,0.08)"),row=1,col=1)
        fig.add_hline(y=0,line_color="rgba(255,255,255,.3)",line_dash="dot",row=1,col=1)

        # Annotate current real rate
        cur_rr = float(real.iloc[-1])
        fig.add_annotation(x=real.index[-1],y=cur_rr,
            text=f" RR actual: {cur_rr:+.2f}%",
            showarrow=False,font=dict(size=11,color=GREEN if cur_rr>0 else RED),
            xanchor="left",row=1,col=1)

    # ── Panel 2: TIPS reales ─────────────────────────────────────────────────
    tips_colors = {
        "DFII2": (CYAN,   "Real 2Y (TIPS)"),
        "DFII5": (BLUE,   "Real 5Y (TIPS)"),
        "DFII10":(GREEN,  "Real 10Y (TIPS)"),
    }
    for sid,(col_t,name_t) in tips_colors.items():
        if sid in macro:
            s=macro[sid]["series"]
            fig.add_trace(go.Scatter(x=s.index,y=s.values,name=name_t,
                line=dict(color=col_t,width=1.5)),row=1,col=2)
    fig.add_hline(y=0,line_color="rgba(255,255,255,.3)",line_dash="dot",row=1,col=2)
    # Zero-line annotation
    fig.add_annotation(xref="x2 domain",yref="y2 domain",x=0.02,y=0.05,
        text="0% = tipos reales neutros",showarrow=False,
        font=dict(size=9,color=MUTED),xanchor="left",yanchor="bottom")

    # ── Panel 3: Curva de tipos (snapshot actual) ─────────────────────────────
    curve_map = {"DGS2":"2Y","DGS5":"5Y","DGS10":"10Y","DGS30":"30Y"}
    maturities, yields = [], []
    for sid, mat in curve_map.items():
        if sid in macro:
            maturities.append(mat); yields.append(macro[sid]["latest"])
    if maturities:
        y_c = [GREEN if y==max(yields) else RED if y==min(yields) else CYAN for y in yields]
        fig.add_trace(go.Scatter(x=maturities,y=yields,mode="lines+markers",
            name="Curva actual",line=dict(color=CYAN,width=2),
            marker=dict(color=y_c,size=10)),row=2,col=1)
        # Historical curve comparison (12m ago)
        hist_yields = []
        for sid in curve_map:
            if sid in macro:
                s = macro[sid]["series"]
                if len(s)>=252:
                    hist_yields.append(float(s.iloc[-252]))
                else:
                    hist_yields.append(None)
        if any(v is not None for v in hist_yields):
            hy_clean = [v for v in hist_yields if v is not None]
            hm_clean = [m for m,v in zip(maturities,hist_yields) if v is not None]
            fig.add_trace(go.Scatter(x=hm_clean,y=hy_clean,mode="lines",
                name="Hace 12 meses",
                line=dict(color=MUTED,width=1.5,dash="dash")),row=2,col=1)

    # ── Panel 4: Spreads ──────────────────────────────────────────────────────
    spread_map = {"T10Y2Y":"Spread 10Y−2Y","T10Y3M":"Spread 10Y−3M"}
    for sid, name_s in spread_map.items():
        if sid in macro:
            s=macro[sid]["series"]
            clr=[RED if v<0 else GREEN for v in s.values]
            fig.add_trace(go.Bar(x=s.index,y=s.values,name=name_s,
                marker_color=clr,opacity=0.7),row=2,col=2)
    fig.add_hline(y=0,line_color="rgba(255,255,255,.4)",line_width=1.5,row=2,col=2)
    fig.add_hrect(y0=-5,y1=0,fillcolor="rgba(255,23,68,0.04)",line_width=0,row=2,col=2)
    fig.add_annotation(xref="x4 domain",yref="y4 domain",x=0.02,y=0.05,
        text="Zona invertida = señal recesión",showarrow=False,
        font=dict(size=9,color=RED),xanchor="left",yanchor="bottom")

    fig.update_layout(template="plotly_dark",paper_bgcolor=BG,plot_bgcolor=S1,
        height=650,margin=dict(l=8,r=8,t=40,b=8),
        legend=dict(orientation="h",y=-0.05,bgcolor="rgba(0,0,0,0)",font=dict(size=10)))
    fig.update_xaxes(gridcolor=BORDER); fig.update_yaxes(gridcolor=BORDER)
    fig.update_yaxes(title_text="Tasa %",row=1,col=1)
    fig.update_yaxes(title_text="Tasa Real %",row=1,col=2)
    fig.update_yaxes(title_text="Rendimiento %",row=2,col=1)
    fig.update_yaxes(title_text="Spread %",row=2,col=2)
    return fig


def chart_macro_indicators(macro: dict) -> go.Figure:
    """Panel secundario: VIX, desempleo, condiciones financieras."""
    fig = make_subplots(rows=1, cols=3,
        subplot_titles=["VIX — Volatilidad Implícita","Desempleo EEUU","Cond. Financieras Chicago Fed"],
        horizontal_spacing=0.08)

    if "VIXCLS" in macro:
        s=macro["VIXCLS"]["series"]
        clr=[RED if v>30 else ORANGE if v>20 else GREEN for v in s.values]
        fig.add_trace(go.Scatter(x=s.index,y=s.values,name="VIX",
            line=dict(color=ORANGE,width=1.5),fill="tozeroy",
            fillcolor="rgba(255,145,0,0.08)"),row=1,col=1)
        fig.add_hline(y=20,line_color="rgba(255,145,0,.4)",line_dash="dash",row=1,col=1)
        fig.add_hline(y=30,line_color="rgba(255,23,68,.4)",line_dash="dash",row=1,col=1)

    if "UNRATE" in macro:
        s=macro["UNRATE"]["series"]
        fig.add_trace(go.Scatter(x=s.index,y=s.values,name="Desempleo",
            line=dict(color=CYAN,width=1.5),fill="tozeroy",
            fillcolor="rgba(0,229,255,0.06)"),row=1,col=2)

    if "NFCI" in macro:
        s=macro["NFCI"]["series"]
        fig.add_trace(go.Bar(x=s.index,y=s.values,name="NFCI",
            marker_color=[RED if v>0 else GREEN for v in s.values],opacity=0.7),row=1,col=3)
        fig.add_hline(y=0,line_color="rgba(255,255,255,.3)",row=1,col=3)
        fig.add_annotation(xref="x3 domain",yref="y3 domain",x=0.02,y=0.98,
            text=">0 = condiciones restrictivas",showarrow=False,
            font=dict(size=9,color=MUTED),xanchor="left",yanchor="top")

    fig.update_layout(template="plotly_dark",paper_bgcolor=BG,plot_bgcolor=S1,
        height=320,margin=dict(l=8,r=8,t=40,b=8),showlegend=False)
    fig.update_xaxes(gridcolor=BORDER); fig.update_yaxes(gridcolor=BORDER)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
#  GEMINI AI
# ══════════════════════════════════════════════════════════════════════════════

def build_ai_prompt(ticker, asset_type, r: dict, summ: dict, macro: dict) -> str:
    """Construye el prompt completo con todos los datos del modelo para Gemini."""
    price  = r["price"]; dec=1 if price>1000 else 2 if price>100 else 4
    zctx   = r["zdiff_ctx"]; vd=r["vol_data"]; mk=r["markov"]
    nd     = mk["next_day"]
    rr     = summ.get("real_rates",{})
    today  = datetime.now().strftime("%A, %d de %B de %Y — %H:%M UTC")

    macro_block = ""
    if rr:
        macro_block = f"""
Datos macroeconómicos (FRED):
  - Fed Funds Rate: {rr.get('fed_funds','N/A')}%
  - PCE Core (objetivo Fed): {rr.get('pce_core','N/A')}%
  - Tipo Real aproximado (FF-PCE): {rr.get('real_rate_approx','N/A'):+.2f}%
  - TIPS 10Y (tipo real mercado): {rr.get('DFII10','N/A')}%
  - Spread 10Y-2Y: {rr.get('spread_10y2y','N/A')}% {'⚠ INVERTIDA' if rr.get('inverted') else ''}
  - Breakeven inflación 10Y: {rr.get('breakeven_10y','N/A')}%
  - Entorno macro: {summ['signals'][-1]['lbl']}"""

    return f"""Fecha: {today}

Eres un analista cuantitativo senior especializado en mercados financieros. Tu análisis debe ser técnico, objetivo y en español. Sin asteriscos ni markdown.

═══════ DATOS DEL MODELO ═══════

Activo: {ticker} ({asset_type})
Precio actual: {price:.{dec}f}

ORDER FLOW Z-DIFF:
  - Z-Diff H4: {r['last_z']:.3f}
  - Señal: {zctx['signal']}
  - Posición en rango: {zctx['price_pct']*100:.0f}% (0=mínimos, 100=máximos)
  - Precio: {'subiendo' if zctx['rising'] else 'cayendo'}
  - {'Ruptura alcista activa' if zctx['breaking_up'] else 'Ruptura bajista activa' if zctx['breaking_down'] else 'Sin ruptura'}

MONTE CARLO (GBM, 3000 sims):
  - P(alcista): {r['adj_bull']:.1f}%
  - Rango IC 90%: [{r['p5']:.{dec}f}, {r['p95']:.{dec}f}]
  - Media MC: {float(r['final'].mean()):.{dec}f}

CADENAS DE MARKOV:
  - Estado actual: {mk['current_label']}
  - P(alcista mañana): {nd[2]*100:.1f}%
  - P(bajista mañana): {nd[0]*100:.1f}%

VOLATILIDAD:
  - Realizada 14v: {vd['rv_current']*100:.2f}% anualizada
  - Garman-Klass: {vd['garman_klass']*100:.2f}%
  - Régimen: {vd['vol_regime']}
  - Movimiento 1σ/día: ±{vd['price_1s']:.{dec}f}
{macro_block}

SCORE TOTAL DEL MODELO: {summ['total']:+.0f}/100
Veredicto cuantitativo: {summ['verdict']}
Convicción: {summ['conviction']}

═══════ ANÁLISIS SOLICITADO ═══════

1. Analiza la coherencia entre el Z-Diff, el Monte Carlo y el régimen de volatilidad.
2. Interpreta qué está haciendo el flujo institucional según el contexto del precio (posición en rango, ruptura/rebote).
3. Comenta el entorno macroeconómico: tipos reales, curva de tipos y su impacto en este activo.
4. Identifica el escenario más probable para las próximas 24-48 horas.
5. Señala los niveles clave a vigilar (soporte/resistencia cuantitativa).

Responde en 5-6 párrafos concisos. No des recomendaciones de compra o venta explícitas."""


def call_gemini(prompt: str, api_key: str) -> str:
    """Llama a Gemini 2.0 Flash con la key del cliente."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        resp  = model.generate_content(prompt)
        return resp.text
    except ImportError:
        # Fallback: REST API directa
        url  = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        body = {"contents":[{"parts":[{"text":prompt}]}]}
        r    = requests.post(url, json=body, timeout=30)
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"Error al conectar con Gemini: {e}\n\nVerifica que tu API key sea correcta y tengas acceso a Gemini 2.0 Flash."

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS UI
# ══════════════════════════════════════════════════════════════════════════════

def kpi(col, lbl, val, sub="", color=TEXT):
    col.markdown(f"""<div class='kpi'>
        <div class='kpi-lbl'>{lbl}</div>
        <div class='kpi-val' style='color:{color}'>{val}</div>
        <div class='kpi-sub'>{sub}</div>
    </div>""", unsafe_allow_html=True)


def info_card(title, body, border_color=CYAN, bg=S0):
    st.markdown(f"""<div style='background:{bg};border:1px solid {BORDER};
        border-left:4px solid {border_color};border-radius:6px;
        padding:16px 20px;margin:8px 0'>
        <div style='font-size:9px;letter-spacing:3px;color:{border_color};
            text-transform:uppercase;margin-bottom:8px;font-family:JetBrains Mono,monospace'>{title}</div>
        <div style='font-size:13px;color:{TEXT};line-height:1.75'>{body}</div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""<div style='padding:16px 0 8px'>
        <div style='font-family:Rajdhani,sans-serif;font-size:24px;font-weight:700;
            color:{CYAN};letter-spacing:3px'>Quant<span style='color:{YELLOW}'>Edge</span>
            <span style='font-size:14px;color:{MUTED}'>PRO</span></div>
        <div style='font-size:10px;color:{MUTED};letter-spacing:2px'>v{APP_VERSION} · Dashboard Cuantitativo</div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    # Gemini API Key
    with st.expander("🔑 Gemini API Key", expanded=not bool(st.session_state.get("gemini_key",""))):
        st.caption("Obtén tu key gratis en [aistudio.google.com](https://aistudio.google.com/apikey)")
        gemini_key = st.text_input("API Key", type="password",
                                    value=st.session_state.get("gemini_key",""),
                                    placeholder="AIza...",
                                    label_visibility="collapsed")
        if gemini_key:
            st.session_state["gemini_key"] = gemini_key
            st.success("✓ Key configurada")
    st.divider()

    st.markdown("#### Activo")
    quick = st.selectbox("Acceso rápido", list(QUICK_MAP.keys()))
    default_sym, default_type = QUICK_MAP[quick]
    ticker = st.text_input("Símbolo Yahoo Finance", value=default_sym,
                            placeholder="ES=F, NQ=F, EURUSD=X...")
    if ticker in FUTURES_NOTE:
        st.caption(f"🔄 {FUTURES_NOTE[ticker]} — 23h/día")
    elif ticker in ("^GSPC","^GDAXI","^IXIC"):
        st.warning("Índice spot. Usa ES=F/NQ=F para datos continuos.")
    asset_type = st.selectbox("Tipo de activo",
        ["forex","index","commodity","stock","crypto"],
        index=["forex","index","commodity","stock","crypto"].index(default_type))

    st.divider()
    st.markdown("#### Modelo H4")
    horizon   = st.selectbox("Horizonte", [1,3,5], format_func=lambda x: f"{x} día{'s' if x>1 else ''}")
    n_candles = st.slider("Velas H4", 30, 90, 60)
    z_period  = st.slider("Periodo Z-Diff", 10, 30, 14)
    mc_sims   = st.selectbox("Simulaciones MC", [1000,3000,5000], index=1)
    threshold = st.selectbox("Umbral alerta %", [60,65,70], index=1)

    st.divider()
    st.markdown("#### Gestión de riesgo")
    account  = st.number_input("Capital ($)", value=10000, step=500)
    risk_pct = st.slider("Riesgo %", 0.5, 10.0, 2.0, 0.5)
    instr    = st.selectbox("Instrumento",
        ["Forex std (100k)","Forex mini (10k)","XAU/USD","Índice CFD"])

    st.divider()
    col_l, col_r = st.columns(2)
    load_btn = col_l.button("📡 Cargar H4", use_container_width=True, type="primary")
    run_btn  = col_r.button("▶ Ejecutar",   use_container_width=True,
                             disabled=st.session_state.df is None)
    macro_btn= st.button("🌐 Cargar Macro FRED", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
if load_btn and ticker:
    with st.spinner(f"Descargando velas H4 — {ticker}..."):
        try:
            raw = yf.download(ticker, period="60d", interval="4h",
                               auto_adjust=True, progress=False)
            if raw.empty:
                st.error(f"Sin datos para '{ticker}'")
            else:
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                df = raw.tail(n_candles).copy()
                df.index = pd.to_datetime(df.index)
                df = calc_order_flow(df, z_period)
                st.session_state.df = df
                st.session_state.results = None
                p=float(df["Close"].iloc[-1]); d=1 if p>1000 else 2 if p>100 else 5
                st.sidebar.success(f"✓ {len(df)} velas · {p:.{d}f}")
        except Exception as e:
            st.error(f"Error: {e}")

if macro_btn:
    with st.spinner("Cargando datos FRED (tipos, inflación, curva)..."):
        st.session_state.macro_data = load_macro_data()
        n = len(st.session_state.macro_data)
        st.sidebar.success(f"✓ {n} series FRED cargadas")

# ══════════════════════════════════════════════════════════════════════════════
#  RUN MODEL
# ══════════════════════════════════════════════════════════════════════════════
if run_btn and st.session_state.df is not None:
    df    = st.session_state.df
    price = float(df["Close"].iloc[-1])
    rets  = np.diff(np.log(df["Close"].values.astype(float)))

    with st.spinner("Ejecutando modelos cuantitativos..."):
        last_z    = float(df["z_diff"].iloc[-1])
        z_adj     = float(np.clip(last_z,-2,2))
        mc_steps  = horizon * H4_PER_DAY
        zdiff_ctx = interpret_zdiff(last_z, df)
        mc_paths  = run_mc(price, rets, mc_sims, mc_steps, z_adj)
        markov    = calc_markov(df)
        vol_data  = calc_volatility(df)
        vp        = calc_volume_profile(df)
        vwap_s    = calc_vwap(df)
        delta_s   = calc_volume_delta(df)
        df_anom   = calc_volume_anomalies(df)

        final     = mc_paths[:,-1]
        adj_bull  = float(np.clip(np.mean(final>price)*100, 10, 90))
        adj_bear  = 100 - adj_bull

        st.session_state.results = dict(
            price=price, last_z=last_z, last_rmf=float(df["rmf"].iloc[-1]),
            adj_bull=adj_bull, adj_bear=adj_bear,
            mc_paths=mc_paths, final=final,
            zdiff_ctx=zdiff_ctx, markov=markov, vol_data=vol_data,
            p5=float(np.percentile(final,5)),   p95=float(np.percentile(final,95)),
            p20=float(np.percentile(final,20)), p80=float(np.percentile(final,80)),
            p8=float(np.percentile(final,8)),   p92=float(np.percentile(final,92)),
            vol_profile=vp, vwap_series=vwap_s,
            delta_series=delta_s, df_anom=df_anom,
        )
        st.session_state.ai_analysis = None

# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""<div style='display:flex;align-items:baseline;gap:12px;margin-bottom:8px'>
  <span style='font-family:Rajdhani,sans-serif;font-size:32px;font-weight:700;
    color:{CYAN};letter-spacing:3px'>Quant<span style='color:{YELLOW}'>Edge</span> PRO</span>
  <span style='font-size:11px;color:{MUTED};letter-spacing:2px'>
    H4 · CUANTITATIVO · FRED MACRO · GEMINI AI</span>
</div>""", unsafe_allow_html=True)

r = st.session_state.results

if r is None:
    st.info("👈 Selecciona un activo y pulsa **Cargar H4** → **Ejecutar** para comenzar el análisis.")
    c1,c2,c3 = st.columns(3)
    with c1:
        info_card("DATOS DE MERCADO","Yahoo Finance H4 — futuros continuos disponibles (ES=F, NQ=F, FDAX=F) para datos 23h/día sin gaps de sesión.",CYAN)
    with c2:
        info_card("MACRO CUANTITATIVA","Datos FRED (Federal Reserve): tipos reales, curva de tipos, spreads, VIX, condiciones financieras — gratuitos, sin API key.",ORANGE)
    with c3:
        info_card("ANÁLISIS IA","Introduce tu Gemini API key (gratuita en Google AI Studio). El análisis combina todos los modelos cuantitativos con contexto macroeconómico.",PURPLE)
    st.stop()

# KPI bar global
price = r["price"]; dec=1 if price>1000 else 2 if price>100 else 4
summ  = build_summary_score(r)
bc    = summ["verdict_color"]
mk    = r["markov"]; vdata=r["vol_data"]; zctx=r["zdiff_ctx"]
nd    = mk["next_day"]

h1,h2,h3,h4,h5,h6 = st.columns(6)
kpi(h1,"Veredicto",summ["verdict"],f"Score {summ['total']:+.0f}/100 · {summ['conviction']}",bc)
kpi(h2,"Z-Diff H4",f"{r['last_z']:.3f}",zctx["signal"][:24],zctx["color"])
kpi(h3,"Precio",f"{price:.{dec}f}",ticker,TEXT)
kpi(h4,"IC 90% MC",f"{r['p5']:.{dec}f}–{r['p95']:.{dec}f}",f"{horizon}d horizonte",BLUE)
kpi(h5,"Vol Realizada",f"{vdata['rv_current']*100:.2f}%",vdata["vol_regime"],vdata["vol_color"])
kpi(h6,"Markov",mk["labels"][int(np.argmax(nd))],f"{max(nd)*100:.0f}% prob. mañana",
    [RED,YELLOW,GREEN][int(np.argmax(nd))])

# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab0,tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "⬡ RESUMEN",
    "① MERCADO",
    "② VOLATILIDAD",
    "③ VOLUMEN",
    "④ MACRO",
    "⑤ IA",
])

# ─────────────────────────────────────────────────────────────────────────────
# ⬡ RESUMEN EJECUTIVO
# ─────────────────────────────────────────────────────────────────────────────
with tab0:
    dec=1 if r["price"]>1000 else 2 if r["price"]>100 else 4
    vc=summ["verdict_color"]

    # Veredicto central
    st.markdown(f"""<div style='background:{S1};border:2px solid {vc};border-radius:8px;
        padding:30px 36px;margin-bottom:20px;text-align:center'>
      <div style='font-size:10px;letter-spacing:4px;color:{MUTED};
          text-transform:uppercase;margin-bottom:10px;font-family:JetBrains Mono,monospace'>ANÁLISIS CUANTITATIVO — {ticker}</div>
      <div style='font-family:Rajdhani,sans-serif;font-size:48px;font-weight:700;
          color:{vc};letter-spacing:3px;line-height:1.1'>{summ["verdict"]}</div>
      <div style='margin-top:14px;display:flex;justify-content:center;gap:36px;font-size:12px;color:{MUTED}'>
        <span>Score: <b style='color:{vc};font-size:18px'>{summ["total"]:+.0f}</b><span style='font-size:11px'>/100</span></span>
        <span>Convicción: <b style='color:{summ["conviction_color"]}'>{summ["conviction"]}</b></span>
        <span><b style='color:{GREEN}'>{summ["bulls"]}</b> alcistas · <b style='color:{RED}'>{summ["bears"]}</b> bajistas · <b style='color:{MUTED}'>{summ["neuts"]}</b> neutras</span>
        <span>Alineación: <b style='color:{TEXT}'>{summ["align"]*100:.0f}%</b></span>
      </div>
    </div>""", unsafe_allow_html=True)

    # Score bar
    sc=summ["total"]; pp=max(0,sc)/100*100; pn=max(0,-sc)/100*100
    st.markdown(f"""<div style='margin-bottom:20px'>
      <div style='display:flex;justify-content:space-between;font-size:9px;color:{MUTED};
          letter-spacing:2px;text-transform:uppercase;margin-bottom:5px'>
        <span>BAJISTA −100</span><span>NEUTRAL 0</span><span>+100 ALCISTA</span></div>
      <div style='position:relative;height:14px;background:{S1};border-radius:7px;
          border:1px solid {BORDER};overflow:hidden'>
        <div style='position:absolute;left:50%;top:0;bottom:0;width:2px;background:{MUTED}'></div>
        {"<div style='position:absolute;left:50%;top:0;bottom:0;width:"+str(pp/2)+"%;background:"+GREEN+";border-radius:0 4px 4px 0'></div>" if sc>0 else ""}
        {"<div style='position:absolute;right:50%;top:0;bottom:0;width:"+str(pn/2)+"%;background:"+RED+";border-radius:4px 0 0 4px'></div>" if sc<0 else ""}
      </div></div>""", unsafe_allow_html=True)

    # Señales individuales
    st.markdown("### Modelos Cuantitativos")
    cols = st.columns(len(summ["signals"]))
    for col, sig in zip(cols, summ["signals"]):
        bw = abs(sig["score"]) / max(sig["peso"],1) * 100
        bc2= sig["col"] if sig["score"]!=0 else MUTED
        col.markdown(f"""<div class='kpi' style='border-top:3px solid {sig["col"]}'>
            <div style='font-size:18px;margin-bottom:6px'>{sig["ico"]}</div>
            <div class='kpi-lbl'>{sig["cat"]}</div>
            <div style='font-family:Rajdhani,sans-serif;font-size:18px;font-weight:700;
                color:{sig["col"]};margin-bottom:4px'>{sig["val"]}</div>
            <div style='font-size:10px;color:{sig["col"]};margin-bottom:8px'>{sig["lbl"]}</div>
            <div style='height:3px;background:{BORDER};border-radius:2px;overflow:hidden;margin-bottom:5px'>
              <div style='height:100%;width:{bw:.0f}%;background:{bc2};border-radius:2px'></div>
            </div>
            <div style='font-size:9px;color:{MUTED}'>
              <b style='color:{sig["col"]}'>{sig["score"]:+.1f}</b> pts · peso {sig["peso"]}
            </div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # Zonas clave de volumen
    st.markdown("### Zonas Clave de Volumen")
    price_now = r["price"]
    vp        = r["vol_profile"]
    vwap_now  = float(r["vwap_series"].iloc[-1])
    poc,vah,val_v = summ["poc"],summ["vah"],summ["val"]

    zones = [
        (vah,      "VAH",         BLUE,   "Techo del 70% del volumen — resistencia"),
        (poc,      "POC",         ORANGE, "Mayor volumen negociado — imán de precio"),
        (vwap_now, "VWAP",        YELLOW, "Precio medio ponderado — referencia institucional"),
        (val_v,    "VAL",         BLUE,   "Suelo del 70% del volumen — soporte"),
    ]
    all_items = zones + [(price_now,"PRECIO",TEXT,"Último cierre H4")]
    all_items.sort(key=lambda x: x[0], reverse=True)
    mn_z,mx_z = min(x[0] for x in all_items), max(x[0] for x in all_items)
    span_z    = mx_z-mn_z if mx_z!=mn_z else 1e-10

    z_html = f"<div style='background:{S0};border:1px solid {BORDER};border-radius:6px;padding:16px 20px;margin-bottom:12px'>"
    z_html += f"<div style='font-size:9px;letter-spacing:3px;color:{MUTED};text-transform:uppercase;margin-bottom:12px;font-family:JetBrains Mono,monospace'>MAPA DE NIVELES — precio más alto arriba</div>"
    for nivel,tipo,color,desc in all_items:
        is_p = tipo=="PRECIO"
        dist = (price_now-nivel)/price_now*100 if not is_p else 0
        ds   = f"{'▲' if dist>0 else '▼'} {abs(dist):.3f}%" if not is_p else "◀ AQUÍ"
        z_html += f"""<div style='display:flex;align-items:center;gap:12px;padding:{'10px 12px' if is_p else '7px 12px'};
            margin-bottom:3px;background:{'rgba(255,255,255,0.04)' if is_p else 'transparent'};
            border-radius:4px;{"border:1px solid "+color+";" if is_p else ""}'>
          <div style='width:72px;font-family:JetBrains Mono,monospace;font-size:{'14px' if is_p else '12px'};
              font-weight:{"700" if is_p else "400"};color:{color};text-align:right'>{nivel:.{dec}f}</div>
          <div style='flex:1;position:relative;height:5px;background:{BORDER};border-radius:3px'>
            <div style='position:absolute;left:{100-(nivel-mn_z)/span_z*100:.0f}%;top:-4px;
                width:{'10px' if is_p else '6px'};height:{'13px' if is_p else '8px'};
                background:{color};border-radius:2px;transform:translateX(-50%)'></div>
          </div>
          <div style='width:72px;font-size:10px;color:{color};font-weight:{"700" if is_p else "400"};letter-spacing:1px'>{tipo}</div>
          <div style='width:80px;font-size:10px;color:{MUTED};text-align:right'>{ds}</div>
          <div style='font-size:10px;color:{MUTED};flex:2'>{desc}</div>
        </div>"""
    z_html += "</div>"
    st.markdown(z_html, unsafe_allow_html=True)

    zk1,zk2 = st.columns(2)
    distances = {"POC":abs(price_now-poc),"VAH":abs(price_now-vah),"VAL":abs(price_now-val_v),"VWAP":abs(price_now-vwap_now)}
    nearest   = min(distances, key=distances.get)
    nv        = {"POC":poc,"VAH":vah,"VAL":val_v,"VWAP":vwap_now}[nearest]
    nc        = {"POC":ORANGE,"VAH":BLUE,"VAL":BLUE,"VWAP":YELLOW}[nearest]
    with zk1:
        info_card("📍 ZONA MÁS CERCANA",
            f"<b style='color:{nc};font-size:20px'>{nearest} {nv:.{dec}f}</b><br>"
            f"A <b>{distances[nearest]/price_now*100:.3f}%</b> del precio actual. "
            f"{'Imán de precio — alta probabilidad de reacción.' if nearest=='POC' else 'Resistencia clave del Value Area.' if nearest=='VAH' else 'Soporte clave del Value Area.' if nearest=='VAL' else 'Referencia institucional — precio equilibrado.'}", nc)
    with zk2:
        in_va = val_v<=price_now<=vah
        if price_now>vah:
            zt=f"Precio sobre VAH ({vah:.{dec}f}). Compradores en control. VAH es ahora soporte."; zc=GREEN
        elif price_now<val_v:
            zt=f"Precio bajo VAL ({val_v:.{dec}f}). Vendedores en control. VAL es ahora resistencia."; zc=RED
        elif price_now>poc:
            zt=f"En Value Area sobre POC ({poc:.{dec}f}). Equilibrio con sesgo alcista. POC es soporte inmediato."; zc="#69f0ae"
        else:
            zt=f"En Value Area bajo POC ({poc:.{dec}f}). Equilibrio con sesgo bajista. POC es resistencia inmediata."; zc="#ff6b6b"
        info_card("🗺️ ESTRUCTURA DE PRECIO", zt, zc)

    st.divider()

    # Checklist operativa
    st.markdown("### Checklist de Condiciones")
    rr_val = summ.get("real_rates",{}).get("real_rate_approx", None)
    checks = [
        (abs(summ["total"])>=40,        f"Score ≥ 40 (actual: {summ['total']:+.0f})",       "Señal de fuerza direccional suficiente"),
        (summ["conviction"] in("ALTA","MEDIA"), f"Convicción {summ['conviction']}",         "Mínimo 50% de modelos alineados"),
        (abs(r["last_z"])>=1.5,         f"Z-Diff fuera de neutral ({r['last_z']:.3f})",     "Flujo institucional confirmado"),
        (max(r["adj_bull"],r["adj_bear"])>=60, f"MC ≥ 60% ({max(r['adj_bull'],r['adj_bear']):.1f}%)", "Probabilidad estadística suficiente"),
        (vdata["vol_regime"]!="COMPRESIÓN", f"Régimen: {vdata['vol_regime']}",              "Compresión = esperar ruptura antes de entrar"),
        (not (val_v<=price_now<=vah) or abs(price_now-poc)/price_now*100>0.15,
             f"Precio fuera del POC ({poc:.{dec}f})",                                       "En el POC el R:R es peor — zona de equilibrio"),
        (rr_val is None or rr_val>-1,   f"Tipos reales {'N/A (sin datos FRED)' if rr_val is None else f'{rr_val:+.1f}%'}", "Tipos muy negativos = entorno inflacionario distorsionado"),
    ]
    for ok,title,detail in checks:
        col_c=GREEN if ok else RED
        st.markdown(f"""<div style='display:flex;align-items:center;gap:12px;padding:9px 16px;
            margin-bottom:4px;background:{S0};border-left:3px solid {col_c};border-radius:4px'>
            <span style='font-size:16px'>{"✅" if ok else "❌"}</span>
            <div>
                <div style='font-size:12px;color:{TEXT};font-weight:500'>{title}</div>
                <div style='font-size:10px;color:{MUTED}'>{detail}</div>
            </div>
        </div>""", unsafe_allow_html=True)
    ok_n = sum(1 for ok,_,_ in checks if ok)
    msg  = "Condiciones favorables" if ok_n>=5 else "Condiciones parciales" if ok_n>=3 else "Condiciones insuficientes"
    mc2  = GREEN if ok_n>=5 else ORANGE if ok_n>=3 else RED
    st.markdown(f"<div style='text-align:center;margin-top:10px;font-size:12px;color:{mc2}'><b>{ok_n}/{len(checks)} criterios — {msg}</b></div>",unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ① MERCADO
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    df = st.session_state.df
    st.markdown(f"#### {ticker} H4 — Monte Carlo {r['mc_paths'].shape[0]:,} sims · {r['mc_paths'].shape[1]} pasos · Histograma integrado")
    fig_m, bull_pct = chart_price_mc(df, r["mc_paths"])
    st.plotly_chart(fig_m, use_container_width=True)

    cz,ce = st.columns(2)
    with cz:
        info_card(f"⚡ Z-DIFF ORDER FLOW — {zctx['signal']}",
            f"Z = <b style='color:{zctx['color']}'>{r['last_z']:.3f}</b> · "
            f"Precio en <b>{zctx['price_pct']*100:.0f}%</b> del rango · "
            f"{'↑ subiendo' if zctx['rising'] else '↓ cayendo'}"
            f"{'<br>💥 <b>Ruptura alcista activa</b>' if zctx['breaking_up'] else ''}"
            f"{'<br>💥 <b>Ruptura bajista activa</b>' if zctx['breaking_down'] else ''}",
            zctx["color"])
    with ce:
        info_card("🎲 MONTE CARLO GBM",
            f"P(alcista): <b style='color:{GREEN if r['adj_bull']>=60 else RED}'>{r['adj_bull']:.1f}%</b> · "
            f"P(bajista): <b style='color:{RED if r['adj_bear']>=60 else GREEN}'>{r['adj_bear']:.1f}%</b><br>"
            f"IC 90%: [{r['p5']:.{dec}f} – {r['p95']:.{dec}f}]<br>"
            f"Media MC: <b>{float(r['final'].mean()):.{dec}f}</b> · "
            f"Mediana: <b>{float(np.median(r['final'])):.{dec}f}</b>", CYAN)

    st.divider()
    st.markdown("#### 🔗 Cadenas de Markov — Probabilidades de Transición de Estado")
    T      = mk["transition"]; labels=mk["labels"]
    fig_mk = make_subplots(rows=1,cols=2,subplot_titles=["Matriz de Transición","P(Estado) mañana"],horizontal_spacing=0.14)
    fig_mk.add_trace(go.Heatmap(z=T*100,x=labels,y=labels,
        colorscale=[[0,BG],[0.5,"rgba(0,144,255,.4)"],[1,CYAN]],
        text=[[f"{v:.0f}%" for v in row] for row in T*100],
        texttemplate="%{text}",showscale=False),row=1,col=1)
    fig_mk.add_trace(go.Bar(x=labels,y=nd*100,marker_color=[RED,YELLOW,GREEN],
        text=[f"{v*100:.1f}%" for v in nd],textposition="outside",showlegend=False),row=1,col=2)
    fig_mk.update_layout(template="plotly_dark",paper_bgcolor=BG,plot_bgcolor=S1,
        height=320,margin=dict(l=8,r=8,t=40,b=8))
    st.plotly_chart(fig_mk, use_container_width=True)

    m1,m2,m3 = st.columns(3)
    for col,lbl,val,clr in zip([m1,m2,m3],mk["labels"],nd,[RED,YELLOW,GREEN]):
        kpi(col,f"P({lbl})",f"{val*100:.1f}%",f"Estado actual: {mk['current_label']}",clr)

    st.divider()
    st.markdown("#### 📊 Niveles MC + Calculadora de Posición")
    pc1,pc2,pc3,pc4 = st.columns(4)
    kpi(pc1,f"P(positivo {horizon}d)",f"{r['adj_bull']:.1f}%","Monte Carlo GBM",GREEN)
    kpi(pc2,f"P(negativo {horizon}d)",f"{r['adj_bear']:.1f}%","Monte Carlo GBM",RED)
    kpi(pc3,"Media MC",f"{float(r['final'].mean()):.{dec}f}",f"mediana {float(np.median(r['final'])):.{dec}f}",CYAN)
    kpi(pc4,"Dispersión 1σ",f"±{float(r['final'].std()):.{dec}f}",f"IC 90%: {r['p5']:.{dec}f}–{r['p95']:.{dec}f}",ORANGE)

    prob     = r["adj_bull"] if r["adj_bull"]>r["adj_bear"] else r["adj_bear"]
    prim_bull= r["adj_bull"]>r["adj_bear"]
    sl_      = r["p8"]  if prim_bull else r["p92"]
    tp_      = r["p80"] if prim_bull else r["p20"]
    en_      = float(r["final"].mean()) if zctx.get("pattern","") in ["ruptura_momentum","ruptura_confirmada"] else (r["p80"] if prim_bull else r["p20"])
    rr_ratio = abs(tp_-en_)/max(abs(en_-sl_),1e-10)
    risk_usd = account*(risk_pct/100)
    sl_dist  = abs(en_-sl_)
    if instr=="Forex std (100k)":   lots=risk_usd/((sl_dist/0.0001)*10); ll=f"{lots:.2f} lotes std"
    elif instr=="Forex mini (10k)": lots=risk_usd/((sl_dist/0.0001)*1);  ll=f"{lots:.2f} mini lotes"
    elif instr=="XAU/USD":          lots=risk_usd/(sl_dist*100);          ll=f"{lots:.3f} lotes XAU"
    else:                           lots=risk_usd/max(sl_dist,1e-10);     ll=f"{lots:.2f} contratos"
    sc_ = GREEN if prim_bull else RED
    st.markdown(f"""<div class='card'>
        <div style='display:grid;grid-template-columns:repeat(5,1fr);gap:16px;align-items:center'>
          <div style='text-align:center;padding:12px;background:{"rgba(0,230,118,.08)" if prim_bull else "rgba(255,23,68,.08)"};
              border:1px solid {sc_};border-radius:6px'>
            <div style='font-family:Rajdhani,sans-serif;font-size:22px;font-weight:700;color:{sc_}'>{"LARGO" if prim_bull else "CORTO"}</div>
            <div style='font-size:11px;color:{sc_}'>{"BUY STOP/LIMIT" if prim_bull else "SELL STOP/LIMIT"}</div>
          </div>
          <div>
            <div style='font-family:Rajdhani,sans-serif;font-size:24px;font-weight:700'>{en_:.{dec}f}</div>
            <div style='font-size:10px;color:{MUTED}'>nivel de entrada</div>
          </div>
          <div style='font-size:12px;line-height:2'>
            SL <span style='color:{RED};font-weight:600'>{sl_:.{dec}f}</span><br>
            TP <span style='color:{GREEN};font-weight:600'>{tp_:.{dec}f}</span><br>
            RR <span>1:{rr_ratio:.1f}</span>
          </div>
          <div style='font-size:11px;color:{MUTED};line-height:1.7'>
            {ll}<br>Riesgo <span style='color:{RED}'>${risk_usd:.0f}</span><br>
            Potencial <span style='color:{GREEN}'>${risk_usd*rr_ratio:.0f}</span>
          </div>
          <div style='text-align:right'>
            <div style='font-family:Rajdhani,sans-serif;font-size:36px;font-weight:700;color:{sc_}'>{prob:.1f}%</div>
            <div style='font-size:10px;color:{MUTED}'>probabilidad MC</div>
          </div>
        </div>
        <div style='font-size:10px;color:{MUTED};margin-top:10px;padding-top:10px;border-top:1px solid {BORDER}'>
            Niveles derivados de percentiles de la distribución Monte Carlo (p8/p80 largo · p92/p20 corto).
            Estos no son consejos de inversión — úsalos como referencia estadística.
        </div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ② VOLATILIDAD
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    vd  = r["vol_data"]; dec=1 if price>1000 else 2 if price>100 else 4
    v1,v2,v3,v4,v5,v6 = st.columns(6)
    kpi(v1,"Vol Realizada (14v)",f"{vd['rv_current']*100:.2f}%","anualizada",ORANGE)
    kpi(v2,"Vol Parkinson (H-L)",f"{vd['parkinson']*100:.2f}%","estimador H-L",PURPLE)
    kpi(v3,"Vol Garman-Klass",f"{vd['garman_klass']*100:.2f}%","estimador OHLC",BLUE)
    kpi(v4,"Régimen",vd["vol_regime"],f"σ corto {vd['rv_short']*100:.1f}% / largo {vd['rv_long']*100:.1f}%",vd["vol_color"])
    kpi(v5,"Movimiento 1σ/día",f"±{vd['price_1s']:.{dec}f}",f"±{vd['price_1s']/price*100:.3f}%",CYAN)
    kpi(v6,"Movimiento 2σ/día",f"±{vd['price_2s']:.{dec}f}",f"±{vd['price_2s']/price*100:.3f}%",YELLOW)

    st.plotly_chart(chart_volatility(st.session_state.df, vd, r["mc_paths"]), use_container_width=True)

    st.markdown("#### ATR Multi-Periodo")
    rows=[]
    for w,av in vd["atr"].items():
        rows.append({"Periodo":f"{w}v H4 (~{w//6:.0f}d)","ATR":f"{av:.{dec}f}",
                     "ATR %":f"{av/price*100:.3f}%","SL 1×ATR":f"{av:.{dec}f}",
                     "TP 2×ATR":f"{av*2:.{dec}f}","TP 3×ATR":f"{av*3:.{dec}f}"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    vi1,vi2 = st.columns(2)
    with vi1:
        reg_txt = ("🔴 <b>EXPANSIÓN</b> — Vol corta >> larga. Movimiento fuerte en curso. Aumenta SL/TP, reduce tamaño."
                   if vd["vol_regime"]=="EXPANSIÓN" else
                   "🔵 <b>COMPRESIÓN</b> — Vol corta << larga. Mercado comprimido — ruptura inminente. Espera confirmación."
                   if vd["vol_regime"]=="COMPRESIÓN" else
                   "⚪ <b>NORMAL</b> — Régimen estable. Parámetros estándar. Usa ATR como referencia directa.")
        info_card("RÉGIMEN DE VOLATILIDAD", reg_txt, vd["vol_color"])
    with vi2:
        info_card("NIVELES ESTADÍSTICOS MAÑANA",
            f"Precio actual: <b>{price:.{dec}f}</b><br>"
            f"1σ alcista (68%): <b style='color:{GREEN}'>{price+vd['price_1s']:.{dec}f}</b> · "
            f"1σ bajista: <b style='color:{RED}'>{price-vd['price_1s']:.{dec}f}</b><br>"
            f"2σ alcista (95%): <b style='color:{GREEN}'>{price+vd['price_2s']:.{dec}f}</b> · "
            f"2σ bajista: <b style='color:{RED}'>{price-vd['price_2s']:.{dec}f}</b>", CYAN)

    rets_arr = np.diff(np.log(st.session_state.df["Close"].values.astype(float)))
    kurt=float(stats.kurtosis(rets_arr)); skew=float(stats.skew(rets_arr))
    if abs(kurt)>1:
        st.warning(f"**Colas gordas** — Kurtosis={kurt:.2f}. Movimientos extremos más frecuentes de lo que asume el modelo GBM. Usa percentiles p5/p95 como referencia.")


# ─────────────────────────────────────────────────────────────────────────────
# ③ VOLUMEN
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    vp   = r["vol_profile"]; vwap=r["vwap_series"]; delt=r["delta_series"]
    danom= r["df_anom"];     df=st.session_state.df; pn=r["price"]
    dec  = 1 if pn>1000 else 2 if pn>100 else 4
    cum_d= float(delt.sum()); rec_d=float(delt.iloc[-6:].sum())
    vwap_v=float(vwap.iloc[-1])

    va1,va2,va3,va4,va5,va6 = st.columns(6)
    kpi(va1,"POC",f"{vp['poc']:.{dec}f}",f"{'▲' if pn>vp['poc'] else '▼'} {abs((pn-vp['poc'])/pn*100):.3f}%",GREEN if pn>vp["poc"] else RED)
    kpi(va2,"VAH",f"{vp['vah']:.{dec}f}","70% del volumen",BLUE)
    kpi(va3,"VAL",f"{vp['val']:.{dec}f}","70% del volumen",BLUE)
    kpi(va4,"VWAP",f"{vwap_v:.{dec}f}","precio medio ponderado",GREEN if pn>vwap_v else RED)
    kpi(va5,"Delta Acumulado",f"{cum_d:+,.0f}","comprador/vendedor",GREEN if cum_d>0 else RED)
    n_sp=int((danom["anomaly"].isin(["ABSORCIÓN","SPIKE VOL"])).sum())
    kpi(va6,"Anomalías Vol",f"{n_sp} spikes","velas anómalas",ORANGE if n_sp>0 else MUTED)

    st.plotly_chart(chart_volume_profile(df,vp,vwap,delt,danom), use_container_width=True)

    cp1,cp2 = st.columns(2)
    with cp1:
        if vp["val"]<=pn<=vp["vah"]:
            pi="Precio en <b>Value Area</b>. Mercado en equilibrio — sin dirección institucional clara. Tiende a regresar al POC."; pc=YELLOW
        elif pn>vp["vah"]:
            pi=f"Precio <b>sobre VAH</b> ({vp['vah']:.{dec}f}). Compradores en control. VAH es soporte en retrocesos."; pc=GREEN
        else:
            pi=f"Precio <b>bajo VAL</b> ({vp['val']:.{dec}f}). Vendedores en control. VAL es resistencia en rebotes."; pc=RED
        info_card("📍 POC · VALUE AREA", pi, pc)
    with cp2:
        if pn>vwap_v*1.002:
            vi2="Precio <b>sobre VWAP</b>. Compradores pagando encima de la media. VWAP es soporte dinámico."; vc2=GREEN
        elif pn<vwap_v*0.998:
            vi2="Precio <b>bajo VWAP</b>. Vendedores dominan. VWAP es resistencia dinámica."; vc2=RED
        else:
            vi2="Precio <b>en el VWAP</b>. Equilibrio comprador/vendedor. Espera separación con volumen."; vc2=YELLOW
        info_card("📈 VWAP", vi2, vc2)

    st.markdown("#### ⚡ Volume Delta")
    delta_trend="compradora" if rec_d>0 else "vendedora"
    delta_str  = f"{rec_d:+,.0f}; Acumulado: {cum_d:+,.0f}"
    delta_diag = (f"Delta acumulado <b>{'positivo' if cum_d>0 else 'negativo'}</b> con último día "
                  f"<b>{'confirmando' if (rec_d>0)==(cum_d>0) else 'divergiendo'}</b>. "
                  f"{'⚠️ <b>Divergencia:</b> delta reciente vs acumulado opuesto — posible giro.' if (rec_d>0)!=(cum_d>0) else ''}")
    info_card("DIAGNÓSTICO VOLUME DELTA", delta_diag, GREEN if cum_d>0 else RED)

    st.markdown("#### 🔍 Anomalías de Volumen")
    anom_df = danom[danom["anomaly"]!="NORMAL"][
        ["Open","High","Low","Close","vol_eff","vol_z","anomaly","anom_score"]].copy().tail(20)
    if len(anom_df)>0:
        anom_df.columns=["Open","High","Low","Close","Vol","Z-Score","Tipo","Score"]
        anom_df=anom_df.round({"Open":dec,"High":dec,"Low":dec,"Close":dec,"Vol":0,"Z-Score":2,"Score":1})
        def _ca(v): return {"ABSORCIÓN":f"color:{PURPLE}","SPIKE VOL":f"color:{ORANGE}","MOMENTUM":f"color:{CYAN}","RUPTURA SECA":f"color:{YELLOW}"}.get(v,f"color:{MUTED}")
        def _cz(v): return f"color:{ORANGE};font-weight:bold" if v>2.5 else f"color:{YELLOW}" if v>1.8 else f"color:{MUTED}"
        _s=anom_df.style; _fn="map" if hasattr(_s,"map") else "applymap"
        st.dataframe(getattr(_s,_fn)(_ca,subset=["Tipo"]).pipe(lambda s:getattr(s,_fn)(_cz,subset=["Z-Score"])), use_container_width=True)
    else:
        st.info("No se detectaron anomalías de volumen significativas.")


# ─────────────────────────────────────────────────────────────────────────────
# ④ MACRO CUANTITATIVA
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    macro = st.session_state.macro_data

    if not macro:
        st.info("👈 Pulsa **Cargar Macro FRED** en la barra lateral para cargar datos macroeconómicos gratuitos.")
        st.markdown("""
        **Datos disponibles (Federal Reserve — sin API key):**
        - **Tipos reales**: Fed Funds Rate, PCE Core, TIPS (2Y, 5Y, 10Y)
        - **Curva de tipos**: Treasuries 2Y, 5Y, 10Y, 30Y + spreads
        - **Inflación**: CPI, PCE, Breakeven 10Y
        - **Condiciones financieras**: Chicago Fed NFCI, VIX
        - **Macro real**: Desempleo, Nóminas no agrícolas
        """)
    else:
        rr = calc_real_rates(macro)

        # KPIs macro
        mk1,mk2,mk3,mk4,mk5,mk6 = st.columns(6)
        kpi(mk1,"Fed Funds Rate",f"{rr.get('fed_funds',macro.get('FEDFUNDS',{}).get('latest','N/A'))}%","tipo nominal actual",CYAN)
        kpi(mk2,"PCE Core",f"{rr.get('pce_core',macro.get('PCEPILFE',{}).get('latest','N/A'))}%","inflación objetivo Fed",ORANGE)
        rra=rr.get("real_rate_approx","N/A")
        kpi(mk3,"Tipo Real Aprox",f"{rra:+.2f}%" if isinstance(rra,float) else "N/A","FF - PCE Core",GREEN if isinstance(rra,float) and rra>0 else RED)
        tips10=rr.get("DFII10","N/A"); kpi(mk4,"TIPS 10Y",f"{tips10}%" if tips10!="N/A" else "N/A","tipo real mercado",GREEN if isinstance(tips10,float) and tips10>0 else RED)
        sp=rr.get("spread_10y2y","N/A"); kpi(mk5,"Spread 10Y-2Y",f"{sp:+.2f}%" if isinstance(sp,float) else "N/A","invertida = recesión",RED if isinstance(sp,float) and sp<0 else GREEN)
        be=rr.get("breakeven_10y","N/A"); kpi(mk6,"Breakeven 10Y",f"{be}%" if be!="N/A" else "N/A","inflación esperada",ORANGE)

        # Gráfico tipos reales + curva
        st.markdown("#### Tipos Reales · Curva de Tipos · Spreads")
        st.plotly_chart(chart_real_rates(macro), use_container_width=True)

        # Gráfico indicadores secundarios
        st.markdown("#### Condiciones de Mercado")
        st.plotly_chart(chart_macro_indicators(macro), use_container_width=True)

        # Interpretación
        st.markdown("#### Interpretación Macroeconómica")
        mi1,mi2,mi3 = st.columns(3)
        with mi1:
            if isinstance(rra,float):
                if rra>2:
                    rt_txt="<b>Tipos reales muy positivos</b> — entorno restrictivo. La Fed está por encima de la inflación. Presión sobre activos de riesgo, favorable para el dólar."
                    rt_c=RED
                elif rra>0:
                    rt_txt="<b>Tipos reales positivos</b> — política moderadamente restrictiva. Equilibrio entre crecimiento e inflación."
                    rt_c=YELLOW
                elif rra>-1:
                    rt_txt="<b>Tipos reales ligeramente negativos</b> — política levemente expansiva. Contexto favorable para activos de riesgo."
                    rt_c="#69f0ae"
                else:
                    rt_txt="<b>Tipos reales muy negativos</b> — represión financiera. Históricamente favorable para oro y activos reales."
                    rt_c=GREEN
                info_card("ENTORNO DE TIPOS REALES", rt_txt, rt_c)
        with mi2:
            if isinstance(sp,float):
                if sp<-0.5:
                    cur_txt=f"<b>Curva invertida ({sp:+.2f}%)</b> — señal histórica de recesión. Los inversores esperan bajadas de tipos. Negativo para cíclicos."
                    cur_c=RED
                elif sp<0:
                    cur_txt=f"<b>Curva ligeramente invertida ({sp:+.2f}%)</b> — precaución. Mercados anticipan desaceleración."
                    cur_c=ORANGE
                elif sp<0.5:
                    cur_txt=f"<b>Curva plana ({sp:+.2f}%)</b> — transición. Sin señal direccional clara."
                    cur_c=YELLOW
                else:
                    cur_txt=f"<b>Curva normal ({sp:+.2f}%)</b> — entorno expansivo. Los mercados no anticipan recesión inmediata."
                    cur_c=GREEN
                info_card("CURVA DE TIPOS", cur_txt, cur_c)
        with mi3:
            vix_val=macro.get("VIXCLS",{}).get("latest","N/A")
            nfci_val=macro.get("NFCI",{}).get("latest","N/A")
            if isinstance(vix_val,(int,float)):
                if vix_val>30:
                    vix_txt=f"<b>VIX alto ({vix_val:.1f})</b> — pánico de mercado. Volatilidad implícita elevada. Históricamente buena zona de compra a largo plazo."
                    vix_c=RED
                elif vix_val>20:
                    vix_txt=f"<b>VIX moderado ({vix_val:.1f})</b> — cautela. Mercados en modo risk-off parcial."
                    vix_c=ORANGE
                else:
                    vix_txt=f"<b>VIX bajo ({vix_val:.1f})</b> — complacencia. Mercados tranquilos. Riesgo de spike si hay sorpresa negativa."
                    vix_c=GREEN
                info_card("VIX & CONDICIONES FINANCIERAS", vix_txt+
                          (f"<br>NFCI: <b>{nfci_val:.2f}</b> {'(restrictivo)' if isinstance(nfci_val,(int,float)) and nfci_val>0 else '(acomodaticio)'}" if isinstance(nfci_val,(int,float)) else ""), vix_c)

        # Tabla resumen
        st.markdown("#### Tabla de Datos FRED")
        fred_rows=[]; 
        for sid,d in macro.items():
            s=d["series"]
            fred_rows.append({
                "Serie":d["name"],"Código":sid,
                "Actual":f"{d['latest']:.2f}",
                "Hace 1m":f"{float(s.iloc[-2]):.2f}" if len(s)>=2 else "N/A",
                "Hace 6m":f"{float(s.iloc[-6]):.2f}" if len(s)>=6 else "N/A",
                "Hace 1a":f"{float(s.iloc[-13]):.2f}" if len(s)>=13 else "N/A",
                "Variación":f"{float(s.iloc[-1])-float(s.iloc[-2]):+.2f}" if len(s)>=2 else "N/A"
            })
        if fred_rows:
            st.dataframe(pd.DataFrame(fred_rows), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# ⑤ IA — GEMINI
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    gkey = st.session_state.get("gemini_key","")

    if not gkey:
        st.info("👈 Introduce tu **Gemini API Key** en la barra lateral.\n\nObtén una gratis en [aistudio.google.com/apikey](https://aistudio.google.com/apikey)")
    else:
        ai_prompt = build_ai_prompt(ticker, asset_type, r, summ,
                                     st.session_state.macro_data or {})

        with st.expander("📋 Prompt enviado a Gemini", expanded=False):
            st.code(ai_prompt, language="text")
            st.caption("Este prompt incluye todos los datos cuantitativos del modelo.")

        col_ai1, col_ai2 = st.columns([1,3])
        with col_ai1:
            if st.button("🤖 Generar análisis Gemini", use_container_width=True, type="primary"):
                with st.spinner("Gemini analizando los datos del modelo..."):
                    result = call_gemini(ai_prompt, gkey)
                    st.session_state.ai_analysis = result
            if st.button("🔄 Regenerar", use_container_width=True):
                with st.spinner("Regenerando análisis..."):
                    st.session_state.ai_analysis = call_gemini(ai_prompt, gkey)

        if st.session_state.ai_analysis:
            st.markdown("#### Análisis Cuantitativo — Gemini 2.0 Flash")
            st.markdown(f"""<div class='card' style='border-left:4px solid {PURPLE}'>
                <div style='font-size:10px;letter-spacing:2px;color:{PURPLE};
                    text-transform:uppercase;margin-bottom:12px;font-family:JetBrains Mono,monospace'>
                    ⬡ ANÁLISIS GENERADO POR IA — SOLO INFORMATIVO, NO ASESORAMIENTO FINANCIERO
                </div>
                <div style='font-size:13px;color:{TEXT};line-height:1.85;white-space:pre-wrap'>{st.session_state.ai_analysis}</div>
            </div>""", unsafe_allow_html=True)

            # Datos del modelo que alimentaron el análisis
            st.markdown("#### Datos del modelo utilizados")
            d1,d2,d3,d4 = st.columns(4)
            kpi(d1,"Z-Diff",f"{r['last_z']:.3f}",zctx["signal"],zctx["color"])
            kpi(d2,"MC P(alcista)",f"{r['adj_bull']:.1f}%","Monte Carlo GBM",GREEN if r["adj_bull"]>=60 else RED)
            kpi(d3,"Markov mañana",mk["labels"][int(np.argmax(nd))],f"{max(nd)*100:.0f}%",[RED,YELLOW,GREEN][int(np.argmax(nd))])
            rr2=summ.get("real_rates",{}); rra2=rr2.get("real_rate_approx","N/A")
            kpi(d4,"Tipo Real",f"{rra2:+.2f}%" if isinstance(rra2,float) else "N/A","FF - PCE Core",GREEN if isinstance(rra2,float) and rra2<0 else RED)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown(f"""<div style='font-size:10px;color:{MUTED};text-align:center;line-height:2'>
⚠️ <b>QuantEdge PRO</b> es una herramienta de análisis cuantitativo educativa.
No constituye asesoramiento financiero ni recomendación de inversión.<br>
Monte Carlo GBM · Z-Diff Order Flow · Cadenas de Markov · Volatilidad Parkinson/Garman-Klass
· Datos FRED (Federal Reserve) · Gemini AI<br>
Yahoo Finance H4 · Tipos reales FRED · Sin API key de mercado requerida
</div>""", unsafe_allow_html=True)
