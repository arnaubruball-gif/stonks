"""
OrderFlow PRO — Dashboard Cuantitativo Profesional
===================================================
3 pestañas:
  1. Direccionalidad — Z-Diff, Markov, MC histogram en el gráfico
  2. Volatilidad     — ATR, cono de volatilidad, bandas de probabilidad
  3. Macro           — Contexto, correlaciones, calendario de riesgo
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from datetime import datetime, timedelta
import json, re, warnings
warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OrderFlow PRO — Dashboard",
    page_icon="📊", layout="wide",
    initial_sidebar_state="expanded"
)

# ─── THEME ────────────────────────────────────────────────────────────────────
BG     = "#04070d"
S0     = "#060b13"
S1     = "#0a1019"
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
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Rajdhani:wght@600;700&display=swap');
html,body,[class*="css"]{{font-family:'JetBrains Mono',monospace;}}
.stTabs [data-baseweb="tab-list"]{{gap:4px;background:{S1};padding:4px;border-radius:4px;border:1px solid {BORDER};}}
.stTabs [data-baseweb="tab"]{{background:transparent;color:{MUTED};font-family:'Rajdhani',sans-serif;
  font-size:14px;font-weight:600;letter-spacing:2px;padding:8px 20px;border-radius:3px;}}
.stTabs [aria-selected="true"]{{background:{BG};color:{CYAN};border-bottom:2px solid {CYAN};}}
.kpi{{background:{S1};border:1px solid {BORDER};border-radius:4px;padding:14px 18px;margin-bottom:8px;}}
.kpi-lbl{{font-size:9px;letter-spacing:3px;text-transform:uppercase;color:{MUTED};margin-bottom:4px;}}
.kpi-val{{font-family:'Rajdhani',sans-serif;font-size:28px;font-weight:700;line-height:1;}}
.kpi-sub{{font-size:10px;color:{MUTED};margin-top:3px;}}
.signal-box{{border-radius:4px;padding:14px 18px;margin-bottom:10px;}}
.entry-box{{background:{S0};border:1px solid {BORDER};border-left:4px solid {CYAN};
  border-radius:4px;padding:16px 20px;margin:10px 0;}}
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
TF_INTERVAL = "4h"
TF_PERIOD   = "60d"
TF_LABEL    = "H4"
H4_PER_DAY  = 6
TRADING_DAYS= 252

# Futures cotizan ~23h/día — sin gaps de mercado cerrado
# Spot indices (^GSPC etc) solo tienen datos en horario NY
QUICK_MAP = {
    "EUR/USD":          ("EURUSD=X",  "forex"),
    "GBP/USD":          ("GBPUSD=X",  "forex"),
    "USD/JPY":          ("USDJPY=X",  "forex"),
    "XAU/USD 🥇":       ("GC=F",      "commodity"),
    "S&P 500 🔄":       ("ES=F",      "index"),    # Futuro S&P — 23h/día
    "NASDAQ 🔄":        ("NQ=F",      "index"),    # Futuro Nasdaq — 23h/día
    "DOW JONES 🔄":     ("YM=F",      "index"),    # Futuro Dow — 23h/día
    "DAX 🔄":           ("FDAX=F",    "index"),    # Futuro DAX — 23h/día
    "CRUDE OIL 🔄":     ("CL=F",      "commodity"),# Futuro WTI — 23h/día
    "S&P 500 (spot)":   ("^GSPC",     "index"),    # Solo horario NY
    "DAX (spot)":       ("^GDAXI",    "index"),    # Solo horario EU
    "NASDAQ (spot)":    ("^IXIC",     "index"),    # Solo horario NY
    "— Manual —":       ("",          "forex"),
}

# Nota: símbolos 🔄 = futuros continuos, cotizan casi 24h
FUTURES_NOTE = {
    "ES=F": "S&P 500 E-mini Futures",
    "NQ=F": "Nasdaq 100 E-mini Futures",
    "YM=F": "Dow Jones E-mini Futures",
    "FDAX=F": "DAX Futures (Eurex)",
    "CL=F": "Crude Oil WTI Futures",
}

# ─── SESSION STATE ────────────────────────────────────────────────────────────
for k in ["df","results","context","macro_prompt"]:
    if k not in st.session_state:
        st.session_state[k] = None

# ═══════════════════════════════════════════════════════════════════════════════
#  QUANTITATIVE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def calc_order_flow(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    df["tp"]      = (df["High"] + df["Low"] + df["Close"]) / 3
    df["tp_prev"] = df["tp"].shift(1)
    vol = df["Volume"].fillna(0)
    if vol.sum() == 0 or vol.nunique() <= 3:
        rng = df["High"] - df["Low"]
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


def calc_markov(df: pd.DataFrame, n_states: int = 3) -> dict:
    returns = df["Close"].pct_change().dropna()
    b0, b1  = returns.quantile(0.33), returns.quantile(0.67)
    labels  = ["BAJISTA", "NEUTRAL", "ALCISTA"]

    def label(r):
        if r <= b0:  return 0
        elif r <= b1: return 1
        else:         return 2

    states = returns.apply(label).values
    T = np.zeros((3, 3))
    for i in range(len(states)-1):
        T[states[i], states[i+1]] += 1
    rs = T.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    T = T / rs

    try:
        vals, vecs = np.linalg.eig(T.T)
        si  = np.argmin(np.abs(vals - 1))
        stat = np.abs(vecs[:, si].real)
        stat = stat / stat.sum()
    except Exception:
        stat = np.ones(3) / 3

    current = int(states[-1])
    dist    = np.zeros(3); dist[current] = 1.0
    next_d  = dist @ np.linalg.matrix_power(T, H4_PER_DAY)
    next_3d = dist @ np.linalg.matrix_power(T, H4_PER_DAY*3)

    return {
        "transition": T, "labels": labels,
        "current_state": current, "current_label": labels[current],
        "next_day": next_d, "next_3day": next_3d,
        "stationary": stat, "returns": returns,
    }


def run_mc(price, returns, sims=3000, steps=6, z_adj=0.0, vol_mult=1.0):
    mu    = returns.mean()
    sigma = returns.std() * vol_mult
    drift = mu + z_adj * sigma * 0.15
    rng   = np.random.default_rng(42)
    eps   = rng.standard_normal((sims, steps))
    paths = price * np.exp(((drift - 0.5*sigma**2) + sigma*eps).cumsum(axis=1))
    return paths


def calc_volatility(df: pd.DataFrame) -> dict:
    c = df["Close"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    r = np.diff(np.log(c))

    tr  = np.maximum(h[1:]-l[1:],
          np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    atr = {w: float(pd.Series(tr).rolling(w).mean().iloc[-1])
           for w in [5, 14, 20, 50] if w <= len(tr)}

    rv_s   = pd.Series(r).rolling(14).std() * np.sqrt(TRADING_DAYS * H4_PER_DAY)
    rv_cur = float(rv_s.iloc[-1]) if not np.isnan(rv_s.iloc[-1]) else float(r[-14:].std() * np.sqrt(TRADING_DAYS*H4_PER_DAY))

    hl  = np.log(h[1:]/l[1:])
    pk  = float(np.sqrt((hl**2 / (4*np.log(2)))[-14:].mean() * TRADING_DAYS*H4_PER_DAY))
    gk  = 0.5*hl**2 - (2*np.log(2)-1)*(np.log(c[1:]/c[:-1]))**2
    gkv = float(np.sqrt(gk[-14:].mean() * TRADING_DAYS*H4_PER_DAY))

    rs  = float(pd.Series(r).rolling(5).std().iloc[-1]  * np.sqrt(TRADING_DAYS*H4_PER_DAY))
    rl  = float(pd.Series(r).rolling(20).std().iloc[-1] * np.sqrt(TRADING_DAYS*H4_PER_DAY)) if len(r)>=20 else rv_cur

    if rs > rl*1.3:   reg, rc = "EXPANSIÓN",  ORANGE
    elif rs < rl*0.7: reg, rc = "COMPRESIÓN", CYAN
    else:             reg, rc = "NORMAL",      YELLOW

    daily_s  = float(r[-14:].std())
    p        = float(c[-1])
    s1d      = p * daily_s * np.sqrt(H4_PER_DAY)

    return {
        "atr": atr, "rv_current": rv_cur, "rv_series": rv_s,
        "parkinson": pk, "garman_klass": gkv,
        "rv_short": rs, "rv_long": rl,
        "vol_regime": reg, "vol_color": rc,
        "sigma_1d": daily_s, "price_1s": s1d, "price_2s": s1d*2,
        "tr_series": pd.Series(tr),
    }


def interpret_zdiff(z, df, macro=0):
    c = df["Close"].values; h = df["High"].values; l = df["Low"].values
    n = len(c)
    lb = min(14,n)
    rh = h[-lb:].max(); rl_v = l[-lb:].min()
    rspan = rh - rl_v if rh != rl_v else 1e-10
    ppct  = (c[-1] - rl_v) / rspan
    in_top = ppct > 0.75; in_bot = ppct < 0.25
    ar = c[-3:].mean() if n>=3 else c[-1]
    ap = c[-6:-3].mean() if n>=6 else c[0]
    rising = ar > ap
    ph = h[-6:-1].max() if n>=6 else rh
    pl = l[-6:-1].min() if n>=6 else rl_v
    bu = c[-1] > ph; bd = c[-1] < pl
    az = abs(z)

    if az > 2.2:
        if z > 0:
            if in_top and not bu:
                sig,lbl,col,bull,pat = "DISTRIBUCIÓN EN TECHO","▼ VENTA — distribución en máximos",RED,False,"sobreextension"
                expl  = f"Z extremo ({z:.2f}) con precio en techo ({ppct*100:.0f}% del rango) sin ruptura. Distribución institucional clásica."
                entry = f"**Sobreextensión alcista.** Flujo agotado estadísticamente en máximos. **SELL LIMIT** en el nivel actual. No perseguir con Stop."
            elif bu:
                sig,lbl,col,bull,pat = "RUPTURA ALCISTA","▲ COMPRA — ruptura institucional",GREEN,True,"ruptura_momentum"
                expl  = f"Z extremo ({z:.2f}) confirmando ruptura del máximo previo. Flujo real."
                entry = f"**Ruptura con flujo institucional.** El Z valida la ruptura — no es falsa. **BUY STOP** por encima del máximo roto."
            elif in_bot:
                sig,lbl,col,bull,pat = "ACUMULACIÓN OCULTA","▲ COMPRA — acumulación en mínimos",GREEN,True,"acumulacion_oculta"
                expl  = f"Z extremo ({z:.2f}) con precio en mínimos ({ppct*100:.0f}%). Acumulación silenciosa — Wyckoff Phase B/C."
                entry = f"**Acumulación oculta en suelos.** Institucionales comprando en zona de valor. **BUY LIMIT** escalonado en la zona baja."
            else:
                sig,lbl,col,bull,pat = "AGOTAMIENTO","⚠ Z EXTREMO — zona media",ORANGE,None,"extremo_medio"
                expl  = f"Z extremo ({z:.2f}) en zona media. Flujo insostenible."
                entry = f"**Agotamiento en zona media.** Sin contexto claro. Esperar extremo de rango o ruptura confirmada."
        else:
            if in_bot and not bd:
                sig,lbl,col,bull,pat = "CAPITULACIÓN","▲ COMPRA — capitulación en suelos",GREEN,True,"sobreextension"
                expl  = f"Z extremo negativo ({z:.2f}) en suelos sin ruptura. Capitulación vendedora."
                entry = f"**Sobreextensión bajista / Capitulación.** Flujo vendedor agotado. **BUY LIMIT** en soporte."
            elif bd:
                sig,lbl,col,bull,pat = "RUPTURA BAJISTA","▼ VENTA — ruptura institucional",RED,False,"ruptura_momentum"
                expl  = f"Z extremo negativo ({z:.2f}) confirmando ruptura del mínimo previo."
                entry = f"**Ruptura bajista con flujo.** **SELL STOP** bajo el mínimo roto."
            elif in_top:
                sig,lbl,col,bull,pat = "DISTRIBUCIÓN OCULTA","▼ VENTA — distribución en máximos",RED,False,"distribucion_oculta"
                expl  = f"Z extremo negativo ({z:.2f}) con precio en máximos. Distribución silenciosa."
                entry = f"**Distribución oculta en techo.** Institucionales vendiendo en altos. **SELL LIMIT** en el nivel actual."
            else:
                sig,lbl,col,bull,pat = "AGOTAMIENTO BAJISTA","⚠ Z EXTREMO NEGATIVO",ORANGE,None,"extremo_medio"
                expl  = f"Z extremo negativo ({z:.2f}) zona media. Posible rebote."
                entry = f"**Agotamiento bajista en zona media.** Esperar confirmación."
    elif az > 1.5:
        if z > 0:
            if bu or (rising and in_top and macro >= 0):
                sig,lbl,col,bull,pat = "COMPRA MOMENTUM","▲ COMPRA — momentum confirmado",GREEN,True,"ruptura_confirmada"
                expl  = f"Z {z:.2f} con {'ruptura alcista' if bu else 'precio alto subiendo'}. Flujo y precio alineados."
                entry = f"**Ruptura/Momentum alcista.** **BUY STOP** en ruptura del rango."
            elif in_bot and rising:
                sig,lbl,col,bull,pat = "REBOTE EN SOPORTE","▲ COMPRA — rebote confirmado","#69f0ae",True,"rebote_confirmado"
                expl  = f"Z {z:.2f} con precio rebotando desde mínimos ({ppct*100:.0f}%)."
                entry = f"**Rebote en soporte con flujo.** **BUY LIMIT** en retroceso al soporte. Mejor R:R."
            elif in_top and not rising:
                sig,lbl,col,bull,pat = "DIVERGENCIA","▼ DIVERGENCIA — flujo no confirma",RED,False,"divergencia"
                expl  = f"Z alto ({z:.2f}) pero precio cediendo desde máximos. Distribución."
                entry = f"**Divergencia alcista-precio.** Flujo alto pero precio cae — distribución. **SELL LIMIT**."
            else:
                sig,lbl,col,bull,pat = "SESGO ALCISTA","↑ SESGO LARGO","#69f0ae",True,"momentum_moderado"
                expl  = f"Flujo positivo ({z:.2f}), precio en {ppct*100:.0f}% del rango."
                entry = f"**Momentum alcista moderado.** **BUY STOP** en ruptura del rango reciente."
        else:
            if bd or (not rising and in_bot and macro <= 0):
                sig,lbl,col,bull,pat = "VENTA MOMENTUM","▼ VENTA — momentum confirmado",RED,False,"ruptura_confirmada"
                expl  = f"Z {z:.2f} con {'ruptura bajista' if bd else 'precio bajo cayendo'}."
                entry = f"**Ruptura/Momentum bajista.** **SELL STOP** en ruptura."
            elif in_top and not rising:
                sig,lbl,col,bull,pat = "RECHAZO RESISTENCIA","▼ RECHAZO — flujo confirma",RED,False,"rebote_confirmado"
                expl  = f"Z {z:.2f} negativo con precio rechazando desde máximos ({ppct*100:.0f}%)."
                entry = f"**Rechazo en resistencia con flujo.** **SELL LIMIT** en rebote al alza."
            elif in_bot and rising:
                sig,lbl,col,bull,pat = "DIVERGENCIA ALCISTA","⚡ POSIBLE GIRO",YELLOW,None,"divergencia"
                expl  = f"Z negativo pero precio subiendo desde mínimos. Posible giro."
                entry = f"**Divergencia bajista-precio.** Espera confirmación antes de operar."
            else:
                sig,lbl,col,bull,pat = "SESGO BAJISTA","↓ SESGO CORTO","#ff6b6b",False,"momentum_moderado"
                expl  = f"Flujo negativo ({z:.2f}), precio en {ppct*100:.0f}% del rango."
                entry = f"**Momentum bajista moderado.** **SELL STOP** en ruptura bajista."
    elif az > 0.5:
        if z > 0:
            sig,lbl,col,bull,pat = "SESGO ALCISTA","↑ SESGO LARGO MODERADO","#69f0ae",True,"moderado"
            expl  = f"Flujo positivo moderado ({z:.2f}). Precio en {ppct*100:.0f}% del rango."
            entry = f"Flujo positivo moderado. Espera Z > 1.5 o extremo de rango para señal de alta convicción."
        else:
            sig,lbl,col,bull,pat = "SESGO BAJISTA","↓ SESGO CORTO MODERADO","#ff6b6b",False,"moderado"
            expl  = f"Flujo negativo moderado ({z:.2f}). Precio en {ppct*100:.0f}% del rango."
            entry = f"Flujo negativo moderado. Espera Z < -1.5 o extremo de rango."
    else:
        sig,lbl,col,bull,pat = "NEUTRAL","➡ NEUTRAL",YELLOW,None,"neutro"
        expl  = f"Z-Diff neutral ({z:.2f}). No hay mano fuerte. Precio en {ppct*100:.0f}% del rango."
        entry = f"Sin señal operativa. No operar — espera que Z supere ±1.5."

    return {
        "signal":sig,"label":lbl,"color":col,"expl":expl,
        "entry_reason":entry,"pattern":pat,
        "bull":bull,"rising":rising,"abs_z":az,"extreme":az>2.2,
        "price_pct":ppct,"in_top":in_top,"in_bottom":in_bot,
        "breaking_up":bu,"breaking_down":bd,
    }


def build_macro_prompt(ticker, asset_type, horizon):
    today = datetime.now().strftime("%A, %d de %B de %Y")
    return f"""Hoy es {today}. Analiza el contexto macroeconómico para {ticker} ({asset_type}) los próximos {horizon} días.

Responde SOLO con este JSON exacto, sin backticks ni texto extra:
{{"macro":0,"macro_label":"Neutral","macro_why":"1 frase","news":0,"news_label":"Neutros","news_why":"1 frase con eventos","vol":"normal","vol_label":"Normal","vol_why":"1 frase","risk_events":["evento 1","evento 2","evento 3"],"correlations":{{"USD_INDEX":"neutral","RISK_APPETITE":"neutral","BONDS":"neutral"}},"summary":"2-3 frases sobre sesgo swing {horizon}d de {ticker}"}}

macro y news = entero -2 a 2 · vol = low/normal/high"""


# ═══════════════════════════════════════════════════════════════════════════════
#  VOLUME ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def calc_volume_profile(df: pd.DataFrame, bins: int = 40) -> dict:
    """
    Perfil de volumen: distribuye el volumen por niveles de precio.
    Usa volumen real o proxy (rango × tick) para forex.
    Calcula POC, VAH, VAL (70% del volumen).
    """
    c = df["Close"].values.astype(float)
    h = df["High"].values.astype(float)
    l = df["Low"].values.astype(float)
    v = df["Volume"].fillna(0).values.astype(float)

    # Proxy para forex (volumen = 0)
    vol_ok = v.sum() > 0 and pd.Series(v).nunique() > 3
    if not vol_ok:
        rng = h - l
        tp  = (h + l + c) / 3
        v   = (rng / rng.mean() * tp.mean() * 1000)

    price_min = l.min()
    price_max = h.max()
    edges     = np.linspace(price_min, price_max, bins + 1)
    centers   = (edges[:-1] + edges[1:]) / 2
    vol_bins  = np.zeros(bins)

    # Distribuir volumen de cada vela en los bins que toca
    for i in range(len(df)):
        c_lo, c_hi, c_v = l[i], h[i], v[i]
        mask = (centers >= c_lo) & (centers <= c_hi)
        n    = mask.sum()
        if n > 0:
            vol_bins[mask] += c_v / n

    # POC — precio con mayor volumen
    poc_idx = int(np.argmax(vol_bins))
    poc     = float(centers[poc_idx])

    # VAH / VAL — rango del 70% del volumen alrededor del POC
    total_vol   = vol_bins.sum()
    target_vol  = total_vol * 0.70
    sorted_idx  = np.argsort(vol_bins)[::-1]
    cum_vol     = 0.0
    va_indices  = []
    for idx in sorted_idx:
        if cum_vol >= target_vol:
            break
        cum_vol += vol_bins[idx]
        va_indices.append(idx)
    vah = float(centers[max(va_indices)])
    val = float(centers[min(va_indices)])

    return {
        "centers": centers, "vol_bins": vol_bins,
        "poc": poc, "vah": vah, "val": val,
        "total_vol": total_vol, "vol_ok": vol_ok,
        "price_min": price_min, "price_max": price_max,
    }


def calc_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP rolling de la sesión (desde inicio del dataframe)."""
    tp  = (df["High"] + df["Low"] + df["Close"]) / 3
    v   = df["Volume"].fillna(0)
    vol_ok = v.sum() > 0 and v.nunique() > 3
    if not vol_ok:
        rng = df["High"] - df["Low"]
        v   = (rng / rng.mean() * tp.mean() * 1000).fillna(1.0)
    cum_tpv = (tp * v).cumsum()
    cum_v   = v.cumsum()
    return cum_tpv / cum_v.replace(0, np.nan)


def calc_volume_delta(df: pd.DataFrame) -> pd.Series:
    """
    Volume Delta aproximado: volumen alcista - bajista por vela.
    Sin datos de tick usamos la posición del cierre en el rango como proxy.
    """
    v   = df["Volume"].fillna(0).values.astype(float)
    c   = df["Close"].values.astype(float)
    h   = df["High"].values.astype(float)
    l   = df["Low"].values.astype(float)
    vol_ok = v.sum() > 0 and pd.Series(v).nunique() > 3

    if vol_ok:
        # Proxy: % del rango que es comprador
        rng = h - l
        rng[rng == 0] = 1e-10
        buy_pct  = (c - l) / rng          # 0=todo vendedor, 1=todo comprador
        sell_pct = 1 - buy_pct
        delta    = (buy_pct - sell_pct) * v
    else:
        # Forex: usa variación de precio como proxy
        delta = pd.Series(c).diff().fillna(0).values
        delta = delta * abs(delta) * 1000  # amplificar señal

    return pd.Series(delta, index=df.index)


def calc_volume_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detecta anomalías de volumen:
    - Volumen > 2σ sobre la media = spike
    - Volumen alto + vela pequeña = absorción (institucional)
    - Volumen bajo + vela grande = ruptura sin volumen (sospechosa)
    """
    v     = df["Volume"].fillna(0).values.astype(float)
    c     = df["Close"].values.astype(float)
    h     = df["High"].values.astype(float)
    l     = df["Low"].values.astype(float)
    vol_ok = v.sum() > 0 and pd.Series(v).nunique() > 3

    if not vol_ok:
        rng = h - l
        tp  = (h + l + c) / 3
        v   = (rng / rng.mean() * tp.mean() * 1000)

    v_mean = pd.Series(v).rolling(20, min_periods=5).mean().values.copy()
    v_std  = pd.Series(v).rolling(20, min_periods=5).std().values.copy()
    # Replace 0 and NaN in std to avoid division errors
    v_std  = np.where((v_std == 0) | np.isnan(v_std), 1.0, v_std)
    v_mean = np.where(np.isnan(v_mean), np.nanmean(v) if len(v) > 0 else 1.0, v_mean)

    body   = np.abs(c - np.roll(c, 1))
    rng_v  = h - l
    rng_v  = np.where(rng_v == 0, 1e-10, rng_v)
    body_pct = body / rng_v  # % del rango que es cuerpo

    anomaly_type  = []
    anomaly_score = []

    for i in range(len(df)):
        z = (v[i] - v_mean[i]) / v_std[i]
        bp = body_pct[i]

        if z > 2.5 and bp < 0.3:
            at = "ABSORCIÓN"    # Volumen muy alto, vela pequeña → institucional absorbiendo
            sc = min(abs(z), 5)
        elif z > 2.5:
            at = "SPIKE VOLUMEN" # Volumen muy alto con movimiento fuerte
            sc = min(abs(z), 5)
        elif z > 1.8 and bp > 0.7:
            at = "MOMENTUM"     # Volumen alto + cuerpo grande → impulso real
            sc = min(abs(z), 4)
        elif z < -1.5 and bp > 0.6:
            at = "RUPTURA SECA" # Volumen bajo + cuerpo grande → ruptura sospechosa
            sc = min(abs(z), 3)
        else:
            at = "NORMAL"
            sc = 0

        anomaly_type.append(at)
        anomaly_score.append(sc)

    df_out = df.copy()
    df_out["vol_z"]     = (v - v_mean) / v_std
    df_out["anomaly"]   = anomaly_type
    df_out["anom_score"]= anomaly_score
    df_out["volume_eff"]= v
    return df_out


# ═══════════════════════════════════════════════════════════════════════════════
#  SUMMARY ENGINE — Puntuación agregada multi-modelo
# ═══════════════════════════════════════════════════════════════════════════════

def build_summary(r: dict, ctx: dict) -> dict:
    """
    Agrega todas las señales en una puntuación direccional unificada.
    Retorna score -100 a +100, señales individuales, zonas clave de volumen,
    y un veredicto operativo con nivel de convicción.
    """
    price    = r["price"]
    zctx     = r["zdiff_ctx"]
    vdata    = r["vol_data"]
    mk       = r["markov"]
    vp       = r["vol_profile"]
    vwap     = r["vwap_series"]
    delt     = r["delta_series"]
    adj_bull = r["adj_bull"]
    macro    = ctx.get("macro", 0) if ctx else 0
    news     = ctx.get("news",  0) if ctx else 0

    signals = []

    # ── 1. DIRECCIONALIDAD — Monte Carlo (peso 25) ────────────────────────────
    mc_score = (adj_bull - 50) * 0.5          # -25 a +25
    mc_label = ("Alcista" if adj_bull >= 60
                else "Bajista" if adj_bull <= 40 else "Neutral")
    mc_color = GREEN if adj_bull >= 60 else RED if adj_bull <= 40 else YELLOW
    signals.append({
        "categoria": "Monte Carlo",
        "icono": "🎲",
        "valor": f"P={adj_bull:.1f}%",
        "label": mc_label,
        "color": mc_color,
        "score": mc_score,
        "peso":  25,
        "detalle": f"Monte Carlo GBM proyecta {adj_bull:.1f}% de probabilidad alcista en el horizonte."
    })

    # ── 2. ORDER FLOW Z-DIFF (peso 25) ───────────────────────────────────────
    z = r["last_z"]
    z_bull = zctx.get("bull")
    if z_bull is True:     z_score = min(abs(z), 2.5) / 2.5 * 25
    elif z_bull is False:  z_score = -min(abs(z), 2.5) / 2.5 * 25
    else:                   z_score = 0
    z_label = zctx.get("signal", "Neutral")
    z_color = zctx.get("color", YELLOW)
    signals.append({
        "categoria": "Z-Diff Order Flow",
        "icono": "⚡",
        "valor": f"{z:.3f}",
        "label": z_label,
        "color": z_color,
        "score": z_score,
        "peso":  25,
        "detalle": zctx.get("expl", "—")
    })

    # ── 3. MARKOV (peso 15) ───────────────────────────────────────────────────
    nd  = mk["next_day"]
    mk_bull = float(nd[2])    # P(ALCISTA)
    mk_bear = float(nd[0])    # P(BAJISTA)
    mk_score = (mk_bull - mk_bear) * 15
    mk_dom   = mk["labels"][int(np.argmax(nd))]
    mk_color = GREEN if mk_bull > mk_bear else RED if mk_bear > mk_bull else YELLOW
    signals.append({
        "categoria": "Cadena de Markov",
        "icono": "🔗",
        "valor": f"↑{mk_bull*100:.0f}% ↓{mk_bear*100:.0f}%",
        "label": f"→ {mk_dom} mañana",
        "color": mk_color,
        "score": mk_score,
        "peso":  15,
        "detalle": f"Probabilidad de transición: {mk_dom} con {max(nd)*100:.0f}% de probabilidad."
    })

    # ── 4. VOLATILIDAD — Régimen (peso 10) ───────────────────────────────────
    reg = vdata["vol_regime"]
    if reg == "COMPRESIÓN":
        # Compresión: señal de movimiento próximo, dirección incierta
        vol_score = 0   # no añade dirección
        vol_label = "Compresión — Ruptura Próxima"
        vol_color = CYAN
        vol_det   = f"Volatilidad corta ({vdata['rv_short']*100:.1f}%) muy inferior a la larga ({vdata['rv_long']*100:.1f}%). Movimiento inminente — tamaño reducido."
    elif reg == "EXPANSIÓN":
        # En expansión el momentum ya tiene dirección capturada por Z-Diff
        vol_score = 0
        vol_label = "Expansión — Aumenta SL/TP"
        vol_color = ORANGE
        vol_det   = f"Volatilidad en expansión. Aumenta stops y reduce tamaño."
    else:
        vol_score = 0
        vol_label = "Normal — Parámetros estándar"
        vol_color = YELLOW
        vol_det   = "Régimen estable. Usa ATR como referencia directa."
    signals.append({
        "categoria": "Régimen Volatilidad",
        "icono": "📊",
        "valor": f"{vdata['rv_current']*100:.1f}%",
        "label": vol_label,
        "color": vol_color,
        "score": vol_score,
        "peso":  10,
        "detalle": vol_det
    })

    # ── 5. VOLUMEN — Posición respecto a zonas clave (peso 15) ───────────────
    poc, vah, val_v = vp["poc"], vp["vah"], vp["val"]
    vwap_now = float(vwap.iloc[-1])
    cum_delta = float(delt.sum())
    recent_delta = float(delt.iloc[-6:].sum())

    # Posición precio vs zonas
    in_va     = val_v <= price <= vah
    above_va  = price > vah
    below_va  = price < val_v
    above_poc = price > poc
    above_vwap= price > vwap_now
    delta_bull= cum_delta > 0 and recent_delta > 0

    if above_va and above_vwap and delta_bull:
        vol_dir_score = 15; vol_dir_lbl = "Alcista — precio sobre VA+VWAP"
        vol_dir_col   = GREEN
        vol_dir_det   = f"Precio {price:.4f} sobre VAH {vah:.4f} y VWAP {vwap_now:.4f} con delta comprador. Estructura de volumen alcista."
    elif above_va and not delta_bull:
        vol_dir_score = 5;  vol_dir_lbl = "Alcista débil — VA pero delta mixto"
        vol_dir_col   = "#69f0ae"
        vol_dir_det   = f"Precio sobre VAH pero delta vendedor reciente. Posible distribución en techo."
    elif below_va and not above_vwap and not delta_bull:
        vol_dir_score = -15; vol_dir_lbl = "Bajista — precio bajo VA+VWAP"
        vol_dir_col   = RED
        vol_dir_det   = f"Precio {price:.4f} bajo VAL {val_v:.4f} y VWAP {vwap_now:.4f} con delta vendedor. Estructura bajista."
    elif below_va and delta_bull:
        vol_dir_score = -5; vol_dir_lbl = "Bajista débil — bajo VA pero delta mixto"
        vol_dir_col   = "#ff6b6b"
        vol_dir_det   = f"Precio bajo VAL pero delta comprador reciente. Posible acumulación en suelo."
    elif in_va and above_poc:
        vol_dir_score = 8;  vol_dir_lbl = "Neutro-alcista — en VA sobre POC"
        vol_dir_col   = "#69f0ae"
        vol_dir_det   = f"Precio en Value Area por encima del POC ({poc:.4f}). Zona de equilibrio con ligero sesgo alcista."
    elif in_va and not above_poc:
        vol_dir_score = -8; vol_dir_lbl = "Neutro-bajista — en VA bajo POC"
        vol_dir_col   = "#ff6b6b"
        vol_dir_det   = f"Precio en Value Area por debajo del POC ({poc:.4f}). Zona de equilibrio con ligero sesgo bajista."
    else:
        vol_dir_score = 0; vol_dir_lbl = "Neutral"
        vol_dir_col   = YELLOW
        vol_dir_det   = "Sin señal de volumen clara."

    signals.append({
        "categoria": "Volumen & Zonas Clave",
        "icono": "📦",
        "valor": f"POC {poc:.4f}",
        "label": vol_dir_lbl,
        "color": vol_dir_col,
        "score": vol_dir_score,
        "peso":  15,
        "detalle": vol_dir_det
    })

    # ── 6. MACRO (peso 10) ────────────────────────────────────────────────────
    macro_score = (macro + news) / 4 * 10   # -10 a +10
    macro_label = ctx.get("macro_label", "Sin contexto") if ctx else "Sin contexto macro"
    macro_color = GREEN if macro_score > 2 else RED if macro_score < -2 else YELLOW
    signals.append({
        "categoria": "Macro + Noticias",
        "icono": "🌐",
        "valor": f"M:{macro:+d} N:{news:+d}",
        "label": macro_label,
        "color": macro_color,
        "score": macro_score,
        "peso":  10,
        "detalle": ctx.get("summary", "Sin contexto macro. Añade contexto en Tab ④.") if ctx else "Sin contexto macro. Añade contexto en Tab ④."
    })

    # ── SCORE TOTAL ───────────────────────────────────────────────────────────
    total_score = sum(s["score"] for s in signals)
    total_score = float(np.clip(total_score, -100, 100))

    # Alineación de señales (cuántas apuntan en la misma dirección)
    bullish_sigs = sum(1 for s in signals if s["score"] > 2)
    bearish_sigs = sum(1 for s in signals if s["score"] < -2)
    neutral_sigs = len(signals) - bullish_sigs - bearish_sigs
    alignment    = max(bullish_sigs, bearish_sigs) / len(signals)

    # Convicción
    if abs(total_score) >= 55 and alignment >= 0.75:
        conviction = "ALTA"; conviction_color = GREEN if total_score > 0 else RED
    elif abs(total_score) >= 35 and alignment >= 0.5:
        conviction = "MEDIA"; conviction_color = ORANGE
    else:
        conviction = "BAJA"; conviction_color = MUTED

    # Veredicto operativo
    if total_score >= 40 and conviction in ("ALTA","MEDIA"):
        verdict       = "OPERAR — LARGO"
        verdict_color = GREEN
        verdict_icon  = "▲"
    elif total_score <= -40 and conviction in ("ALTA","MEDIA"):
        verdict       = "OPERAR — CORTO"
        verdict_color = RED
        verdict_icon  = "▼"
    elif abs(total_score) >= 25:
        verdict       = "MONITORIZAR"
        verdict_color = ORANGE
        verdict_icon  = "◉"
    else:
        verdict       = "NO OPERAR"
        verdict_color = MUTED
        verdict_icon  = "—"

    # Zonas clave de volumen
    key_zones = []
    key_zones.append({"nivel": poc,     "tipo": "POC",  "color": ORANGE,
                       "desc": "Mayor volumen negociado — imán de precio"})
    key_zones.append({"nivel": vah,     "tipo": "VAH",  "color": BLUE,
                       "desc": "Techo del 70% del volumen — resistencia clave"})
    key_zones.append({"nivel": val_v,   "tipo": "VAL",  "color": BLUE,
                       "desc": "Suelo del 70% del volumen — soporte clave"})
    key_zones.append({"nivel": vwap_now,"tipo": "VWAP", "color": YELLOW,
                       "desc": "Precio medio ponderado por volumen — soporte/resistencia dinámica"})
    key_zones.sort(key=lambda x: x["nivel"], reverse=True)

    return {
        "signals":       signals,
        "total_score":   total_score,
        "verdict":       verdict,
        "verdict_color": verdict_color,
        "verdict_icon":  verdict_icon,
        "conviction":    conviction,
        "conviction_color": conviction_color,
        "bullish_sigs":  bullish_sigs,
        "bearish_sigs":  bearish_sigs,
        "neutral_sigs":  neutral_sigs,
        "alignment":     alignment,
        "key_zones":     key_zones,
        "vol_regime":    reg,
        "poc": poc, "vah": vah, "val": val_v, "vwap": vwap_now,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def fig_price_with_mc(df, mc_paths):
    price    = float(df["Close"].iloc[-1])
    last_ts  = df.index[-1]
    h4d      = timedelta(hours=4)
    steps    = mc_paths.shape[1]
    fts      = [last_ts + h4d*(i+1) for i in range(steps)]

    p5  = np.percentile(mc_paths,  5, axis=0)
    p25 = np.percentile(mc_paths, 25, axis=0)
    p50 = np.percentile(mc_paths, 50, axis=0)
    p75 = np.percentile(mc_paths, 75, axis=0)
    p95 = np.percentile(mc_paths, 95, axis=0)
    final    = mc_paths[:, -1]
    bull_pct = float(np.mean(final > price) * 100)

    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.78, 0.22],
        row_heights=[0.70, 0.30],
        shared_xaxes=False, shared_yaxes=False,
        horizontal_spacing=0.01, vertical_spacing=0.06,
        specs=[[{"type":"xy"}, {"type":"bar","rowspan":2}],[{"type":"bar"}, None]],
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_fillcolor=GREEN, increasing_line_color=GREEN,
        decreasing_fillcolor=RED,   decreasing_line_color=RED,
        name="H4", showlegend=False
    ), row=1, col=1)

    # MC bands
    fig.add_trace(go.Scatter(
        x=fts+fts[::-1], y=list(p95)+list(p5[::-1]),
        fill="toself", fillcolor="rgba(0,144,255,0.07)",
        line=dict(color="rgba(0,0,0,0)"), name="IC 90%"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=fts+fts[::-1], y=list(p75)+list(p25[::-1]),
        fill="toself", fillcolor="rgba(0,144,255,0.17)",
        line=dict(color="rgba(0,0,0,0)"), name="IC 50%"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=fts, y=p50, line=dict(color=CYAN, width=2, dash="dash"),
        name="Mediana MC"
    ), row=1, col=1)

    # Sample paths
    rng = np.random.default_rng(7)
    for i in rng.choice(len(mc_paths), size=min(50,len(mc_paths)), replace=False):
        c = "rgba(0,230,118,0.04)" if mc_paths[i,-1]>price else "rgba(255,23,68,0.04)"
        fig.add_trace(go.Scatter(
            x=fts, y=mc_paths[i],
            line=dict(color=c, width=1), showlegend=False, hoverinfo="skip"
        ), row=1, col=1)

    # Current price line
    fig.add_hline(y=price, line_color="rgba(255,255,255,0.35)",
                   line_dash="dot", line_width=1, row=1, col=1)

    # Z-Diff bars
    zc = df["z_diff"].apply(
        lambda z: GREEN if z>1.5 else "#69f0ae" if z>0.5
        else YELLOW if z>-0.5 else "#ff6b6b" if z>-1.5 else RED
    )
    fig.add_trace(go.Bar(x=df.index, y=df["z_diff"],
                          marker_color=zc, showlegend=False), row=2, col=1)
    for yv, clr in [(1.5,"rgba(0,230,118,.3)"),(-1.5,"rgba(255,23,68,.3)"),(0,"rgba(255,255,255,.1)")]:
        fig.add_hline(y=yv, line_color=clr, line_dash="dash", row=2, col=1)

    # Histogram (horizontal, col 2)
    bins   = 55
    mn, mx = final.min(), final.max()
    edges  = np.linspace(mn, mx, bins+1)
    counts,_ = np.histogram(final, bins=edges)
    centers  = (edges[:-1]+edges[1:])/2
    hcols    = [GREEN if c>price else RED for c in centers]

    fig.add_trace(go.Bar(
        x=counts, y=centers, orientation="h",
        marker_color=hcols, marker_line_width=0,
        opacity=0.85, showlegend=False
    ), row=1, col=2)
    fig.add_hline(y=price, line_color="rgba(255,255,255,0.7)",
                   line_dash="dot", line_width=1.5, row=1, col=2)

    # Annotation
    fig.add_annotation(
        xref="x3 domain", yref="y3 domain",
        x=0.5, y=1.0, yanchor="top",
        text=f"▲ {bull_pct:.1f}%",
        showarrow=False,
        font=dict(size=18, color=GREEN if bull_pct>=50 else RED,
                  family="Rajdhani"),
    )

    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=S1,
        height=600, margin=dict(l=8,r=8,t=8,b=8),
        legend=dict(orientation="h", y=1.02, x=0,
                    font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(gridcolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER)
    fig.update_yaxes(title_text="Precio H4", row=1, col=1)
    fig.update_yaxes(title_text="Z-Diff",    row=2, col=1)
    fig.update_xaxes(title_text="Simulaciones", row=1, col=2)
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    return fig, bull_pct


def fig_volatility(df, vol_data, mc_paths):
    price    = float(df["Close"].iloc[-1])
    last_ts  = df.index[-1]
    h4d      = timedelta(hours=4)
    steps    = mc_paths.shape[1]
    fts      = [last_ts + h4d*(i+1) for i in range(steps)]
    sig_h4   = vol_data["sigma_1d"] / np.sqrt(H4_PER_DAY)
    ts       = np.arange(1, steps+1)

    c1u = [price*np.exp( sig_h4*np.sqrt(t)) for t in ts]
    c1d = [price*np.exp(-sig_h4*np.sqrt(t)) for t in ts]
    c2u = [price*np.exp( 2*sig_h4*np.sqrt(t)) for t in ts]
    c2d = [price*np.exp(-2*sig_h4*np.sqrt(t)) for t in ts]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"Precio H4 + Cono de Volatilidad ({steps} pasos)",
                        "Volatilidad Realizada Rolling (anualizada)",
                        "ATR Multi-Periodo — % sobre precio",
                        "Distribución de Retornos H4 vs Normal"],
        vertical_spacing=0.14, horizontal_spacing=0.08
    )

    # Price + cones
    fig.add_trace(go.Scatter(x=df.index[-40:], y=df["Close"].values[-40:],
                              line=dict(color=TEXT, width=1.5),
                              showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=fts+fts[::-1], y=c2u+c2d[::-1],
        fill="toself", fillcolor="rgba(0,144,255,0.07)",
        line=dict(color="rgba(0,0,0,0)"), name="2σ (95%)"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=fts+fts[::-1], y=c1u+c1d[::-1],
        fill="toself", fillcolor="rgba(0,144,255,0.17)",
        line=dict(color="rgba(0,0,0,0)"), name="1σ (68%)"
    ), row=1, col=1)
    for arr, clr in [(c1u,"rgba(0,144,255,.6)"),(c1d,"rgba(0,144,255,.6)"),
                     (c2u,"rgba(0,144,255,.3)"),(c2d,"rgba(0,144,255,.3)")]:
        fig.add_trace(go.Scatter(x=fts, y=arr,
                                  line=dict(color=clr, width=1, dash="dot"),
                                  showlegend=False, hoverinfo="skip"), row=1, col=1)

    # Realized vol rolling
    rv  = vol_data["rv_series"].dropna() * 100
    fig.add_trace(go.Scatter(
        x=df.index[-len(rv):], y=rv,
        line=dict(color=ORANGE, width=1.5),
        fill="toself", fillcolor="rgba(255,145,0,0.1)",
        showlegend=False
    ), row=1, col=2)
    fig.add_hline(y=float(rv.mean()),
                   line_dash="dash", line_color="rgba(255,145,0,0.4)", row=1, col=2)

    # ATR %
    aw  = list(vol_data["atr"].keys())
    av  = list(vol_data["atr"].values())
    ap  = [v/price*100 for v in av]
    dec = 1 if price>1000 else 2 if price>100 else 5
    fig.add_trace(go.Bar(
        x=[f"{w}v" for w in aw], y=ap,
        marker_color=[CYAN, BLUE, ORANGE, PURPLE][:len(aw)],
        text=[f"{v:.{dec}f}" for v in av],
        textposition="outside", showlegend=False
    ), row=2, col=1)

    # Return distribution
    rets = np.diff(np.log(df["Close"].values.astype(float))) * 100
    fig.add_trace(go.Histogram(
        x=rets, nbinsx=40,
        marker_color="rgba(0,144,255,0.6)",
        showlegend=False
    ), row=2, col=2)
    xf = np.linspace(rets.min(), rets.max(), 100)
    yf = stats.norm.pdf(xf, rets.mean(), rets.std()) * len(rets) * (xf[1]-xf[0])
    fig.add_trace(go.Scatter(x=xf, y=yf,
                              line=dict(color=ORANGE, width=2),
                              showlegend=False), row=2, col=2)

    # Kurtosis annotation
    kurt = float(stats.kurtosis(rets))
    skew = float(stats.skew(rets))
    fig.add_annotation(
        xref="x4 domain", yref="y4 domain", x=0.98, y=0.95,
        text=f"Kurt: {kurt:.2f}<br>Skew: {skew:.2f}",
        showarrow=False, align="right",
        font=dict(size=10, color=MUTED),
        xanchor="right", yanchor="top"
    )

    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=S1,
        height=620, margin=dict(l=8,r=8,t=40,b=8), showlegend=True
    )
    fig.update_xaxes(gridcolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER)
    fig.update_yaxes(title_text="Precio",      row=1, col=1)
    fig.update_yaxes(title_text="Vol Anual %", row=1, col=2)
    fig.update_yaxes(title_text="ATR %",       row=2, col=1)
    fig.update_yaxes(title_text="Frecuencia",  row=2, col=2)
    return fig


def fig_markov(markov):
    T      = markov["transition"]
    labels = markov["labels"]
    fig    = make_subplots(rows=1, cols=2,
                            subplot_titles=["Matriz de Transición",
                                            "P(Estado) — Próximo Día"],
                            horizontal_spacing=0.14)
    fig.add_trace(go.Heatmap(
        z=T*100, x=labels, y=labels,
        colorscale=[[0,BG],[0.5,"rgba(0,144,255,.4)"],[1,CYAN]],
        text=[[f"{v:.0f}%" for v in row] for row in T*100],
        texttemplate="%{text}", showscale=False
    ), row=1, col=1)
    nd = markov["next_day"]
    fig.add_trace(go.Bar(
        x=labels, y=nd*100,
        marker_color=[RED, YELLOW, GREEN],
        text=[f"{v*100:.1f}%" for v in nd],
        textposition="outside", showlegend=False
    ), row=1, col=2)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=S1,
        height=320, margin=dict(l=8,r=8,t=40,b=8)
    )
    fig.update_yaxes(title_text="Prob %", row=1, col=2)
    return fig


def fig_volume_profile(df, vp, vwap, delta, df_anom):
    """Gráfico de perfil de volumen con VWAP, delta y anomalías."""
    price = float(df["Close"].iloc[-1])
    dec   = 1 if price>1000 else 2 if price>100 else 5

    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.72, 0.28],
        row_heights=[0.62, 0.38],
        shared_xaxes=False,
        horizontal_spacing=0.02, vertical_spacing=0.08,
        specs=[[{"type":"xy"}, {"type":"bar","rowspan":2}],
               [{"type":"bar"}, None]],
        subplot_titles=["", "Perfil de Volumen", "Volume Delta", ""]
    )

    # ── Candlestick + VWAP ────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_fillcolor=GREEN, increasing_line_color=GREEN,
        decreasing_fillcolor=RED,   decreasing_line_color=RED,
        name="H4", showlegend=False
    ), row=1, col=1)

    # VWAP
    fig.add_trace(go.Scatter(
        x=df.index, y=vwap,
        line=dict(color=YELLOW, width=1.5, dash="dash"),
        name="VWAP"
    ), row=1, col=1)

    # POC line
    fig.add_hline(y=vp["poc"], line_color=ORANGE, line_width=2,
                   line_dash="solid", row=1, col=1,
                   annotation_text=f"POC {vp['poc']:.{dec}f}",
                   annotation_font_color=ORANGE, annotation_position="right")

    # VAH / VAL
    fig.add_hrect(y0=vp["val"], y1=vp["vah"],
                   fillcolor="rgba(0,144,255,0.07)",
                   line_width=0, row=1, col=1)
    fig.add_hline(y=vp["vah"], line_color="rgba(0,144,255,.5)",
                   line_width=1, line_dash="dot", row=1, col=1,
                   annotation_text=f"VAH {vp['vah']:.{dec}f}",
                   annotation_font_color=BLUE, annotation_position="right")
    fig.add_hline(y=vp["val"], line_color="rgba(0,144,255,.5)",
                   line_width=1, line_dash="dot", row=1, col=1,
                   annotation_text=f"VAL {vp['val']:.{dec}f}",
                   annotation_font_color=BLUE, annotation_position="right")

    # Anomaly markers
    for _, row_d in df_anom[df_anom["anomaly"] != "NORMAL"].iterrows():
        acolor = {
            "ABSORCIÓN":    PURPLE,
            "SPIKE VOLUMEN": ORANGE,
            "MOMENTUM":     CYAN,
            "RUPTURA SECA": YELLOW,
        }.get(row_d["anomaly"], TEXT)
        fig.add_trace(go.Scatter(
            x=[row_d.name], y=[float(row_d["High"]) * 1.001],
            mode="markers+text",
            marker=dict(symbol="triangle-down", size=10, color=acolor),
            text=[row_d["anomaly"][:3]], textposition="top center",
            textfont=dict(size=8, color=acolor),
            showlegend=False, hovertext=row_d["anomaly"]
        ), row=1, col=1)

    # ── Volume Profile (horizontal bars) ──────────────────────────────────────
    poc_mask = vp["centers"] == vp["poc"]
    va_mask  = (vp["centers"] >= vp["val"]) & (vp["centers"] <= vp["vah"])
    bar_colors = []
    for i, c in enumerate(vp["centers"]):
        if abs(c - vp["poc"]) < (vp["price_max"]-vp["price_min"])/len(vp["centers"]):
            bar_colors.append(ORANGE)   # POC
        elif va_mask[i]:
            bar_colors.append(BLUE)     # Value Area
        else:
            bar_colors.append(MUTED)    # fuera VA

    fig.add_trace(go.Bar(
        x=vp["vol_bins"], y=vp["centers"],
        orientation="h",
        marker_color=bar_colors, marker_line_width=0,
        opacity=0.85, showlegend=False,
        hovertemplate="Precio: %{y:.5f}<br>Vol: %{x:,.0f}<extra></extra>"
    ), row=1, col=2)

    # Price line on profile
    fig.add_hline(y=price, line_color="rgba(255,255,255,0.6)",
                   line_width=1.5, line_dash="dot", row=1, col=2)

    # ── Volume Delta bars ─────────────────────────────────────────────────────
    delta_colors = [GREEN if d >= 0 else RED for d in delta.values]
    fig.add_trace(go.Bar(
        x=df.index, y=delta.values,
        marker_color=delta_colors, marker_line_width=0,
        opacity=0.8, showlegend=False, name="Vol Delta"
    ), row=2, col=1)
    fig.add_hline(y=0, line_color="rgba(255,255,255,.2)",
                   line_dash="dot", row=2, col=1)

    # Cumulative delta line
    cum_delta = delta.cumsum()
    fig.add_trace(go.Scatter(
        x=df.index, y=cum_delta.values,
        line=dict(color=CYAN, width=1.5),
        name="Delta Acumulado",
        yaxis="y5"
    ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=S1,
        height=640, margin=dict(l=8,r=8,t=30,b=8),
        legend=dict(orientation="h", y=1.02, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        xaxis_rangeslider_visible=False,
        barmode="overlay",
        yaxis5=dict(overlaying="y3", side="right", showgrid=False,
                    showticklabels=False, title="Δ Acum."),
    )
    fig.update_xaxes(gridcolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER)
    fig.update_yaxes(title_text="Precio H4", row=1, col=1)
    fig.update_xaxes(title_text="Volumen",   row=1, col=2)
    fig.update_yaxes(showticklabels=False,   row=1, col=2)
    fig.update_yaxes(title_text="Delta",     row=2, col=1)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"<div style='font-family:Rajdhani,sans-serif;font-size:22px;"
                f"font-weight:700;color:{CYAN};letter-spacing:3px'>"
                f"ORDER<span style='color:{YELLOW}'>FLOW</span> PRO</div>",
                unsafe_allow_html=True)
    st.caption("Dashboard Cuantitativo · H4 · Sin API Key")
    st.divider()

    st.markdown("### Activo")
    quick = st.selectbox("Acceso rápido", list(QUICK_MAP.keys()))
    default_sym, default_type = QUICK_MAP[quick]
    ticker     = st.text_input("Símbolo Yahoo Finance", value=default_sym,
                                placeholder="ES=F, NQ=F, GC=F, EURUSD=X...")

    # Show futures note
    if ticker in FUTURES_NOTE:
        st.caption(f"🔄 **{FUTURES_NOTE[ticker]}** — cotiza ~23h/día")
    elif ticker in ("^GSPC","^GDAXI","^IXIC"):
        st.warning("⚠️ Índice spot — solo datos en horario de bolsa. Usa `ES=F`, `NQ=F` o `FDAX=F` para datos continuos 23h.", icon="⏰")

    asset_type = st.selectbox("Tipo",
                               ["forex","index","commodity","stock","crypto"],
                               index=["forex","index","commodity","stock","crypto"].index(default_type))

    st.divider()
    st.markdown("### Modelo")
    horizon   = st.selectbox("Horizonte", [1, 3, 5],
                               format_func=lambda x: f"{x} día{'s' if x>1 else ''}")
    n_candles = st.slider("Velas H4", 30, 90, 60)
    z_period  = st.slider("Periodo Z-Diff", 10, 30, 14)
    mc_sims   = st.selectbox("Simulaciones MC", [1000, 3000, 5000], index=1)
    threshold = st.selectbox("Umbral mínimo %", [60, 65, 70], index=1)

    st.divider()
    st.markdown("### Gestión de riesgo")
    account  = st.number_input("Capital ($)", value=10000, step=500)
    risk_pct = st.slider("Riesgo %", 0.5, 10.0, 2.0, 0.5)
    instr    = st.selectbox("Instrumento",
                             ["Forex std (100k)","Forex mini (10k)","XAU/USD","Índice CFD"])

    st.divider()
    load_btn = st.button("📡 CARGAR DATOS H4", use_container_width=True, type="primary")
    run_btn  = st.button("▶ EJECUTAR MODELO",  use_container_width=True,
                          disabled=st.session_state.df is None)
    st.divider()
    st.caption("**Símbolos**\n\n`EURUSD=X` `GBPUSD=X` `USDJPY=X`\n\n"
               "`GC=F` (XAU) · `^GSPC` (SP500)\n\n`^GDAXI` (DAX) · `BTC-USD`")

# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
if load_btn and ticker:
    with st.spinner(f"Descargando {n_candles} velas H4 — {ticker}..."):
        try:
            raw = yf.download(ticker, period=TF_PERIOD, interval=TF_INTERVAL,
                               auto_adjust=True, progress=False)
            if raw.empty:
                st.error(f"Sin datos para '{ticker}'.")
            else:
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                df = raw.tail(n_candles).copy()
                df.index = pd.to_datetime(df.index)
                df = calc_order_flow(df, z_period)
                st.session_state.df      = df
                st.session_state.results = None
                p  = float(df["Close"].iloc[-1])
                d  = 1 if p>1000 else 2 if p>100 else 5
                st.sidebar.success(f"✓ {len(df)} velas · {p:.{d}f}")
        except Exception as e:
            st.error(f"Error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
#  RUN MODEL
# ═══════════════════════════════════════════════════════════════════════════════
if run_btn and st.session_state.df is not None:
    df    = st.session_state.df
    price = float(df["Close"].iloc[-1])
    rets  = np.diff(np.log(df["Close"].values.astype(float)))
    ctx   = st.session_state.context or {}
    macro = ctx.get("macro", 0)
    news  = ctx.get("news",  0)
    vol_v = ctx.get("vol",   "normal")

    with st.spinner("Calculando modelos cuantitativos..."):
        last_z    = float(df["z_diff"].iloc[-1])
        z_adj     = float(np.clip(last_z, -2, 2))
        vm        = 0.7 if vol_v=="low" else 1.5 if vol_v=="high" else 1.0
        mc_steps  = horizon * H4_PER_DAY
        zdiff_ctx = interpret_zdiff(last_z, df, macro)
        mc_paths  = run_mc(price, rets, mc_sims, mc_steps, z_adj, vm)
        markov    = calc_markov(df)
        vol_data  = calc_volatility(df)

        final     = mc_paths[:, -1]
        ctx_boost = (macro + news) / 4 * 8
        adj_bull  = float(np.clip(np.mean(final>price)*100 + ctx_boost, 10, 90))
        adj_bear  = 100 - adj_bull

        # Volume models
        vol_profile = calc_volume_profile(df)
        vwap_series = calc_vwap(df)
        delta_series= calc_volume_delta(df)
        df_anom     = calc_volume_anomalies(df)

        st.session_state.results = dict(
            price=price, last_z=last_z, last_rmf=float(df["rmf"].iloc[-1]),
            adj_bull=adj_bull, adj_bear=adj_bear,
            mc_paths=mc_paths, final=final,
            zdiff_ctx=zdiff_ctx, markov=markov, vol_data=vol_data,
            macro=macro, news=news, rets=rets,
            p5  = float(np.percentile(final, 5)),
            p95 = float(np.percentile(final, 95)),
            p20 = float(np.percentile(final, 20)),
            p80 = float(np.percentile(final, 80)),
            p8  = float(np.percentile(final, 8)),
            p92 = float(np.percentile(final, 92)),
            # Volume
            vol_profile  = vol_profile,
            vwap_series  = vwap_series,
            delta_series = delta_series,
            df_anom      = df_anom,
        )
        st.session_state.macro_prompt = build_macro_prompt(ticker, asset_type, horizon)


# ═══════════════════════════════════════════════════════════════════════════════
#  HEADER + GUARD
# ═══════════════════════════════════════════════════════════════════════════════
r = st.session_state.results

st.markdown(f"""
<div style='display:flex;align-items:baseline;gap:16px;margin-bottom:4px'>
  <span style='font-family:Rajdhani,sans-serif;font-size:30px;font-weight:700;
    color:{CYAN};letter-spacing:4px'>ORDER<span style='color:{YELLOW}'>FLOW</span> PRO</span>
  <span style='font-size:11px;color:{MUTED};letter-spacing:2px'>H4 · DASHBOARD CUANTITATIVO · SIN API KEY</span>
</div>
""", unsafe_allow_html=True)

if r is None:
    st.info("👈 **Carga los datos H4** y pulsa **Ejecutar Modelo** para ver el análisis completo.")
    st.stop()

# Global KPI bar
price = r["price"]
dec   = 1 if price>1000 else 2 if price>100 else 4
bc    = GREEN if r["adj_bull"]>=60 else RED if r["adj_bull"]<=40 else YELLOW
bt    = ("SESGO ALCISTA ▲" if r["adj_bull"]>=60
         else "SESGO BAJISTA ▼" if r["adj_bull"]<=40 else "SESGO NEUTRAL ➡")
zctx  = r["zdiff_ctx"]
vdata = r["vol_data"]
mk    = r["markov"]

def kpi(col, lbl, val, sub="", color=TEXT):
    col.markdown(f"""<div class='kpi'>
        <div class='kpi-lbl'>{lbl}</div>
        <div class='kpi-val' style='color:{color}'>{val}</div>
        <div class='kpi-sub'>{sub}</div>
    </div>""", unsafe_allow_html=True)

h1,h2,h3,h4,h5,h6 = st.columns(6)
kpi(h1, "Veredicto",          bt,                                       f"MC P={r['adj_bull']:.1f}%", bc)
kpi(h2, "Z-Diff H4",          f"{r['last_z']:.3f}",                    zctx["label"][:22],            zctx["color"])
kpi(h3, "Precio actual",      f"{price:.{dec}f}",                       ticker,                        TEXT)
kpi(h4, f"Rango 90% MC",      f"{r['p5']:.{dec}f} – {r['p95']:.{dec}f}", f"{horizon}d horizonte",    BLUE)
kpi(h5, "Vol Realizada",      f"{vdata['rv_current']*100:.2f}%",        vdata["vol_regime"],           vdata["vol_color"])
kpi(h6, "Markov — mañana",
    mk["labels"][int(np.argmax(mk["next_day"]))],
    f"{max(mk['next_day'])*100:.0f}% probabilidad",
    [RED, YELLOW, GREEN][int(np.argmax(mk["next_day"]))])

# ═══════════════════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "⬡ RESUMEN EJECUTIVO",
    "① DIRECCIONALIDAD — Order Flow · Markov · MC",
    "② VOLATILIDAD — Cono · ATR · Distribución",
    "③ VOLUMEN — Perfil · Delta · VWAP · Anomalías",
    "④ MACRO — Contexto · Eventos · Correlaciones",
])

# ═════════════════════════════════════════════════════════════════════════════
with tab0:
    ctx_now = st.session_state.context or {}
    summ    = build_summary(r, ctx_now)
    dec     = 1 if r["price"]>1000 else 2 if r["price"]>100 else 4

    # ── VEREDICTO CENTRAL ────────────────────────────────────────────────────
    vc = summ["verdict_color"]
    st.markdown(f"""
    <div style='background:{S1};border:2px solid {vc};border-radius:6px;
        padding:28px 32px;margin-bottom:20px;text-align:center'>
      <div style='font-size:10px;letter-spacing:4px;color:{MUTED};
          text-transform:uppercase;margin-bottom:8px'>VEREDICTO OPERATIVO</div>
      <div style='font-family:Rajdhani,sans-serif;font-size:52px;font-weight:700;
          color:{vc};letter-spacing:4px;line-height:1'>{summ["verdict_icon"]} {summ["verdict"]}</div>
      <div style='margin-top:12px;display:flex;justify-content:center;gap:32px;
          font-size:11px;color:{MUTED}'>
        <span>Score: <b style='color:{vc};font-size:16px'>{summ["total_score"]:+.0f}</b>/100</span>
        <span>Convicción: <b style='color:{summ["conviction_color"]}'>{summ["conviction"]}</b></span>
        <span>Señales: <b style='color:{GREEN}'>{summ["bullish_sigs"]} alcistas</b> · <b style='color:{RED}'>{summ["bearish_sigs"]} bajistas</b> · <b style='color:{MUTED}'>{summ["neutral_sigs"]} neutrales</b></span>
        <span>Alineación: <b style='color:{TEXT}'>{summ["alignment"]*100:.0f}%</b></span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SCORE BAR ─────────────────────────────────────────────────────────────
    sc = summ["total_score"]
    pct_pos  = max(0, sc) / 100 * 100
    pct_neg  = max(0, -sc) / 100 * 100
    bar_html = f"""
    <div style='margin-bottom:20px'>
      <div style='display:flex;justify-content:space-between;font-size:9px;
          color:{MUTED};letter-spacing:2px;text-transform:uppercase;margin-bottom:4px'>
        <span>BAJISTA −100</span><span>NEUTRAL</span><span>+100 ALCISTA</span>
      </div>
      <div style='position:relative;height:12px;background:{S1};border-radius:6px;
          border:1px solid {BORDER};overflow:hidden'>
        <div style='position:absolute;left:50%;top:0;bottom:0;width:2px;
            background:{MUTED};'></div>
        {"<div style='position:absolute;left:50%;top:0;bottom:0;width:"+str(pct_pos/2)+"%;background:"+GREEN+";border-radius:0 4px 4px 0'></div>" if sc > 0 else ""}
        {"<div style='position:absolute;right:50%;top:0;bottom:0;width:"+str(pct_neg/2)+"%;background:"+RED+";border-radius:4px 0 0 4px'></div>" if sc < 0 else ""}
      </div>
    </div>"""
    st.markdown(bar_html, unsafe_allow_html=True)

    # ── SEÑALES INDIVIDUALES ─────────────────────────────────────────────────
    st.markdown("### Señales por Modelo")
    cols = st.columns(len(summ["signals"]))
    for col, sig in zip(cols, summ["signals"]):
        bar_w = abs(sig["score"]) / sig["peso"] * 100
        bar_c = sig["color"] if sig["score"] != 0 else MUTED
        col.markdown(f"""<div class='kpi' style='border-top:3px solid {sig["color"]}'>
            <div style='font-size:16px;margin-bottom:4px'>{sig["icono"]}</div>
            <div class='kpi-lbl'>{sig["categoria"]}</div>
            <div style='font-family:Rajdhani,sans-serif;font-size:20px;font-weight:700;
                color:{sig["color"]};line-height:1.2;margin-bottom:4px'>{sig["valor"]}</div>
            <div style='font-size:10px;color:{sig["color"]};margin-bottom:8px'>{sig["label"]}</div>
            <div style='height:4px;background:{BORDER};border-radius:2px;overflow:hidden;margin-bottom:6px'>
              <div style='height:100%;width:{bar_w:.0f}%;background:{bar_c};border-radius:2px'></div>
            </div>
            <div style='font-size:9px;color:{MUTED}'>Aporte: <b style='color:{sig["color"]}'>{sig["score"]:+.1f}</b> / peso {sig["peso"]}</div>
        </div>""", unsafe_allow_html=True)

    # Detalles expandibles
    with st.expander("📋 Ver razonamiento detallado de cada señal"):
        for sig in summ["signals"]:
            st.markdown(f"""<div style='padding:10px 14px;margin-bottom:6px;
                background:{S0};border-left:3px solid {sig["color"]};border-radius:3px'>
                <b style='color:{sig["color"]}'>{sig["icono"]} {sig["categoria"]}</b>
                <span style='color:{MUTED};font-size:11px;margin-left:8px'>Score: {sig["score"]:+.1f}</span><br>
                <span style='font-size:12px;color:{TEXT}'>{sig["detalle"]}</span>
            </div>""", unsafe_allow_html=True)

    st.divider()

    # ── ZONAS CLAVE DE VOLUMEN ────────────────────────────────────────────────
    st.markdown("### Zonas Clave de Volumen")
    price_now = r["price"]

    # Visual de niveles
    zones    = summ["key_zones"]
    all_levs = [z["nivel"] for z in zones] + [price_now]
    mn_z, mx_z = min(all_levs), max(all_levs)
    span_z   = mx_z - mn_z if mx_z != mn_z else 1e-10

    zone_html = f"""<div style='background:{S0};border:1px solid {BORDER};
        border-radius:4px;padding:16px 20px;margin-bottom:14px'>
      <div style='font-size:9px;letter-spacing:3px;color:{MUTED};
          text-transform:uppercase;margin-bottom:14px'>MAPA DE NIVELES — arriba = precio más alto</div>"""

    # Sort all levels descending and render
    all_items = [(z["nivel"], z["tipo"], z["color"], z["desc"]) for z in zones]
    all_items.append((price_now, "PRECIO ACTUAL", TEXT, "Último cierre H4"))
    all_items.sort(key=lambda x: x[0], reverse=True)

    for nivel, tipo, color, desc in all_items:
        pct   = (nivel - mn_z) / span_z * 80 + 10  # 10-90%
        is_price = tipo == "PRECIO ACTUAL"
        dist_pct = (price_now - nivel) / price_now * 100 if not is_price else 0
        dist_str = f"{'▲' if dist_pct>0 else '▼'} {abs(dist_pct):.3f}%" if not is_price else "← AQUÍ"

        zone_html += f"""
        <div style='display:flex;align-items:center;gap:12px;
            padding:{'10px 12px' if is_price else '7px 12px'};margin-bottom:4px;
            background:{'rgba(255,255,255,0.04)' if is_price else 'transparent'};
            border-radius:3px;{"border:1px solid "+color+";" if is_price else ""}'>
          <div style='width:70px;font-family:Rajdhani,sans-serif;font-size:{'16px' if is_price else '13px'};
              font-weight:700;color:{color};text-align:right'>{nivel:.{dec}f}</div>
          <div style='flex:1;position:relative;height:6px;background:{BORDER};border-radius:3px'>
            <div style='position:absolute;left:{100-(nivel-mn_z)/span_z*100:.0f}%;top:-3px;
                width:{'10px' if is_price else '6px'};height:{'12px' if is_price else '8px'};
                background:{color};border-radius:2px;transform:translateX(-50%)'></div>
          </div>
          <div style='width:80px;font-size:10px;color:{color};
              font-weight:{"700" if is_price else "400"};letter-spacing:1px'>{tipo}</div>
          <div style='width:80px;font-size:10px;color:{MUTED};text-align:right'>{dist_str}</div>
          <div style='font-size:10px;color:{MUTED};flex:2'>{desc}</div>
        </div>"""

    zone_html += "</div>"
    st.markdown(zone_html, unsafe_allow_html=True)

    # Interpretación de zonas
    poc, vah, val_v, vwap_now = summ["poc"], summ["vah"], summ["val"], summ["vwap"]
    zk1, zk2 = st.columns(2)
    with zk1:
        # Zona de interés más cercana
        distances = {
            "POC":  abs(price_now - poc),
            "VAH":  abs(price_now - vah),
            "VAL":  abs(price_now - val_v),
            "VWAP": abs(price_now - vwap_now),
        }
        nearest   = min(distances, key=distances.get)
        nearest_d = distances[nearest] / price_now * 100
        nearest_v = {"POC":poc,"VAH":vah,"VAL":val_v,"VWAP":vwap_now}[nearest]
        n_color   = {"POC":ORANGE,"VAH":BLUE,"VAL":BLUE,"VWAP":YELLOW}[nearest]

        st.markdown(f"""<div class='entry-box' style='border-left-color:{n_color}'>
            <div style='font-size:9px;letter-spacing:3px;color:{n_color};
                text-transform:uppercase;margin-bottom:8px'>📍 ZONA MÁS CERCANA</div>
            <div style='font-family:Rajdhani,sans-serif;font-size:24px;font-weight:700;
                color:{n_color};margin-bottom:6px'>{nearest} {nearest_v:.{dec}f}</div>
            <div style='font-size:12px;color:{TEXT};line-height:1.8'>
            A <b>{nearest_d:.3f}%</b> del precio actual ({price_now:.{dec}f}).<br>
            {"El POC actúa como imán — el precio tiende a volver a él. Zona de alta probabilidad de reacción." if nearest=="POC" else
             "VAH es resistencia clave del Value Area. Ruptura con volumen = alcista confirmado." if nearest=="VAH" else
             "VAL es soporte clave del Value Area. Ruptura con volumen = bajista confirmado." if nearest=="VAL" else
             "VWAP es el nivel de referencia institucional. Por encima = largo, por debajo = corto."}
            </div>
        </div>""", unsafe_allow_html=True)

    with zk2:
        # Estructura de precio en el mapa de zonas
        in_va = val_v <= price_now <= vah
        if price_now > vah:
            zona_txt = f"Precio <b style='color:{GREEN}'>por encima del Value Area</b> (VAH {vah:.{dec}f}). Compradores en control. El VAH se convierte en soporte. Próxima resistencia: máximos del período."
            zona_col = GREEN
        elif price_now < val_v:
            zona_txt = f"Precio <b style='color:{RED}'>por debajo del Value Area</b> (VAL {val_v:.{dec}f}). Vendedores en control. El VAL se convierte en resistencia. Próximo soporte: mínimos del período."
            zona_col = RED
        elif price_now > poc:
            zona_txt = f"Precio <b style='color:'#69f0ae''>en el Value Area, sobre el POC</b> ({poc:.{dec}f}). Zona de equilibrio con ligero sesgo alcista. El POC ({poc:.{dec}f}) actúa como soporte inmediato."
            zona_col = "#69f0ae"
        else:
            zona_txt = f"Precio <b style='color:#ff6b6b'>en el Value Area, bajo el POC</b> ({poc:.{dec}f}). Zona de equilibrio con ligero sesgo bajista. El POC ({poc:.{dec}f}) actúa como resistencia inmediata."
            zona_col = "#ff6b6b"

        st.markdown(f"""<div class='entry-box' style='border-left-color:{zona_col}'>
            <div style='font-size:9px;letter-spacing:3px;color:{zona_col};
                text-transform:uppercase;margin-bottom:8px'>🗺️ ESTRUCTURA DE PRECIO</div>
            <div style='font-size:12px;color:{TEXT};line-height:1.8'>{zona_txt}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── CHECKLIST OPERATIVA ───────────────────────────────────────────────────
    st.markdown("### ✅ Checklist Operativa")

    checks = [
        (abs(summ["total_score"]) >= 40,
         f"Score total ≥ 40 ({summ['total_score']:+.0f})",
         "Señal de suficiente fuerza direccional"),
        (summ["conviction"] in ("ALTA","MEDIA"),
         f"Convicción {summ['conviction']}",
         "Al menos 50% de señales alineadas"),
        (abs(r["last_z"]) >= 1.5,
         f"Z-Diff fuera de zona neutral ({r['last_z']:.3f})",
         "Flujo institucional confirmado"),
        (max(r["adj_bull"], r["adj_bear"]) >= 60,
         f"Monte Carlo ≥ 60% ({max(r['adj_bull'],r['adj_bear']):.1f}%)",
         "Probabilidad estadística suficiente"),
        (summ["vol_regime"] != "COMPRESIÓN",
         f"Régimen de volatilidad: {summ['vol_regime']}",
         "No operar en compresión sin ruptura confirmada"),
        (not in_va or abs(price_now - poc) / price_now * 100 > 0.1,
         f"Precio alejado del POC ({poc:.{dec}f})",
         "POC es zona de equilibrio, peor ratio R:R"),
        (ctx_now.get("macro", 0) != 0 or ctx_now.get("news", 0) != 0,
         "Contexto macro cargado",
         "Añade contexto en Tab ④ para señal más precisa"),
    ]

    for ok, title, detail in checks:
        icon  = "✅" if ok else "❌"
        color = GREEN if ok else RED
        st.markdown(f"""<div style='display:flex;align-items:center;gap:12px;
            padding:8px 14px;margin-bottom:4px;background:{S0};
            border-left:3px solid {color};border-radius:3px'>
            <span style='font-size:16px'>{icon}</span>
            <div>
                <div style='font-size:12px;color:{TEXT};font-weight:500'>{title}</div>
                <div style='font-size:10px;color:{MUTED}'>{detail}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    ok_count = sum(1 for ok, _, _ in checks if ok)
    st.markdown(f"""<div style='text-align:center;margin-top:12px;font-size:11px;color:{MUTED}'>
        {ok_count}/{len(checks)} criterios cumplidos
        {"— <b style='color:"+GREEN+"'>Condiciones favorables para operar</b>" if ok_count >= 5
         else "— <b style='color:"+ORANGE+"'>Condiciones parciales — reduce tamaño</b>" if ok_count >= 3
         else "— <b style='color:"+RED+"'>Condiciones insuficientes — no operar</b>"}
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    df  = st.session_state.df

    st.markdown(f"#### 📈 {ticker} H4 — Monte Carlo ({r['mc_paths'].shape[0]:,} sims, {r['mc_paths'].shape[1]} pasos) + Histograma de Probabilidad")
    fig_main, bull_pct = fig_price_with_mc(df, r["mc_paths"])
    st.plotly_chart(fig_main, use_container_width=True)

    # Z-Diff card + Entry card
    cz, ce = st.columns(2)
    with cz:
        st.markdown(f"""<div class='signal-box' style='border:1px solid {zctx["color"]};
            border-left:4px solid {zctx["color"]};background:{S1}'>
            <div style='font-family:Rajdhani,sans-serif;font-size:18px;font-weight:700;
                color:{zctx["color"]};letter-spacing:2px;margin-bottom:6px'>{zctx["signal"]}</div>
            <div style='font-size:10px;color:{MUTED};margin-bottom:8px'>
                Z: <b style='color:{zctx["color"]}'>{r["last_z"]:.3f}</b> &nbsp;·&nbsp;
                Precio: <b>{'↑ subiendo' if zctx['rising'] else '↓ cayendo'}</b> &nbsp;·&nbsp;
                Rango: <b>{zctx['price_pct']*100:.0f}%</b>
                {'&nbsp;·&nbsp; <span style="color:#ff9100">⚠ ZONA EXTREMA</span>' if zctx["extreme"] else ""}
                {'&nbsp;·&nbsp; <span style="color:#00e676">💥 RUPTURA ALCISTA</span>' if zctx["breaking_up"] else ""}
                {'&nbsp;·&nbsp; <span style="color:#ff1744">💥 RUPTURA BAJISTA</span>' if zctx["breaking_down"] else ""}
            </div>
            <div style='font-size:12px;color:{TEXT};line-height:1.7'>{zctx["expl"]}</div>
        </div>""", unsafe_allow_html=True)
    with ce:
        st.markdown(f"""<div class='entry-box'>
            <div style='font-size:9px;letter-spacing:3px;color:{CYAN};
                text-transform:uppercase;margin-bottom:8px'>🔍 DIAGNÓSTICO DE ENTRADA</div>
            <div style='font-size:12px;color:{TEXT};line-height:1.8'>{zctx["entry_reason"]}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # Markov
    st.markdown("#### 🔗 Cadenas de Markov — Matriz de Transición de Estados")
    st.plotly_chart(fig_markov(mk), use_container_width=True)
    mc1,mc2,mc3 = st.columns(3)
    for col, lbl, val, clr in zip([mc1,mc2,mc3], mk["labels"], mk["next_day"],
                                   [RED, YELLOW, GREEN]):
        kpi(col, f"P({lbl}) mañana", f"{val*100:.1f}%",
            f"Estado actual: {mk['current_label']}", clr)

    st.divider()

    # MC Summary + Order
    st.markdown("#### 📊 Probabilidades MC + Orden Recomendada")
    pc1,pc2,pc3,pc4 = st.columns(4)
    kpi(pc1, f"P(positivo {horizon}d)", f"{r['adj_bull']:.1f}%", "Monte Carlo GBM", GREEN)
    kpi(pc2, f"P(negativo {horizon}d)", f"{r['adj_bear']:.1f}%", "Monte Carlo GBM", RED)
    kpi(pc3, "Media MC",   f"{float(r['final'].mean()):.{dec}f}",
        f"mediana {float(np.median(r['final'])):.{dec}f}", CYAN)
    kpi(pc4, "Dispersión 1σ", f"±{float(r['final'].std()):.{dec}f}",
        f"90%: {r['p5']:.{dec}f}–{r['p95']:.{dec}f}", ORANGE)

    prob      = r["adj_bull"] if r["adj_bull"] > r["adj_bear"] else r["adj_bear"]
    prim_bull = r["adj_bull"] > r["adj_bear"]

    if prob < threshold or zctx["bull"] is None:
        st.markdown(f"""<div class='signal-box' style='border:1px solid rgba(255,214,0,.3);
            background:rgba(255,214,0,.04)'>
            <div style='font-family:Rajdhani,sans-serif;font-size:22px;font-weight:700;
                color:{YELLOW};letter-spacing:3px;margin-bottom:6px'>⚠ NO OPERAR</div>
            <div style='font-size:12px;color:{MUTED};line-height:1.9'>
                P={prob:.1f}% — por debajo del umbral {threshold}% &nbsp;·&nbsp;
                Z={r["last_z"]:.3f} — {zctx["signal"]}<br>
                <span style='color:{YELLOW}'>💡 Preservar capital es una posición válida.</span>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        side  = "BUY"  if prim_bull else "SELL"
        sc_   = GREEN  if prim_bull else RED
        sl_   = r["p8"]  if prim_bull else r["p92"]
        tp_   = r["p80"] if prim_bull else r["p20"]
        en_   = r["p80"] if prim_bull else r["p20"]
        # entry: if STOP pattern use breakout, else use limit level
        if zctx["pattern"] in ["ruptura_momentum","ruptura_confirmada","momentum_moderado"]:
            en_ = float(r["final"].mean())   # around expected price
            otype = "STOP"
        else:
            otype = "LIMIT"

        rr_   = abs(tp_-en_) / max(abs(en_-sl_), 1e-10)
        risk_usd = account*(risk_pct/100)
        sl_dist  = abs(en_-sl_)
        if instr=="Forex std (100k)":   lots=risk_usd/((sl_dist/0.0001)*10); ll=f"{lots:.2f} lotes std"
        elif instr=="Forex mini (10k)": lots=risk_usd/((sl_dist/0.0001)*1);  ll=f"{lots:.2f} mini lotes"
        elif instr=="XAU/USD":          lots=risk_usd/(sl_dist*100);          ll=f"{lots:.3f} lotes XAU"
        else:                           lots=risk_usd/max(sl_dist,1e-10);     ll=f"{lots:.2f} contratos"

        o1,o2,o3,o4,o5 = st.columns([1,1.4,1.2,2,1])
        o1.markdown(f"""<div style='text-align:center;padding:12px 6px;
            background:{"rgba(0,230,118,.1)" if prim_bull else "rgba(255,23,68,.1)"};
            border:1px solid {sc_};border-radius:4px'>
            <div style='font-family:Rajdhani,sans-serif;font-size:20px;font-weight:700;color:{sc_}'>{side}</div>
            <div style='font-size:11px;color:{sc_}'>{otype}</div>
        </div>""", unsafe_allow_html=True)
        o2.markdown(f"""<div style='padding:6px 0'>
            <div style='font-family:Rajdhani,sans-serif;font-size:24px;font-weight:700'>{en_:.{dec}f}</div>
            <div style='font-size:9px;color:{MUTED}'>entrada GTC · {horizon}d</div>
        </div>""", unsafe_allow_html=True)
        o3.markdown(f"""<div style='font-size:12px;line-height:2.1;padding:4px 0'>
            SL: <span style='color:{RED};font-weight:600'>{sl_:.{dec}f}</span><br>
            TP: <span style='color:{GREEN};font-weight:600'>{tp_:.{dec}f}</span><br>
            RR: <span style='color:{TEXT}'>1:{rr_:.1f}</span>
        </div>""", unsafe_allow_html=True)
        o4.markdown(f"""<div style='font-size:10px;color:{MUTED};line-height:1.7;padding:4px 0'>
            {zctx["entry_reason"][:220]}
        </div>""", unsafe_allow_html=True)
        o5.markdown(f"""<div style='text-align:right;padding:4px 0'>
            <div style='font-family:Rajdhani,sans-serif;font-size:28px;font-weight:700;color:{sc_}'>{prob:.1f}%</div>
            <div style='font-size:10px;color:{MUTED}'>{ll}</div>
            <div style='font-size:10px;color:{RED}'>Riesgo ${risk_usd:.0f}</div>
            <div style='font-size:10px;color:{GREEN}'>Pot. ${risk_usd*rr_:.0f}</div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    vd  = r["vol_data"]
    dec = 1 if price>1000 else 2 if price>100 else 4

    v1,v2,v3,v4,v5,v6 = st.columns(6)
    kpi(v1, "Vol Realizada (14v)",    f"{vd['rv_current']*100:.2f}%",    "anualizada",           ORANGE)
    kpi(v2, "Vol Parkinson (H-L)",    f"{vd['parkinson']*100:.2f}%",     "estimador H-L",        PURPLE)
    kpi(v3, "Vol Garman-Klass",       f"{vd['garman_klass']*100:.2f}%",  "estimador OHLC",       BLUE)
    kpi(v4, "Régimen",                vd["vol_regime"],
        f"σ corto {vd['rv_short']*100:.1f}% / largo {vd['rv_long']*100:.1f}%", vd["vol_color"])
    kpi(v5, "Movimiento 1σ (1 día)",  f"±{vd['price_1s']:.{dec}f}",
        f"±{vd['price_1s']/price*100:.3f}%", CYAN)
    kpi(v6, "Movimiento 2σ (1 día)",  f"±{vd['price_2s']:.{dec}f}",
        f"±{vd['price_2s']/price*100:.3f}%", YELLOW)

    st.markdown("#### 📉 Cono de Volatilidad · Vol Realizada · ATR · Distribución de Retornos")
    st.plotly_chart(fig_volatility(st.session_state.df, vd, r["mc_paths"]),
                     use_container_width=True)

    st.markdown("#### 📏 ATR Multi-Periodo")
    rows = []
    for w, av in vd["atr"].items():
        rows.append({
            "Periodo":         f"{w} velas H4 (~{w//6:.0f}d)",
            "ATR absoluto":    f"{av:.{dec}f}",
            "ATR % precio":    f"{av/price*100:.3f}%",
            "SL 1× ATR":       f"{av:.{dec}f}",
            "TP 2× ATR":       f"{av*2:.{dec}f}",
            "TP 3× ATR":       f"{av*3:.{dec}f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.divider()
    vi1, vi2 = st.columns(2)
    with vi1:
        st.markdown(f"""<div class='entry-box'>
            <div style='font-size:9px;letter-spacing:3px;color:{ORANGE};
                text-transform:uppercase;margin-bottom:8px'>RÉGIMEN DE VOLATILIDAD</div>
            <div style='font-size:12px;color:{TEXT};line-height:1.9'>
            {"🔴 <b>EXPANSIÓN:</b> Vol corta muy por encima de la larga. Mercado en movimiento fuerte. Aumenta SL/TP y reduce tamaño de posición." if vd["vol_regime"]=="EXPANSIÓN" else
             "🔵 <b>COMPRESIÓN:</b> Vol corta muy por debajo de la larga. Mercado comprimido — ruptura inminente. Posición pequeña ahora, aumenta tras la ruptura." if vd["vol_regime"]=="COMPRESIÓN" else
             "⚪ <b>NORMAL:</b> Régimen estable. Parámetros estándar de SL/TP. Usa el ATR como referencia directa."}
            </div>
        </div>""", unsafe_allow_html=True)
    with vi2:
        st.markdown(f"""<div class='entry-box'>
            <div style='font-size:9px;letter-spacing:3px;color:{CYAN};
                text-transform:uppercase;margin-bottom:8px'>NIVELES ESTADÍSTICOS MAÑANA</div>
            <div style='font-size:12px;color:{TEXT};line-height:2.0'>
            Precio: <b>{price:.{dec}f}</b><br>
            1σ alcista (68%): <b style='color:{GREEN}'>{price+vd["price_1s"]:.{dec}f}</b>
            &nbsp;·&nbsp; 1σ bajista: <b style='color:{RED}'>{price-vd["price_1s"]:.{dec}f}</b><br>
            2σ alcista (95%): <b style='color:{GREEN}'>{price+vd["price_2s"]:.{dec}f}</b>
            &nbsp;·&nbsp; 2σ bajista: <b style='color:{RED}'>{price-vd["price_2s"]:.{dec}f}</b>
            </div>
        </div>""", unsafe_allow_html=True)

    # Kurtosis / Fat tails warning
    rets_arr = np.diff(np.log(st.session_state.df["Close"].values.astype(float)))
    kurt = float(stats.kurtosis(rets_arr))
    skew = float(stats.skew(rets_arr))
    if abs(kurt) > 1:
        st.warning(
            f"**Colas gordas detectadas** — Kurtosis = {kurt:.2f} (normal = 0). "
            f"Los retornos de {ticker} tienen colas más gruesas que la distribución normal. "
            f"El Monte Carlo (GBM log-normal) puede **subestimar** los movimientos extremos. "
            f"Usa el **percentil 95/5** como referencia, no solo la media."
        )

# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    vp   = r["vol_profile"]
    vwap = r["vwap_series"]
    delt = r["delta_series"]
    danom= r["df_anom"]
    df   = st.session_state.df
    price_now = r["price"]
    dec  = 1 if price_now>1000 else 2 if price_now>100 else 4

    # KPIs
    poc_dist = (price_now - vp["poc"]) / price_now * 100
    vwap_dist= (price_now - float(vwap.iloc[-1])) / price_now * 100
    cum_delta= float(delt.sum())
    n_spikes = int((danom["anomaly"].isin(["ABSORCIÓN","SPIKE VOLUMEN"])).sum())
    n_abs    = int((danom["anomaly"] == "ABSORCIÓN").sum())
    n_dry    = int((danom["anomaly"] == "RUPTURA SECA").sum())

    va1,va2,va3,va4,va5,va6 = st.columns(6)
    kpi(va1, "POC (Máx. Volumen)",
        f"{vp['poc']:.{dec}f}",
        f"{'▲' if price_now>vp['poc'] else '▼'} {abs(poc_dist):.3f}% del precio",
        GREEN if price_now > vp["poc"] else RED)
    kpi(va2, "Value Area High",   f"{vp['vah']:.{dec}f}", "70% del volumen", BLUE)
    kpi(va3, "Value Area Low",    f"{vp['val']:.{dec}f}", "70% del volumen", BLUE)
    kpi(va4, "VWAP",
        f"{float(vwap.iloc[-1]):.{dec}f}",
        f"{'Por encima' if price_now>float(vwap.iloc[-1]) else 'Por debajo'} del VWAP",
        GREEN if price_now > float(vwap.iloc[-1]) else RED)
    kpi(va5, "Delta Acumulado",
        f"{cum_delta:+,.0f}",
        "comprador (+) / vendedor (−)",
        GREEN if cum_delta > 0 else RED)
    kpi(va6, "Anomalías",
        f"{n_spikes} spikes · {n_abs} absorción · {n_dry} seca",
        "velas con volumen anómalo", ORANGE if n_spikes > 0 else MUTED)

    st.markdown(f"#### 📊 {ticker} H4 — Perfil de Volumen · VWAP · Volume Delta")
    fig_vol = fig_volume_profile(df, vp, vwap, delt, danom)
    st.plotly_chart(fig_vol, use_container_width=True)

    # Interpretación POC / VA
    col_poc, col_vwap = st.columns(2)
    with col_poc:
        price_in_va = vp["val"] <= price_now <= vp["vah"]
        price_above_va = price_now > vp["vah"]
        price_below_va = price_now < vp["val"]
        if price_in_va:
            poc_interp = f"""Precio dentro del <b>Value Area</b> (entre VAL {vp['val']:.{dec}f} y VAH {vp['vah']:.{dec}f}).
            El 70% del volumen se negoció en esta zona. <b>Mercado en equilibrio</b> — sin dirección institucional clara.
            El precio tiende a regresar al POC ({vp['poc']:.{dec}f}) cuando está en VA."""
            poc_color = YELLOW
        elif price_above_va:
            poc_interp = f"""Precio <b>por encima del Value Area</b> (VAH {vp['vah']:.{dec}f}).
            Los compradores han tomado el control sacando el precio de la zona de mayor volumen.
            El VAH actúa como <b>soporte</b> en retrocesos. Si el precio regresa al VA es señal de debilidad alcista."""
            poc_color = GREEN
        else:
            poc_interp = f"""Precio <b>por debajo del Value Area</b> (VAL {vp['val']:.{dec}f}).
            Los vendedores han sacado el precio de la zona de acuerdo. El VAL actúa como <b>resistencia</b>.
            Si el precio regresa al VA sin volumen es señal de trampa bajista."""
            poc_color = RED

        st.markdown(f"""<div class='entry-box' style='border-left-color:{poc_color}'>
            <div style='font-size:9px;letter-spacing:3px;color:{poc_color};
                text-transform:uppercase;margin-bottom:8px'>📍 POC · VALUE AREA</div>
            <div style='font-size:12px;color:{TEXT};line-height:1.8'>{poc_interp}</div>
        </div>""", unsafe_allow_html=True)

    with col_vwap:
        vwap_val = float(vwap.iloc[-1])
        if price_now > vwap_val * 1.002:
            vwap_interp = f"""Precio <b>por encima del VWAP</b> ({vwap_val:.{dec}f}) en {vwap_dist:.3f}%.
            Los compradores están pagando por encima del precio medio ponderado por volumen.
            El VWAP actúa como <b>soporte dinámico</b>. Institucionales que compraron en el día están en beneficio."""
            vwap_color = GREEN
        elif price_now < vwap_val * 0.998:
            vwap_interp = f"""Precio <b>por debajo del VWAP</b> ({vwap_val:.{dec}f}) en {abs(vwap_dist):.3f}%.
            Los vendedores dominan — el precio medio ponderado está por encima del actual.
            El VWAP actúa como <b>resistencia dinámica</b>. Posiciones largas del día están en pérdida."""
            vwap_color = RED
        else:
            vwap_interp = f"""Precio <b>en el VWAP</b> ({vwap_val:.{dec}f}) — zona de equilibrio.
            Compradores y vendedores están igualados en precio medio. Sin dirección institucional clara.
            Espera separación del VWAP para tomar posición."""
            vwap_color = YELLOW

        st.markdown(f"""<div class='entry-box' style='border-left-color:{vwap_color}'>
            <div style='font-size:9px;letter-spacing:3px;color:{vwap_color};
                text-transform:uppercase;margin-bottom:8px'>📈 VWAP</div>
            <div style='font-size:12px;color:{TEXT};line-height:1.8'>{vwap_interp}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # Volume Delta interpretation
    st.markdown("#### ⚡ Volume Delta — Presión Compradora vs Vendedora")
    recent_delta  = delt.iloc[-6:].sum()   # últimas 6 velas H4 = 1 día
    delta_trend   = "compradora" if recent_delta > 0 else "vendedora"
    delta_color   = GREEN if recent_delta > 0 else RED
    delta_strength= abs(recent_delta) / (abs(delt).mean() + 1e-10)

    dc1, dc2, dc3 = st.columns(3)
    kpi(dc1, "Delta último día (6v H4)",
        f"{recent_delta:+,.0f}",
        f"Presión {delta_trend}", delta_color)
    kpi(dc2, "Fuerza del delta",
        f"{delta_strength:.1f}×",
        "vs media histórica", ORANGE if delta_strength > 2 else MUTED)
    kpi(dc3, "Delta acumulado total",
        f"{cum_delta:+,.0f}",
        "desde inicio del período", GREEN if cum_delta > 0 else RED)

    st.markdown(f"""<div class='entry-box'>
        <div style='font-size:9px;letter-spacing:3px;color:{CYAN};
            text-transform:uppercase;margin-bottom:8px'>⚡ DIAGNÓSTICO VOLUME DELTA</div>
        <div style='font-size:12px;color:{TEXT};line-height:1.8'>
        Delta acumulado del período: <b style='color:{GREEN if cum_delta>0 else RED}'>{cum_delta:+,.0f}</b>
        — presión {'compradora dominante' if cum_delta>0 else 'vendedora dominante'} en el período analizado.<br>
        Último día ({recent_delta:+,.0f}): {'Compradores acelerando' if recent_delta>0 and cum_delta>0 else
        'Vendedores acelerando' if recent_delta<0 and cum_delta<0 else
        'Divergencia: delta reciente va en contra del acumulado — posible giro inminente'}.<br>
        {'⚠️ <b>Divergencia Delta:</b> el delta reciente contradice el acumulado. Señal de posible reversión.' if (recent_delta>0) != (cum_delta>0) else ''}
        </div>
    </div>""", unsafe_allow_html=True)

    st.divider()

    # Anomaly table
    st.markdown("#### 🔍 Registro de Anomalías de Volumen")
    anom_df = danom[danom["anomaly"] != "NORMAL"][
        ["Open","High","Low","Close","volume_eff","vol_z","anomaly","anom_score"]
    ].copy().tail(20)

    if len(anom_df) > 0:
        anom_df.columns = ["Open","High","Low","Close","Vol. Efectivo","Z-Score Vol","Tipo","Score"]
        anom_df = anom_df.round({"Open":dec,"High":dec,"Low":dec,"Close":dec,
                                  "Vol. Efectivo":0,"Z-Score Vol":2,"Score":1})

        def color_anomaly(val):
            colors = {"ABSORCIÓN": f"color:{PURPLE}",
                      "SPIKE VOLUMEN": f"color:{ORANGE}",
                      "MOMENTUM": f"color:{CYAN}",
                      "RUPTURA SECA": f"color:{YELLOW}"}
            return colors.get(val, f"color:{MUTED}")

        def color_zscore(val):
            if val > 2.5:   return f"color:{ORANGE};font-weight:bold"
            elif val > 1.8: return f"color:{YELLOW}"
            elif val < -1.5:return f"color:{CYAN}"
            return f"color:{MUTED}"

        # pandas >= 2.1 renamed applymap to map; support both
        _style = anom_df.style
        _fn    = "map" if hasattr(_style, "map") else "applymap"
        styled = (getattr(_style, _fn)(color_anomaly, subset=["Tipo"])
                  .pipe(lambda s: getattr(s, _fn)(color_zscore, subset=["Z-Score Vol"])))
        st.dataframe(styled, use_container_width=True)

        # Legend
        st.markdown(f"""<div style='font-size:10px;color:{MUTED};line-height:2;margin-top:8px'>
        <span style='color:{PURPLE}'>■ ABSORCIÓN</span> — Volumen muy alto + vela pequeña. Institucional absorbiendo oferta/demanda.
        &nbsp;·&nbsp;
        <span style='color:{ORANGE}'>■ SPIKE VOLUMEN</span> — Pico de volumen extremo + movimiento fuerte. Decisión institucional.
        &nbsp;·&nbsp;
        <span style='color:{CYAN}'>■ MOMENTUM</span> — Volumen alto + cuerpo grande. Impulso real con confirmación de volumen.
        &nbsp;·&nbsp;
        <span style='color:{YELLOW}'>■ RUPTURA SECA</span> — Movimiento grande sin volumen. Ruptura sospechosa — puede revertir.
        </div>""", unsafe_allow_html=True)
    else:
        st.info("No se detectaron anomalías de volumen significativas en el período analizado.")


# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### 🌐 Contexto Macro — Prompt para Gemini / ChatGPT")

    with st.expander("📋 Copia este prompt → pégalo en Gemini/ChatGPT → pega la respuesta abajo",
                      expanded=True):
        prompt_text = st.session_state.macro_prompt or build_macro_prompt(ticker, asset_type, horizon)
        st.code(prompt_text, language="text")
        st.caption("💡 [gemini.google.com](https://gemini.google.com) o [chat.openai.com](https://chat.openai.com) — ambos gratuitos.")

    macro_input = st.text_area(
        "📥 Pega aquí el JSON de respuesta de la IA:",
        height=130,
        placeholder='{"macro":1,"macro_label":"Alcista","macro_why":"...","news":0,...}',
        key="macro_json_input"
    )
    if st.button("✅ Procesar contexto macro", use_container_width=True):
        if macro_input.strip():
            try:
                m = re.search(r'\{[\s\S]*\}', macro_input)
                parsed = json.loads(m.group()) if m else None
                if parsed:
                    st.session_state.context = parsed
                    st.success("✓ Contexto procesado. Vuelve a ejecutar el modelo para aplicarlo.")
                    st.rerun()
            except Exception as e:
                st.error(f"Error JSON: {e}")

    ctx = st.session_state.context
    if ctx:
        st.divider()
        sc_ = lambda v: GREEN if v>0 else RED if v<0 else YELLOW
        vc_ = lambda v: ORANGE if v=="high" else CYAN if v=="low" else TEXT

        m1,m2,m3 = st.columns(3)
        kpi(m1, f"Sesgo Macro ({horizon}d)", ctx.get("macro_label","—"), ctx.get("macro_why","—"), sc_(ctx.get("macro",0)))
        kpi(m2, "Noticias / Eventos",        ctx.get("news_label","—"),  ctx.get("news_why","—"),  sc_(ctx.get("news",0)))
        kpi(m3, "Volatilidad Esperada",       ctx.get("vol_label","—"),   ctx.get("vol_why","—"),   vc_(ctx.get("vol","normal")))

        st.info(f"💬 **Resumen:** {ctx.get('summary','—')}")

        # Risk events
        evts = ctx.get("risk_events", [])
        if evts:
            st.markdown("#### 📅 Eventos de Riesgo Esta Semana")
            for i, ev in enumerate(evts):
                clr = [ORANGE, YELLOW, CYAN][i%3]
                st.markdown(f"""<div style='background:{S1};border:1px solid {BORDER};
                    border-left:3px solid {clr};border-radius:3px;
                    padding:8px 14px;margin-bottom:6px;font-size:12px;color:{TEXT}'>{ev}</div>""",
                    unsafe_allow_html=True)

        # Correlations
        corrs = ctx.get("correlations", {})
        if corrs:
            st.markdown("#### 🔗 Correlaciones Institucionales")
            ccs = st.columns(len(corrs))
            for col, (k,v) in zip(ccs, corrs.items()):
                c = GREEN if "alcista" in v.lower() else RED if "bajista" in v.lower() else YELLOW
                kpi(col, k, v.upper(), "correlación", c)

        # Combined macro impact
        st.divider()
        mv = ctx.get("macro",0); nv = ctx.get("news",0)
        boost = (mv+nv)/4*8
        st.markdown(f"""<div class='entry-box'>
            <div style='font-size:9px;letter-spacing:3px;color:{CYAN};
                text-transform:uppercase;margin-bottom:8px'>IMPACTO EN EL MODELO MC</div>
            <div style='font-size:12px;color:{TEXT};line-height:1.9'>
            El contexto ajusta la probabilidad MC en
            <b style='color:{GREEN if boost>0 else RED}'>{boost:+.1f}%</b>
            sobre la base estadística pura.<br>
            Macro <b>{ctx.get("macro_label","Neutral")}</b> + Noticias <b>{ctx.get("news_label","Neutros")}</b>
            → sesgo combinado <b style='color:{sc_(mv+nv)}'>
            {"alcista" if mv+nv>0 else "bajista" if mv+nv<0 else "neutral"}</b><br>
            <span style='color:{MUTED};font-size:10px'>
            Vuelve a ejecutar el modelo para aplicar este contexto.
            </span>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.info("Sin contexto macro aún. Copia el prompt de arriba y pega la respuesta de la IA.")

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown(f"""<div style='font-size:10px;color:{MUTED};text-align:center;line-height:1.8'>
⚠️ Modelo educativo-cuantitativo. No constituye asesoramiento financiero.<br>
Monte Carlo GBM multi-step · Z-Diff Order Flow · Cadenas de Markov · Volatilidad Parkinson / Garman-Klass<br>
Datos: Yahoo Finance H4 · Sin API Key requerida
</div>""", unsafe_allow_html=True)
