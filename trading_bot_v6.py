"""
===================================================================
  BOT DE TRADING AUTOMATISÉ — VERSION PRO v6
===================================================================

CHANGEMENTS v6 :
  P0 — ALERTES TELEGRAM :
    - Alertes sur ordres exécutés, TP/Stop, circuit breaker, drawdown
    - Résumé journalier et résumé scan toutes les 15 min

  P1 — CRITÈRES OPTIMISÉS :
    - Système de scoring global (score >= 5/10) au lieu de filtres binaires
    - Gap optionnel (bonus, plus bloquant)
    - Régime RANGE accepté (ML + strict)
    - ML seuil baissé à 58%, CV min à 52%
    - VWAP assoupli à >= 1/3

  P2 — FINVIZ SCREENING :
    - Watchlist dynamique enrichie par Finviz toutes les 2h
    - Screening LONG et SHORT séparés
    - Max 25 symboles total

  P3 — DONNÉES MACRO :
    - FRED (taux 10Y, spread 10Y-2Y, chômage)
    - Fear & Greed Index
    - BTC risk-off signal
    - Contexte macro influence taille position et max trades par cycle

POUR LANCER :
  export ALPACA_KEY="..."
  export ALPACA_SECRET="..."
  export FINNHUB_KEY="..."
  export TELEGRAM_BOT_TOKEN="..."
  export TELEGRAM_CHAT_ID="..."
  export FRED_API_KEY="..."
  python trading_bot_v6.py
===================================================================
"""

import os
import csv
import json
import time
import logging
import warnings
from datetime import datetime, timedelta
from functools import wraps
from typing import Optional

import numpy as np
import pandas as pd
import pytz
import requests

import talib
import finnhub

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    StopOrderRequest,
    LimitOrderRequest,
    TrailingStopOrderRequest,
    GetOrdersRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# LOGGING — fichier + console (remplace tous les print())
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TradingBotV6')


# ============================================================
# CONFIGURATION
# ============================================================
API_KEY     = os.environ.get("ALPACA_KEY",    "VOTRE_CLE_ICI")
API_SECRET  = os.environ.get("ALPACA_SECRET", "VOTRE_SECRET_ICI")
FINNHUB_KEY = os.environ.get("FINNHUB_KEY",   "VOTRE_CLE_FINNHUB_ICI")

# P0 — Telegram
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")

# P3 — FRED
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

RISK_PER_TRADE   = 0.01
ATR_STOP_MULT    = 1.5
ATR_TP_MULT      = 3.0
MAX_DRAWDOWN     = 0.05
MAX_POS          = 0.10
TRAIL_ATR_MULT   = 1.0
# P1 — Seuils assouplis
ML_CONFIDENCE    = 0.58   # was 0.62
ML_RETRAIN_HOURS = 2
MIN_CV_SCORE     = 0.50   # was 0.52
ML_CONFIDENCE_RANGE = 0.63  # seuil ML plus strict en régime RANGE

# Scoring global
MIN_SCORE = 5  # score minimum sur 10 pour trader

# Circuit breaker
MAX_CONSECUTIVE_LOSSES = 3
MAX_DAILY_LOSS         = 0.02

# Persistance
TRADES_FILE       = "open_trades.json"
EQUITY_CURVE_FILE = "equity_curve.json"

# Corrélation secteur
SECTOR_MAP = {
    'NVDA': 'semiconductors', 'AMD': 'semiconductors', 'SMCI': 'semiconductors',
    'TSLA': 'ev_auto', 'COIN': 'crypto',
    'AAPL': 'big_tech', 'MSFT': 'big_tech', 'GOOGL': 'big_tech',
    'META': 'social_media', 'AMZN': 'ecommerce',
    'NFLX': 'streaming', 'CRM': 'saas', 'UBER': 'transport',
}
MAX_PER_SECTOR = 2

WATCHLIST = [
    'NVDA', 'TSLA', 'AMD', 'SMCI', 'COIN',
    'AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN',
    'NFLX', 'CRM', 'UBER'
]

NYSE_TZ = pytz.timezone("America/New_York")

# ============================================================
# CLIENTS
# ============================================================
client_trade   = TradingClient(API_KEY, API_SECRET, paper=True)
client_data    = StockHistoricalDataClient(API_KEY, API_SECRET)
finnhub_client = finnhub.Client(api_key=FINNHUB_KEY)

# ============================================================
# ÉTAT GLOBAL
# ============================================================
trades_log  = []
ml_models   = {}

initial_equity = float(client_trade.get_account().equity)


# ============================================================
# P0 — ALERTES TELEGRAM
# ============================================================
def send_telegram(message: str, silent: bool = False) -> None:
    """Envoie un message Telegram. Ne bloque pas si ça échoue."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML",
            "disable_notification": silent,
        }, timeout=5)
    except Exception:
        logger.warning("Telegram: envoi échoué")


# Alerte de démarrage
logger.info(f"🚀 BOT PRO v6 DÉMARRÉ | Capital : ${initial_equity:,.2f}")
logger.info("✅ Alpaca-only | ✅ GBM 20-features | ✅ Multi-TF | ✅ Stop ATR | ✅ No yfinance")
logger.info("✅ SHORT support | ✅ Sector filter | ✅ Circuit breaker | ✅ JSON persistence")
logger.info("✅ Telegram alerts | ✅ Finviz screening | ✅ Macro data | ✅ Scoring system")
send_telegram(
    f"🚀 <b>BOT DÉMARRÉ — v6</b>\n"
    f"Capital initial : <b>${initial_equity:,.2f}</b>\n"
    f"Watchlist : {len(WATCHLIST)} symboles\n"
    f"{datetime.now(NYSE_TZ).strftime('%Y-%m-%d %H:%M:%S')} NY"
)


# ============================================================
# P1-5 : PERSISTANCE DES TRADES OUVERTS (JSON)
# ============================================================
def save_open_trades(open_trades: dict) -> None:
    """Sauvegarde open_trades dans un fichier JSON pour survivre aux redémarrages."""
    serializable = {}
    for sym, trade in open_trades.items():
        t = trade.copy()
        t['open_time'] = t['open_time'].isoformat()
        serializable[sym] = t
    with open(TRADES_FILE, 'w') as f:
        json.dump(serializable, f, indent=2)


def load_open_trades() -> dict:
    """Charge open_trades depuis le fichier JSON au démarrage."""
    try:
        with open(TRADES_FILE, 'r') as f:
            data = json.load(f)
        for sym, trade in data.items():
            trade['open_time'] = datetime.fromisoformat(trade['open_time'])
        logger.info(f"📂 {len(data)} trade(s) ouvert(s) rechargé(s) depuis {TRADES_FILE}")
        return data
    except FileNotFoundError:
        return {}


# ============================================================
# P1-9 : PERSISTANCE DE L'EQUITY CURVE (JSON)
# ============================================================
def save_equity_curve(equity_curve: list) -> None:
    """Sauvegarde l'equity curve dans un fichier JSON."""
    with open(EQUITY_CURVE_FILE, 'w') as f:
        json.dump(equity_curve, f)


def load_equity_curve() -> list:
    """Charge l'equity curve depuis le fichier JSON au démarrage."""
    try:
        with open(EQUITY_CURVE_FILE, 'r') as f:
            data = json.load(f)
        logger.info(f"📂 Equity curve rechargée : {len(data)} points")
        return data
    except FileNotFoundError:
        return []


# Chargement au démarrage
open_trades  = load_open_trades()
equity_curve = load_equity_curve()


# ============================================================
# P1-8 : RATE LIMITING SUR get_bars (150 appels/min max)
# ============================================================
def rate_limit(calls_per_minute: int = 150):
    """
    Décorateur de rate limiting.
    Garantit un intervalle minimum entre les appels pour respecter
    les limites de l'API Alpaca (~150-200 appels/min).
    """
    min_interval = 60.0 / calls_per_minute
    last_call    = [0.0]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_call[0]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            last_call[0] = time.time()
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ============================================================
# FONCTION CENTRALE : récupérer les bougies via Alpaca
# ============================================================
@rate_limit(calls_per_minute=150)
def get_bars(symbol: str, timeframe, days: int) -> pd.DataFrame:
    """
    Récupère les bougies OHLCV depuis Alpaca.
    Remplace complètement yfinance.
    Rate-limitée à 150 appels/min pour respecter les limites Alpaca.
    Retourne un DataFrame avec colonnes : open, high, low, close, volume
    """
    try:
        end   = datetime.now(pytz.UTC)
        # Ajoute des jours supplémentaires pour compenser les week-ends
        start = end - timedelta(days=days + 10)
        bars = None
        for feed in ('sip', 'iex'):
            try:
                request = StockBarsRequest(
                    symbol_or_symbols=[symbol],
                    timeframe=timeframe,
                    start=start,
                    end=end,
                    limit=10000,
                    adjustment='raw', feed=feed
                )
                result = client_data.get_stock_bars(request)
                if not result.df.empty:
                    bars = result
                    break
            except Exception as feed_err:
                logger.debug(f"get_bars {symbol} feed={feed} : {feed_err}")
                continue
        if bars is None:
            return pd.DataFrame()

        df = bars.df
        # Alpaca retourne un MultiIndex (symbol, timestamp) — on aplatit
        if isinstance(df.index, pd.MultiIndex):
            df = df.loc[symbol] if symbol in df.index.get_level_values(0) else df.droplevel(0)

        df.index = pd.to_datetime(df.index, utc=True)
        df = df.rename(columns={
            'open': 'open', 'high': 'high', 'low': 'low',
            'close': 'close', 'volume': 'volume'
        })
        # Garder seulement les N derniers jours demandés
        cutoff = end - timedelta(days=days)
        df = df[df.index >= cutoff]
        return df.dropna()

    except Exception as e:
        logger.warning(f"get_bars {symbol} {timeframe} : {e}")
        return pd.DataFrame()


# ============================================================
# P1-7 : CACHE API (TTL ~55 secondes)
# ============================================================
_bar_cache: dict = {}


def get_bars_cached(symbol: str, timeframe, days: int, cache_ttl: int = 55) -> pd.DataFrame:
    """
    Version cachée de get_bars.
    Evite les appels redondants à l'API dans le même cycle de scan.
    TTL par défaut : 55 secondes (légèrement inférieur à 1 cycle de 60s).
    """
    key = f"{symbol}_{timeframe}_{days}"
    now = time.time()
    if key in _bar_cache:
        cached_time, cached_data = _bar_cache[key]
        if now - cached_time < cache_ttl:
            return cached_data
    data = get_bars(symbol, timeframe, days)
    _bar_cache[key] = (now, data)
    return data


# ============================================================
# HEURES DE MARCHÉ
# ============================================================
def is_market_open() -> bool:
    now_ny = datetime.now(NYSE_TZ)
    if now_ny.weekday() >= 5:
        return False
    open_  = now_ny.replace(hour=9,  minute=30, second=0, microsecond=0)
    close_ = now_ny.replace(hour=15, minute=45, second=0, microsecond=0)
    return open_ <= now_ny <= close_


def seconds_until_open() -> int:
    now_ny = datetime.now(NYSE_TZ)
    if now_ny.weekday() < 5:
        next_open = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
        if now_ny < next_open:
            return int((next_open - now_ny).total_seconds())
    days_ahead = 1
    while (now_ny.weekday() + days_ahead) % 7 >= 5:
        days_ahead += 1
    next_open = (now_ny + timedelta(days=days_ahead)).replace(
        hour=9, minute=30, second=0, microsecond=0
    )
    return int((next_open - now_ny).total_seconds())


# ============================================================
# P2-13 : FILTRE HORAIRE (évite ouverture et clôture proches)
# ============================================================
def is_safe_trading_time() -> bool:
    """
    Evite de trader dans les 5 premières minutes (trop volatile à l'ouverture)
    et les 15 dernières minutes (risque de clôture forcée en fin de séance).
    """
    now_ny       = datetime.now(NYSE_TZ)
    market_open  = now_ny.replace(hour=9,  minute=30, second=0, microsecond=0)
    market_close = now_ny.replace(hour=16, minute=0,  second=0, microsecond=0)
    minutes_since_open = (now_ny - market_open).total_seconds() / 60
    minutes_to_close   = (market_close - now_ny).total_seconds() / 60
    return minutes_since_open > 5 and minutes_to_close > 15


# ============================================================
# P2-12 : FILTRE VOLATILITÉ — volatilité annualisée (vs VIX 25)
# ============================================================
def get_volatility_filter() -> bool:
    """
    Remplace le VIX (bloqué sur AWS via yfinance).
    Calcule la volatilité réalisée annualisée de SPY sur 20 jours via Alpaca.
    Volatilité annualisée > 25% → équivalent VIX > 25 → pause.
    """
    try:
        data = get_bars_cached('SPY', TimeFrame.Day, 20)
        if len(data) < 5:
            return True
        daily_vol      = data['close'].pct_change().std()
        annualized_vol = daily_vol * np.sqrt(252) * 100
        logger.info(f"📊 Volatilité SPY annualisée : {annualized_vol:.1f}% (seuil : 25%)")
        if annualized_vol >= 25:
            send_telegram(
                f"⏸️ <b>VOLATILITÉ HAUTE</b>\n"
                f"Volatilité SPY : {annualized_vol:.1f}% (seuil 25%)\n"
                f"Trading suspendu 5 min.",
                silent=True,
            )
        return annualized_vol < 25
    except Exception as e:
        logger.warning(f"Volatilité : {e}")
        return True


# ============================================================
# P2-10 : RÉGIME DE MARCHÉ (TREND_UP, TREND_DOWN, RANGE)
# ============================================================
def market_regime(symbol: str) -> str:
    """
    Détermine le régime de marché du symbole.
    Retourne 'TREND_UP', 'TREND_DOWN' ou 'RANGE'.
    """
    try:
        data = get_bars_cached(symbol, TimeFrame.Hour, 30)
        if len(data) < 50:
            return "RANGE"
        close   = data['close'].values
        high    = data['high'].values
        low     = data['low'].values
        ema20   = talib.EMA(close, 20)[-1]
        ema50   = talib.EMA(close, 50)[-1]
        atr_pct = talib.ATR(high, low, close, 14)[-1] / close[-1]
        if ema20 > ema50 and atr_pct > 0.015:
            return "TREND_UP"
        if ema20 < ema50 and atr_pct > 0.015:
            return "TREND_DOWN"
        return "RANGE"
    except Exception as e:
        logger.warning(f"market_regime {symbol} : {e}")
        return "RANGE"


# ============================================================
# P2-14 : SENTIMENT NEWS (Finnhub amélioré)
# ============================================================
def get_news_sentiment(symbol: str) -> tuple:
    """
    Analyse le sentiment des news Finnhub.
    Utilise d'abord l'endpoint news_sentiment si disponible,
    avec fallback sur l'analyse par mots-clés.
    Guard contre les faux positifs (ex: "cuts ribbon" ≠ négatif).
    Retourne (score: int, news_ok: bool)
    """
    # Mots-clés positifs / négatifs pour le fallback
    POSITIVE_WORDS = [
        'beat', 'surge', 'jump', 'rally', 'gain', 'profit', 'record',
        'growth', 'upgrade', 'buy', 'outperform', 'strong', 'bullish',
        'raises', 'higher', 'positive', 'exceeds', 'wins', 'deal', 'partnership'
    ]
    # Mots négatifs avec contexte — on vérifie qu'il ne s'agit pas d'un faux positif
    NEGATIVE_WORDS = [
        'miss', 'fall', 'drop', 'loss', 'decline', 'downgrade',
        'sell', 'weak', 'bearish', 'lower', 'negative', 'fails', 'lawsuit',
        'investigation', 'fraud', 'warning', 'recall', 'layoff', 'debt'
    ]
    # Expressions trompeuses à exclure du scoring négatif
    FALSE_POSITIVE_PHRASES = [
        'cuts ribbon', 'cuts through', 'rate cuts expected', 'tax cuts',
        'cost cuts benefit', 'rate cut boost'
    ]

    # ── Tentative via news_sentiment (endpoint premium Finnhub) ───
    try:
        sentiment_data = finnhub_client.news_sentiment(symbol)
        if sentiment_data and 'companyNewsScore' in sentiment_data:
            score = sentiment_data['companyNewsScore']
            buzz  = sentiment_data.get('buzz', {}).get('articlesInLastWeek', 0)
            logger.info(f"📰 {symbol} sentiment Finnhub : {score:.2f} | {buzz} articles/semaine")
            # Score > 0.5 = positif, < 0.5 = négatif (centré sur 0.5)
            news_ok = score >= 0.4
            return int((score - 0.5) * 10), news_ok
    except Exception:
        pass  # Fallback sur analyse par mots-clés

    # ── Fallback : analyse par mots-clés ──────────────────────────
    try:
        from_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        to_date   = datetime.now().strftime("%Y-%m-%d")
        news      = finnhub_client.company_news(symbol, _from=from_date, to=to_date)
        if not news:
            return 0, True
        score = 0
        for article in news[:10]:
            headline = (article.get('headline', '') + ' ' + article.get('summary', '')).lower()
            score += sum(1 for w in POSITIVE_WORDS if w in headline)
            # Guard contre les faux positifs pour les mots négatifs
            for w in NEGATIVE_WORDS:
                if w in headline:
                    # Vérifie si ce n'est pas une expression trompeuse
                    is_false_positive = any(fp in headline for fp in FALSE_POSITIVE_PHRASES)
                    if not is_false_positive:
                        score -= 1
        sentiment = "🟢 POSITIF" if score > 0 else ("🔴 NÉGATIF" if score < 0 else "⚪ NEUTRE")
        logger.info(f"📰 {symbol} : {len(news)} news | {sentiment} ({score:+d})")
        return score, score >= 0
    except Exception as e:
        logger.warning(f"News {symbol} : {e}")
        return 0, True


# ============================================================
# CONSTRUCTION DES FEATURES ML
# ============================================================
def build_features(data: pd.DataFrame) -> pd.DataFrame:
    close  = data['close'].values
    high   = data['high'].values
    low    = data['low'].values
    vol    = data['volume'].values
    open_  = data['open'].values

    rsi_14     = talib.RSI(close, 14)
    rsi_7      = talib.RSI(close, 7)
    cci_14     = talib.CCI(high, low, close, 14)
    willr      = talib.WILLR(high, low, close, 14)
    mfi        = talib.MFI(high, low, close, vol, 14)
    ema9       = talib.EMA(close, 9)
    ema20      = talib.EMA(close, 20)
    ema50      = talib.EMA(close, 50)
    macd, macd_sig, _ = talib.MACD(close)
    stoch_k, stoch_d  = talib.STOCH(high, low, close)
    ema20_slope = pd.Series(ema20).pct_change(3).values
    dist_ema20  = (close - ema20) / (ema20 + 1e-9)
    dist_ema50  = (close - ema50) / (ema50 + 1e-9)
    atr_14      = talib.ATR(high, low, close, 14)
    atr_pct     = atr_14 / (close + 1e-9)
    upper, mid, lower = talib.BBANDS(close, 20)
    bb_pct      = np.where((upper - lower) != 0, (close - lower) / (upper - lower), 0.5)
    obv         = talib.OBV(close, vol.astype(float))
    obv_pct     = pd.Series(obv).pct_change().values
    vol_ma      = pd.Series(vol).rolling(20).mean().values
    vol_ratio   = vol / (vol_ma + 1e-9)
    body        = np.abs(close - open_)
    shadow      = high - low
    body_ratio  = np.where(shadow != 0, body / shadow, 0.5)
    mom5        = pd.Series(close).pct_change(5).values
    mom10       = pd.Series(close).pct_change(10).values
    mom20       = pd.Series(close).pct_change(20).values

    features = pd.DataFrame({
        'rsi_14':      rsi_14,
        'rsi_7':       rsi_7,
        'cci_14':      cci_14,
        'willr':       willr,
        'mfi':         mfi,
        'macd_hist':   macd - macd_sig,
        'stoch_k':     stoch_k,
        'stoch_d':     stoch_d,
        'ema20_slope': ema20_slope,
        'dist_ema20':  dist_ema20,
        'dist_ema50':  dist_ema50,
        'atr_pct':     atr_pct,
        'bb_pct':      bb_pct,
        'obv_pct':     obv_pct,
        'vol_ratio':   vol_ratio,
        'body_ratio':  body_ratio,
        'mom5':        mom5,
        'mom10':       mom10,
        'mom20':       mom20,
        'ema_cross':   (ema9 > ema20).astype(float),
    })
    return features.replace([np.inf, -np.inf], np.nan).dropna()


# ============================================================
# P0-3 : ENTRAÎNEMENT ML avec TimeSeriesSplit (corrige data leakage)
# ============================================================
def train_model(symbol: str) -> bool:
    """
    Entraîne le modèle GBM pour un symbole.
    Utilise TimeSeriesSplit (au lieu de cross_val_score standard) pour éviter
    le data leakage sur des données temporelles.
    """
    try:
        logger.info(f"🔧 Entraînement ML {symbol}...")
        data = get_bars(symbol, TimeFrame.Hour, 90)
        if len(data) < 80:
            logger.warning(f"{symbol} : données insuffisantes ({len(data)} bougies)")
            return False

        features = build_features(data)
        target   = (data['close'].pct_change().shift(-1) > 0).astype(int)
        common   = features.index.intersection(target.index)
        X        = features.loc[common].iloc[:-1]
        y        = target.loc[common].iloc[:-1]

        if len(X) < 40:
            logger.warning(f"{symbol} : X trop petit ({len(X)} samples après features)")
            return False

        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05,
            max_depth=3, subsample=0.8, random_state=42
        )

        # P0-3 : TimeSeriesSplit pour éviter le data leakage temporel
        n_splits  = 2 if len(X) < 100 else 3
        tscv      = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='accuracy')
        cv_mean   = cv_scores.mean()
        logger.info(f"🤖 {symbol} CV (TimeSeriesSplit) : {cv_mean:.2%} (min : {MIN_CV_SCORE:.0%})")

        if cv_mean < MIN_CV_SCORE:
            logger.warning(f"❌ {symbol} modèle rejeté (CV trop faible)")
            return False

        model.fit(X_scaled, y)
        ml_models[symbol] = {
            'model':      model,
            'scaler':     scaler,
            'features':   list(X.columns),
            'trained_at': datetime.now(),
            'cv_score':   cv_mean,
        }
        logger.info(f"✅ {symbol} validé | CV:{cv_mean:.2%} | {len(X)} samples")
        return True

    except Exception as e:
        logger.warning(f"Train {symbol} : {e}")
        return False


def predict_ml(symbol: str, data_1h: pd.DataFrame) -> Optional[float]:
    if symbol in ml_models:
        age_h = (datetime.now() - ml_models[symbol]['trained_at']).total_seconds() / 3600
        if age_h > ML_RETRAIN_HOURS:
            logger.info(f"🔄 Refresh ML {symbol}")
            train_model(symbol)
    elif symbol not in ml_models:
        train_model(symbol)

    if symbol not in ml_models:
        return None

    try:
        entry    = ml_models[symbol]
        features = build_features(data_1h)
        if len(features) == 0:
            return None
        X_last   = features.iloc[-1:][entry['features']]
        X_scaled = entry['scaler'].transform(X_last)
        return float(entry['model'].predict_proba(X_scaled)[0][1])
    except Exception as e:
        logger.warning(f"Predict {symbol} : {e}")
        return None


# ============================================================
# ANALYSE MULTI-TIMEFRAME
# ============================================================
def check_timeframe(symbol: str, timeframe, days: int) -> dict:
    """
    Analyse un timeframe donné : tendance EMA, momentum RSI/MACD, VWAP.
    Supporte maintenant à la fois les signaux LONG et SHORT.
    """
    try:
        data = get_bars_cached(symbol, timeframe, days)
        if len(data) < 50:
            return {'valid': False}

        close = data['close'].values
        high  = data['high'].values
        low   = data['low'].values

        ema9  = talib.EMA(close, 9)[-1]
        ema20 = talib.EMA(close, 20)[-1]
        ema50 = talib.EMA(close, 50)[-1]
        rsi   = talib.RSI(close, 14)[-1]
        macd_line, signal_line, _ = talib.MACD(close)
        atr   = talib.ATR(high, low, close, 14)[-1]

        typical = (data['high'] + data['low'] + data['close']) / 3
        vwap    = (typical * data['volume']).cumsum().iloc[-1] / data['volume'].cumsum().iloc[-1]

        return {
            # Signaux LONG : EMA9 > EMA20 > EMA50
            'trend':       bool(ema9 > ema20 > ema50),
            # Signaux SHORT : EMA9 < EMA20 < EMA50
            'trend_short': bool(ema9 < ema20 < ema50),
            # Momentum LONG : RSI > 50, MACD haussier
            'momentum':       bool(rsi > 50 and macd_line[-1] > signal_line[-1]),
            # Momentum SHORT : RSI < 50, MACD baissier
            'momentum_short': bool(rsi < 50 and macd_line[-1] < signal_line[-1]),
            # VWAP LONG : prix au-dessus du VWAP
            'vwap_ok':       bool(close[-1] > vwap),
            # VWAP SHORT : prix en-dessous du VWAP
            'vwap_ok_short': bool(close[-1] < vwap),
            'rsi':           float(rsi),
            'atr':           float(atr),
            'price':         float(close[-1]),
            'valid':         True,
        }
    except Exception as e:
        logger.warning(f"TF {symbol} : {e}")
        return {'valid': False}


# ============================================================
# STOP ET TP DYNAMIQUES (ATR)
# ============================================================
def calc_atr_levels(symbol: str, entry_price: float, side: str) -> tuple:
    try:
        data      = get_bars_cached(symbol, TimeFrame.Hour, 5)
        if len(data) < 15:
            raise ValueError("pas assez de données")
        atr       = talib.ATR(data['high'].values, data['low'].values, data['close'].values, 14)[-1]
        stop_dist = ATR_STOP_MULT * atr
        tp_dist   = ATR_TP_MULT   * atr
        if side == 'BUY':
            return round(entry_price - stop_dist, 2), round(entry_price + tp_dist, 2), float(atr)
        else:
            return round(entry_price + stop_dist, 2), round(entry_price - tp_dist, 2), float(atr)
    except Exception as e:
        logger.warning(f"ATR {symbol} : {e}")
        if side == 'BUY':
            return round(entry_price * 0.98, 2), round(entry_price * 1.04, 2), entry_price * 0.013
        else:
            return round(entry_price * 1.02, 2), round(entry_price * 0.96, 2), entry_price * 0.013


# ============================================================
# TAILLE DE POSITION
# ============================================================
def calc_size(entry_price: float, stop_price: float, size_multiplier: float = 1.0) -> int:
    account_size   = float(client_trade.get_account().equity)
    risk_amount    = account_size * RISK_PER_TRADE * size_multiplier
    risk_per_share = abs(entry_price - stop_price)
    if risk_per_share <= 0:
        return 2
    return max(2, int(risk_amount / risk_per_share))


# ============================================================
# GAP ANALYSIS PROFESSIONNEL — 6 PARAMÈTRES
# ============================================================
def analyse_gap(symbol: str) -> tuple:
    """
    Analyse le gap d'ouverture avec 6 paramètres :
    1. Taille du gap normalisée par l'ATR
    2. Volume ouverture vs moyenne
    3. Direction marché général (SPY)
    4. Comportement 5-15 min après ouverture
    5. Heure de détection
    6. Score global 0-6
    Retourne (gap_score, gap_direction, gap_pct) ou (0, None, 0)
    """
    try:
        now_ny = datetime.now(NYSE_TZ)

        # ── 1. Heure de détection ──────────────────────────────
        # On analyse les gaps uniquement dans les 2 premières heures
        market_open = now_ny.replace(hour=9, minute=30, second=0)
        minutes_since_open = (now_ny - market_open).total_seconds() / 60
        if minutes_since_open > 120:
            return 0, None, 0  # Gap trop tardif → ignoré
        if minutes_since_open < 5:
            return 0, None, 0  # Trop tôt → pas assez de données

        # ── 2. Données journalières pour calcul du gap ─────────
        data_day = get_bars_cached(symbol, TimeFrame.Day, 5)
        if len(data_day) < 2:
            return 0, None, 0

        prev_close = float(data_day['close'].iloc[-2])  # Clôture veille
        today_open = float(data_day['open'].iloc[-1])   # Ouverture aujourd'hui

        gap_pct       = (today_open - prev_close) / prev_close
        gap_direction = 'UP' if gap_pct > 0 else 'DOWN'

        # ── 3. Normalisation du gap par l'ATR ──────────────────
        high  = data_day['high'].values
        low   = data_day['low'].values
        close = data_day['close'].values
        atr   = talib.ATR(high, low, close, min(len(close)-1, 14))[-1]
        atr_pct = atr / prev_close

        # Seuil dynamique = 0.5 × ATR (au lieu de 1% fixe)
        gap_threshold = 0.5 * atr_pct
        if abs(gap_pct) < gap_threshold:
            return 0, None, gap_pct  # Gap trop petit → pas significatif

        score = 0

        # Point 1 : gap significatif détecté
        score += 1
        logger.info(f"📊 GAP {symbol} : {gap_pct:+.2%} (seuil ATR : {gap_threshold:.2%})")

        # ── 4. Volume ouverture vs moyenne ─────────────────────
        data_5m = get_bars_cached(symbol, TimeFrame.Minute, 3)
        if len(data_5m) >= 3:
            vol_open    = data_5m['volume'].iloc[-1]
            vol_moyenne = data_5m['volume'].mean()
            vol_ratio   = vol_open / (vol_moyenne + 1e-9)

            if vol_ratio > 1.5:  # Volume 1.5x supérieur → gap fort
                score += 1
                logger.info(f"📊 GAP {symbol} volume fort : {vol_ratio:.1f}x moyenne")
            elif vol_ratio < 0.8:  # Volume faible → gap peu fiable
                score -= 1
                logger.info(f"⚠️ GAP {symbol} volume faible : {vol_ratio:.1f}x moyenne")

        # ── 5. Direction marché général (SPY) ──────────────────
        spy_day = get_bars_cached('SPY', TimeFrame.Day, 3)
        if len(spy_day) >= 2:
            spy_gap = (float(spy_day['open'].iloc[-1]) - float(spy_day['close'].iloc[-2])) \
                      / float(spy_day['close'].iloc[-2])

            if gap_pct > 0 and spy_gap > 0:
                score += 1
                logger.info(f"📊 GAP {symbol} aligné SPY ({spy_gap:+.2%})")
            elif gap_pct < 0 and spy_gap < 0:
                score += 1
                logger.info(f"📊 GAP {symbol} aligné SPY ({spy_gap:+.2%})")
            else:
                logger.info(f"⚠️ GAP {symbol} contre SPY ({spy_gap:+.2%})")

        # ── 6. Comportement 5-15 min après ouverture ───────────
        if len(data_5m) >= 5:
            price_now  = float(data_5m['close'].iloc[-1])
            price_open = today_open

            if gap_pct > 0:
                if price_now > price_open:
                    score += 1
                    logger.info(f"✅ GAP {symbol} continuation haussière confirmée")
                else:
                    score -= 1
                    logger.info(f"⚠️ GAP {symbol} possible gap fill détecté")
            else:
                if price_now < price_open:
                    score += 1
                    logger.info(f"✅ GAP {symbol} continuation baissière confirmée")
                else:
                    score -= 1
                    logger.info(f"⚠️ GAP {symbol} possible gap fill détecté")

        # ── 7. Score final ─────────────────────────────────────
        score = max(0, score)
        logger.info(f"🎯 GAP {symbol} score final : {score}/4 | Direction : {gap_direction}")
        return score, gap_direction, gap_pct

    except Exception as e:
        logger.warning(f"Gap Analysis {symbol} : {e}")
        return 0, None, 0


# ============================================================
# P3 — DONNÉES MACRO
# ============================================================
_macro_cache: dict = {'last_update': 0, 'data': None}
MACRO_REFRESH_SECONDS = 1800  # 30 minutes


def get_fred_data(series_id: str) -> Optional[float]:
    """Appelle l'API FRED directement pour récupérer la dernière valeur d'une série."""
    if not FRED_API_KEY:
        return None
    try:
        url = "https://api.stlouisfed.org/fred/series/observations"
        resp = requests.get(url, params={
            "series_id":  series_id,
            "api_key":    FRED_API_KEY,
            "limit":      1,
            "sort_order": "desc",
            "file_type":  "json",
        }, timeout=10)
        resp.raise_for_status()
        observations = resp.json().get("observations", [])
        if observations:
            val = observations[0].get("value", ".")
            return float(val) if val != "." else None
    except Exception as e:
        logger.warning(f"FRED {series_id} : {e}")
    return None


def get_fear_greed_index() -> Optional[float]:
    """Récupère le Fear & Greed Index depuis alternative.me (gratuit)."""
    try:
        resp = requests.get("https://api.alternative.me/fng/", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return float(data["data"][0]["value"])
    except Exception as e:
        logger.warning(f"Fear&Greed : {e}")
    return None


def get_btc_change_24h() -> Optional[float]:
    """Récupère la variation BTC sur 24h via CoinGecko (gratuit)."""
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": "bitcoin", "vs_currencies": "usd", "include_24hr_change": "true"},
            timeout=10,
        )
        resp.raise_for_status()
        return float(resp.json()["bitcoin"]["usd_24h_change"])
    except Exception as e:
        logger.warning(f"BTC CoinGecko : {e}")
    return None


def get_macro_context() -> dict:
    """
    Retourne le contexte macro avec les indicateurs clés.
    Résultat: {'favorable': bool, 'details': str, 'reduce_size': bool}
    Mis en cache 30 minutes pour éviter les appels répétés.
    """
    now = time.time()
    if _macro_cache['data'] is not None and (now - _macro_cache['last_update']) < MACRO_REFRESH_SECONDS:
        return _macro_cache['data']

    issues = []
    details_parts = []

    # ── FRED ───────────────────────────────────────────────────
    dgs10   = get_fred_data("DGS10")
    t10y2y  = get_fred_data("T10Y2Y")
    unrate  = get_fred_data("UNRATE")

    if dgs10 is not None:
        details_parts.append(f"10Y={dgs10:.2f}%")
        if dgs10 > 5.0:
            issues.append(f"taux 10Y hostile ({dgs10:.2f}% > 5%)")
    if t10y2y is not None:
        details_parts.append(f"spread10Y2Y={t10y2y:.2f}%")
        if t10y2y < 0:
            issues.append(f"courbe inversée (spread {t10y2y:.2f}%)")
    if unrate is not None:
        details_parts.append(f"chômage={unrate:.1f}%")

    # ── Fear & Greed ───────────────────────────────────────────
    fgi = get_fear_greed_index()
    fgi_bonus = False
    if fgi is not None:
        details_parts.append(f"FGI={fgi:.0f}")
        if fgi < 20:
            fgi_bonus = True
            logger.info(f"😨 Extreme Fear (FGI={fgi:.0f}) → bonus signal LONG")
        elif fgi > 80:
            issues.append(f"Extreme Greed (FGI={fgi:.0f})")
            logger.info(f"😈 Extreme Greed (FGI={fgi:.0f}) → prudence")

    # ── BTC Risk-off ───────────────────────────────────────────
    btc_chg = get_btc_change_24h()
    if btc_chg is not None:
        details_parts.append(f"BTC24h={btc_chg:+.1f}%")
        if btc_chg < -5.0:
            issues.append(f"BTC risk-off ({btc_chg:+.1f}% 24h)")

    favorable   = len(issues) == 0
    reduce_size = len(issues) > 0
    details_str = " | ".join(details_parts) if details_parts else "N/A"

    if issues:
        logger.info(f"⚠️ Macro défavorable : {', '.join(issues)}")
    else:
        logger.info(f"✅ Macro favorable : {details_str}")

    result = {
        'favorable':   favorable,
        'fgi_bonus':   fgi_bonus,
        'details':     details_str,
        'reduce_size': reduce_size,
        'issues':      issues,
    }
    _macro_cache['last_update'] = now
    _macro_cache['data'] = result
    return result


# ============================================================
# P2 — FINVIZ SCREENING (watchlist dynamique)
# ============================================================
_finviz_cache: dict = {'last_update': 0, 'tickers': []}
FINVIZ_REFRESH_SECONDS = 7200  # 2 heures


def finviz_scan() -> list:
    """
    Scanne Finviz pour trouver des actions prometteuses.
    Exécuté une fois par jour au démarrage et toutes les 2 heures.
    Retourne une liste de tickers à ajouter à la watchlist temporairement.
    """
    now = time.time()
    if _finviz_cache['tickers'] and (now - _finviz_cache['last_update']) < FINVIZ_REFRESH_SECONDS:
        return _finviz_cache['tickers']

    tickers = []
    try:
        from finvizfinance.screener.overview import Overview

        # ── Critères LONG ──────────────────────────────────────
        long_filters = {
            'Exchange':             'NASDAQ',
            'Market Cap.':          '+Mid (over $2bln)',
            'Average Volume':       'Over 500K',
            'Relative Volume':      'Over 1',
            '20-Day Simple Moving Average': 'Price above SMA20',
            'Current Volume':       'Over 500K',
        }
        try:
            fov_long = Overview()
            fov_long.set_filter(filters_dict=long_filters)
            df_long = fov_long.screener_view()
            if df_long is not None and not df_long.empty and 'Ticker' in df_long.columns:
                long_tickers = df_long['Ticker'].dropna().tolist()[:5]
                logger.info(f"📈 Finviz LONG : {long_tickers}")
                tickers.extend(long_tickers)
        except Exception as e:
            logger.warning(f"Finviz LONG scan : {e}")

        # ── Critères SHORT ─────────────────────────────────────
        short_filters = {
            'Exchange':             'NASDAQ',
            'Market Cap.':          '+Mid (over $2bln)',
            'Average Volume':       'Over 500K',
            'Relative Volume':      'Over 1',
            '20-Day Simple Moving Average': 'Price below SMA20',
            'Performance':          'Down',
        }
        try:
            fov_short = Overview()
            fov_short.set_filter(filters_dict=short_filters)
            df_short = fov_short.screener_view()
            if df_short is not None and not df_short.empty and 'Ticker' in df_short.columns:
                short_tickers = df_short['Ticker'].dropna().tolist()[:5]
                logger.info(f"📉 Finviz SHORT : {short_tickers}")
                tickers.extend(short_tickers)
        except Exception as e:
            logger.warning(f"Finviz SHORT scan : {e}")

    except ImportError:
        logger.warning("finvizfinance non installé — pip install finvizfinance")
    except Exception as e:
        logger.warning(f"Finviz scan global : {e}")

    _finviz_cache['last_update'] = now
    _finviz_cache['tickers'] = tickers
    return tickers


def get_combined_watchlist() -> list:
    """
    Watchlist finale = WATCHLIST statique + top résultats Finviz.
    Maximum 25 symboles total.
    """
    finviz_tickers = finviz_scan()
    # Déduplique tout en conservant l'ordre (statique en priorité)
    combined = list(WATCHLIST)
    for t in finviz_tickers:
        if t not in combined:
            combined.append(t)
    combined = combined[:25]
    if len(combined) > len(WATCHLIST):
        extras = [t for t in combined if t not in WATCHLIST]
        logger.info(f"📋 Watchlist enrichie Finviz : +{len(extras)} ({extras})")
    return combined


# ============================================================
# P1 — ANALYSE COMPLÈTE v6 avec SCORING GLOBAL
# ============================================================
def full_analyse_pro(symbol: str):
    """
    Analyse complète multi-timeframe avec scoring global (v6).

    Score total = TF_confirmations (0-3) + VWAP_confirmations (0-3)
                + news_bonus (0-1) + gap_bonus (0-1)
                + ml_bonus (0-1) + regime_bonus (0-1)
    Maximum : 10 points. Seuil : >= 5 pour trader.

    LONG  : EMA9 > EMA20 > EMA50, RSI > 50, MACD haussier, prix > VWAP
    SHORT : EMA9 < EMA20 < EMA50, RSI < 50, MACD baissier, prix < VWAP
    """
    # P0-1 : fix timeframe 15m — utilise TimeFrame(15, TimeFrameUnit.Minute)
    tf_5m  = check_timeframe(symbol, TimeFrame.Minute,                    5)
    tf_15m = check_timeframe(symbol, TimeFrame(15, TimeFrameUnit.Minute), 10)
    tf_1h  = check_timeframe(symbol, TimeFrame.Hour,                     30)

    if not all(tf['valid'] for tf in [tf_5m, tf_15m, tf_1h]):
        return None

    # ── Détection LONG ─────────────────────────────────────
    bullish_5m  = tf_5m['trend']  and tf_5m['momentum']
    bullish_15m = tf_15m['trend'] and tf_15m['momentum']
    bullish_1h  = tf_1h['trend']  and tf_1h['momentum']

    tf_confirmations_long = sum([bullish_5m, bullish_15m, bullish_1h])

    # ── Détection SHORT ────────────────────────────────────
    bearish_5m  = tf_5m['trend_short']  and tf_5m['momentum_short']
    bearish_15m = tf_15m['trend_short'] and tf_15m['momentum_short']
    bearish_1h  = tf_1h['trend_short']  and tf_1h['momentum_short']

    tf_confirmations_short = sum([bearish_5m, bearish_15m, bearish_1h])

    # Détermine le signal dominant
    is_long  = bullish_1h  and tf_confirmations_long  >= 2
    is_short = bearish_1h  and tf_confirmations_short >= 2

    if not is_long and not is_short:
        return None

    # En cas de conflit (rare), préférer le signal le plus fort
    if is_long and is_short:
        is_long  = tf_confirmations_long  >= tf_confirmations_short
        is_short = not is_long

    sig = 'BUY' if is_long else 'SELL'
    tf_confirmations = tf_confirmations_long if is_long else tf_confirmations_short

    # ── Confirmations VWAP (P1 : assoupli à >= 1) ──────────
    if is_long:
        vwap_confirmations = sum([tf_5m['vwap_ok'], tf_15m['vwap_ok'], tf_1h['vwap_ok']])
    else:
        vwap_confirmations = sum([tf_5m['vwap_ok_short'], tf_15m['vwap_ok_short'], tf_1h['vwap_ok_short']])

    if vwap_confirmations < 1:
        return None

    # ── Régime de marché (P1 : RANGE accepté avec ML plus strict) ──
    regime = market_regime(symbol)
    # Bloquer seulement les tendances opposées
    if is_long  and regime == "TREND_DOWN":
        return None
    if is_short and regime == "TREND_UP":
        return None

    # ── News sentiment ─────────────────────────────────────
    news_score, news_ok = get_news_sentiment(symbol)
    if is_long  and not news_ok:
        return None
    if is_short and news_score > 0:  # News positives → pas de short
        return None

    # ── Gap Analysis (P1 : OPTIONNEL, bonus seulement) ─────
    gap_score, gap_direction, gap_pct = analyse_gap(symbol)
    # Le gap ne bloque plus, mais une direction opposée reste éliminatoire
    if is_long  and gap_direction == 'DOWN':
        logger.info(f"⚠️ {symbol} gap baissier → skip LONG")
        return None
    if is_short and gap_direction == 'UP':
        logger.info(f"⚠️ {symbol} gap haussier → skip SHORT")
        return None

    # ── Prédiction ML (données 1h) ─────────────────────────
    data_1h = get_bars_cached(symbol, TimeFrame.Hour, 90)
    if len(data_1h) < 80:
        return None

    ml_prob = predict_ml(symbol, data_1h)

    # Seuil ML : standard ou strict si RANGE
    ml_threshold = ML_CONFIDENCE_RANGE if regime == "RANGE" else ML_CONFIDENCE
    if ml_prob is None:
        # Aucun modèle ML disponible — mode dégradé, on continue sans filtre ML
        logger.warning(f"⚠️ ML non disponible pour {symbol}, analyse sans ML")
        score_ml = 0
    else:
        if is_long  and ml_prob < ml_threshold:
            logger.info(f"🤖 ML {symbol} : {ml_prob:.0%} < seuil LONG {ml_threshold:.0%} → skip")
            return None
        if is_short and ml_prob > (1.0 - ml_threshold):
            logger.info(f"🤖 ML {symbol} : {ml_prob:.0%} > seuil SHORT {1.0-ml_threshold:.0%} → skip")
            return None
        score_ml = 1 if (is_long and ml_prob >= ml_threshold) or (is_short and ml_prob <= 1.0 - ml_threshold) else 0

    # ── SCORING GLOBAL ─────────────────────────────────────
    # TF confirmations : 0-3
    score_tf = tf_confirmations

    # VWAP confirmations : 0-3
    score_vwap = vwap_confirmations

    # News bonus : +1 si positif
    score_news = 1 if news_ok else 0

    # Gap bonus : +1 si gap_score >= 2 et direction alignée
    gap_aligned = (gap_direction == 'UP' and is_long) or (gap_direction == 'DOWN' and is_short)
    score_gap = 1 if (gap_score >= 2 and gap_aligned) else 0

    # Régime bonus : +1 si TREND aligné
    score_regime = 1 if (
        (is_long and regime == "TREND_UP") or
        (is_short and regime == "TREND_DOWN")
    ) else 0

    total_score = score_tf + score_vwap + score_news + score_gap + score_ml + score_regime

    logger.info(
        f"🎯 SCORING {symbol} {sig} | "
        f"TF:{score_tf}/3 VWAP:{score_vwap}/3 News:{score_news} Gap:{score_gap} "
        f"ML:{score_ml} Regime:{score_regime} | TOTAL:{total_score}/10"
    )

    if total_score < MIN_SCORE:
        logger.info(f"⚠️ {symbol} score insuffisant ({total_score}/{MIN_SCORE}) → skip")
        return None

    ml_prob_str = f"{ml_prob:.0%}" if ml_prob is not None else "N/A"
    logger.info(
        f"✅ {symbol} {sig} | Score:{total_score}/10 | TF:{tf_confirmations}/3 | "
        f"VWAP:{vwap_confirmations}/3 | ML:{ml_prob_str} | Regime:{regime}"
    )
    return sig, tf_1h['rsi'], tf_5m['price'], total_score, ml_prob


# ============================================================
# SUIVI PNL ET CLÔTURE AUTO
# ============================================================
def sync_open_trades():
    symbols_to_close = []
    for sym, trade in list(open_trades.items()):
        try:
            filled_orders = client_trade.get_orders(
                GetOrdersRequest(
                    status=QueryOrderStatus.CLOSED,
                    symbols=[sym],
                    after=trade['open_time'] - timedelta(minutes=1),
                )
            )
            tp_filled   = any(str(o.id) == trade.get('tp_id')   and o.status.value == 'filled' for o in filled_orders)
            stop_filled = any(str(o.id) == trade.get('stop_id') and o.status.value == 'filled' for o in filled_orders)

            if not (tp_filled or stop_filled):
                continue

            trigger   = "TP ✅" if tp_filled else "STOP 🔴"
            exit_side = 'tp' if tp_filled else 'stop'
            filled_order = next(
                (o for o in filled_orders if str(o.id) == trade.get(f"{exit_side}_id")),
                None
            )
            if filled_order is None:
                continue
            exit_price = float(filled_order.filled_avg_price or trade[exit_side])
            half_qty     = trade['qty'] // 2

            pnl_half = (exit_price - trade['entry']) * half_qty if trade['side'] == 'BUY' \
                       else (trade['entry'] - exit_price) * half_qty

            try:
                other_id = trade['stop_id'] if tp_filled else trade['tp_id']
                client_trade.cancel_order_by_id(other_id)
            except Exception:
                pass

            if trade.get('trail_id'):
                try:
                    client_trade.cancel_order_by_id(trade['trail_id'])
                except Exception:
                    pass

            remaining_qty = trade['qty'] - half_qty
            pnl_remaining = 0.0
            if remaining_qty > 0:
                close_side  = OrderSide.SELL if trade['side'] == 'BUY' else OrderSide.BUY
                close_order = client_trade.submit_order(
                    MarketOrderRequest(symbol=sym, qty=remaining_qty,
                                       side=close_side, time_in_force=TimeInForce.GTC)
                )
                for _ in range(10):
                    time.sleep(1)
                    o = client_trade.get_order_by_id(str(close_order.id))
                    if o.status.value == 'filled':
                        break
                remaining_exit = float(client_trade.get_order_by_id(str(close_order.id)).filled_avg_price or exit_price)
                pnl_remaining  = (remaining_exit - trade['entry']) * remaining_qty if trade['side'] == 'BUY' \
                                 else (trade['entry'] - remaining_exit) * remaining_qty

            total_pnl = pnl_half + pnl_remaining
            icon = '🟢' if total_pnl > 0 else '🔴'

            trades_log.append({
                'sym': sym, 'side': trade['side'], 'qty': trade['qty'],
                'entry': trade['entry'], 'exit': exit_price,
                'stop': trade['stop'], 'tp': trade['tp'],
                'atr': trade.get('atr', 0), 'ml_prob': trade.get('ml_prob', 0),
                'pnl': round(total_pnl, 2), 'trigger': trigger,
                'open_time':  trade['open_time'].strftime("%Y-%m-%d %H:%M:%S"),
                'close_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
            logger.info(f"{icon} {trigger} {sym} | Entry:{trade['entry']:.2f} Exit:{exit_price:.2f} | PnL:${total_pnl:+.2f}")

            # P0 — Alerte Telegram TP/Stop
            tg_icon = "🟢 TP ATTEINT" if tp_filled else "🔴 STOP TOUCHÉ"
            send_telegram(
                f"{tg_icon} <b>{sym}</b>\n"
                f"Entrée : {trade['entry']:.2f} | Sortie : {exit_price:.2f}\n"
                f"PnL : <b>${total_pnl:+.2f}</b>\n"
                f"Side : {trade['side']} | Qty : {trade['qty']}"
            )

            symbols_to_close.append(sym)

        except Exception as e:
            logger.warning(f"sync {sym} : {e}")

    for sym in symbols_to_close:
        del open_trades[sym]

    # Sauvegarde après chaque modification
    if symbols_to_close:
        save_open_trades(open_trades)


# ============================================================
# DRAWDOWN
# ============================================================
def calc_drawdown() -> bool:
    try:
        current = float(client_trade.get_account().equity)
        equity_curve.append(current)
        save_equity_curve(equity_curve)
        if len(equity_curve) > 1:
            peak = max(equity_curve)
            dd   = (current - peak) / peak
            logger.info(f"📉 Drawdown : {dd:.2%} | Peak : ${peak:,.2f}")
            if dd < -MAX_DRAWDOWN:
                logger.warning("🚨 MAX DRAWDOWN — Fermeture totale !")
                send_telegram(
                    f"🚨 <b>MAX DRAWDOWN ATTEINT</b>\n"
                    f"Drawdown : {dd:.2%} (seuil : {MAX_DRAWDOWN:.0%})\n"
                    f"Capital actuel : ${current:,.2f}\n"
                    f"Fermeture de toutes les positions !"
                )
                for pos in client_trade.get_all_positions():
                    client_trade.close_position(pos.symbol)
                return True
    except Exception as e:
        logger.warning(f"Drawdown : {e}")
    return False


# ============================================================
# P1-6 : CIRCUIT BREAKER JOURNALIER
# ============================================================
def check_circuit_breaker() -> bool:
    """
    Stoppe le trading si :
    - 3 pertes consécutives dans la journée
    - Perte journalière > 2% du capital initial
    """
    today = datetime.now().strftime("%Y-%m-%d")
    today_trades = [t for t in trades_log if t['close_time'].startswith(today)]

    # Vérification des pertes consécutives
    recent_pnls = [t['pnl'] for t in today_trades[-MAX_CONSECUTIVE_LOSSES:]]
    if len(recent_pnls) >= MAX_CONSECUTIVE_LOSSES and all(p < 0 for p in recent_pnls):
        logger.warning(f"🚨 Circuit breaker : {MAX_CONSECUTIVE_LOSSES} pertes consécutives !")
        send_telegram(
            f"🚨 <b>CIRCUIT BREAKER</b>\n"
            f"{MAX_CONSECUTIVE_LOSSES} pertes consécutives détectées.\n"
            f"PnL récents : {[f'${p:+.2f}' for p in recent_pnls]}\n"
            f"Trading suspendu jusqu'à demain."
        )
        return True

    # Vérification de la perte journalière
    daily_pnl = sum(t['pnl'] for t in today_trades)
    if daily_pnl < -(initial_equity * MAX_DAILY_LOSS):
        logger.warning(f"🚨 Circuit breaker : perte journalière {daily_pnl:+.2f} > {MAX_DAILY_LOSS:.0%} du capital !")
        send_telegram(
            f"🚨 <b>CIRCUIT BREAKER — PERTE JOURNALIÈRE</b>\n"
            f"PnL du jour : <b>${daily_pnl:+.2f}</b>\n"
            f"Seuil : {MAX_DAILY_LOSS:.0%} du capital (${initial_equity * MAX_DAILY_LOSS:.2f})\n"
            f"Trading suspendu jusqu'à demain."
        )
        return True

    return False


# ============================================================
# P2-11 : FILTRE CORRÉLATION SECTEUR
# ============================================================
def check_sector_limit(symbol: str) -> bool:
    """
    Retourne True si on peut ouvrir un trade sur ce symbole (limite sectorielle non atteinte).
    Evite d'avoir plus de MAX_PER_SECTOR positions dans le même secteur.
    """
    sector = SECTOR_MAP.get(symbol)
    if not sector:
        return True  # Symbole non mappé → pas de restriction

    # Compte les positions ouvertes dans le même secteur
    same_sector_count = sum(
        1 for sym in open_trades
        if SECTOR_MAP.get(sym) == sector
    )
    if same_sector_count >= MAX_PER_SECTOR:
        logger.info(
            f"⚠️ {symbol} ({sector}) : {same_sector_count} positions dans le secteur "
            f"(max:{MAX_PER_SECTOR}) → skip"
        )
        return False
    return True


# ============================================================
# STATISTIQUES
# ============================================================
def update_stats():
    pnls = [t['pnl'] for t in trades_log if t.get('pnl', 0) != 0]
    if len(pnls) < 2:
        return
    wins    = sum(1 for p in pnls if p > 0)
    winrate = wins / len(pnls)
    std     = np.std(pnls)
    sharpe  = (np.mean(pnls) / std * np.sqrt(252)) if std > 0 else 0
    avg_win  = np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0
    avg_loss = np.mean([p for p in pnls if p < 0]) if any(p < 0 for p in pnls) else 0
    rr       = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    logger.info(f"📊 Winrate:{winrate:.1%} | Sharpe:{sharpe:.2f} | R:R:{rr:.2f} | PnL:${sum(pnls):+.2f} | Trades:{len(pnls)}")


# ============================================================
# RAPPORT CSV
# ============================================================
def save_session_report():
    if not trades_log:
        logger.info("📋 Aucun trade complété cette session.")
        return
    filename = f"rapport_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fieldnames = ['sym', 'side', 'qty', 'entry', 'exit', 'stop', 'tp',
                  'atr', 'ml_prob', 'pnl', 'trigger', 'open_time', 'close_time']
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trades_log)
    pnls    = [t['pnl'] for t in trades_log]
    total   = sum(pnls)
    wins    = sum(1 for p in pnls if p > 0)
    winrate = wins / len(pnls) if pnls else 0
    logger.info("=" * 55)
    logger.info("📋  RAPPORT DE SESSION — BOT PRO v6")
    logger.info("=" * 55)
    logger.info(f"  Trades        : {len(trades_log)}")
    logger.info(f"  Winrate       : {winrate:.1%}")
    logger.info(f"  PnL total     : ${total:+.2f}")
    logger.info(f"  Meilleur      : ${max(pnls):+.2f}")
    logger.info(f"  Pire          : ${min(pnls):+.2f}")
    logger.info(f"  Capital final : ${float(client_trade.get_account().equity):,.2f}")
    logger.info(f"  CSV           : {filename}")
    logger.info("=" * 55)

    # P0 — Rapport journalier Telegram
    send_telegram(
        f"📊 <b>RAPPORT JOURNALIER — v6</b>\n"
        f"Trades : {len(trades_log)} | Winrate : {winrate:.1%}\n"
        f"PnL total : <b>${total:+.2f}</b>\n"
        f"Meilleur : ${max(pnls):+.2f} | Pire : ${min(pnls):+.2f}\n"
        f"Capital final : ${float(client_trade.get_account().equity):,.2f}"
    )


# ============================================================
# PRÉ-ENTRAÎNEMENT ML AU DÉMARRAGE (watchlist dynamique complète)
# ============================================================
# 1. D'abord, scan Finviz pour construire la watchlist complète
logger.info("\n📊 Scan Finviz initial pour construire la watchlist dynamique...")
startup_watchlist = get_combined_watchlist()
logger.info(f"📋 Watchlist complète au démarrage : {len(startup_watchlist)} symboles")
logger.info(f"   Statique ({len(WATCHLIST)}) : {WATCHLIST}")
finviz_extras = [s for s in startup_watchlist if s not in WATCHLIST]
if finviz_extras:
    logger.info(f"   Finviz   (+{len(finviz_extras)}) : {finviz_extras}")
else:
    logger.info("   Finviz   : aucun symbole supplémentaire (Finviz indisponible ou pas de résultat)")

# 2. Pré-entraîner les modèles ML sur TOUS les symboles
logger.info(f"\n🔧 Pré-entraînement ML sur {len(startup_watchlist)} symboles...")
for sym in startup_watchlist:
    if not train_model(sym):
        # Retry une fois après 2s en cas d'échec (rate limit ou erreur transitoire)
        time.sleep(2)
        if not train_model(sym):
            logger.debug(f"⚠️ {sym} : modèle non validé après retry")
    time.sleep(0.5)

if len(ml_models) == 0:
    logger.warning(
        "⚠️ AUCUN modèle ML validé au démarrage — le bot fonctionnera en MODE DÉGRADÉ "
        "(sans filtre ML, score réduit). Vérifiez la connexion et les données de marché."
    )
    send_telegram(
        "⚠️ <b>MODE DÉGRADÉ</b> — Aucun modèle ML validé au démarrage.\n"
        f"Symboles analysés : {len(startup_watchlist)}\n"
        "Le bot continuera à trader avec les critères techniques (multi-TF, VWAP, news, gap) "
        "mais sans le filtre ML. Les modèles seront réessayés à la demande."
    )
else:
    logger.info(f"✅ Modèles prêts : {len(ml_models)}/{len(startup_watchlist)} validés\n")

# 3. Alerte Telegram de confirmation
send_telegram(
    f"🔧 <b>PRÉ-ENTRAÎNEMENT TERMINÉ</b>\n"
    f"Symboles analysés : {len(startup_watchlist)}\n"
    f"Modèles validés : {len(ml_models)}/{len(startup_watchlist)}\n"
    f"Watchlist statique : {len(WATCHLIST)} | Finviz : +{len(finviz_extras)}\n"
    f"{'✅ Prêt à trader !' if len(ml_models) > 0 else '⚠️ Mode dégradé — 0 modèles validés'}"
)


# ============================================================
# BOUCLE PRINCIPALE
# ============================================================
logger.info(f"🚀 DÉMARRAGE {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("=" * 60)
trades_count      = 0
last_scan_report  = 0   # timestamp dernier résumé scan Telegram

try:
    while True:
        now_str = datetime.now().strftime('%H:%M:%S')

        if not is_market_open():
            wait_sec = seconds_until_open()
            logger.info(f"🕐 [{now_str}] Marché fermé — {wait_sec//60} min avant ouverture")
            if datetime.now(NYSE_TZ).hour >= 16:
                save_session_report()
            time.sleep(min(wait_sec, 300))
            continue

        # P2-13 : Filtre horaire (5 premières et 15 dernières minutes)
        if not is_safe_trading_time():
            logger.info(f"⏸️  [{now_str}] Heure non sûre pour trader (trop proche ouverture/clôture)")
            time.sleep(60)
            continue

        logger.info(f"\n⏱️  SCAN — {now_str}")

        if open_trades:
            sync_open_trades()

        if calc_drawdown():
            break

        # P1-6 : Circuit breaker journalier
        if check_circuit_breaker():
            logger.warning("🚨 Circuit breaker activé — pause trading jusqu'à demain")
            time.sleep(3600)
            continue

        if not get_volatility_filter():
            logger.info("⏸️  Volatilité trop élevée — pause 5 min")
            time.sleep(300)
            continue

        # P3 — Contexte macro
        macro = get_macro_context()
        max_trades_cycle = 2 if macro['favorable'] else 1
        size_multiplier  = 1.0 if not macro['reduce_size'] else 0.5
        if not macro['favorable']:
            logger.info(f"⚠️ Macro défavorable → max {max_trades_cycle} trade/cycle, taille ×{size_multiplier}")

        # P2 — Watchlist enrichie Finviz
        current_watchlist = get_combined_watchlist()

        opportunities = []
        for sym in current_watchlist:
            if sym in open_trades:
                continue
            result = full_analyse_pro(sym)
            if result:
                sig, rsi_val, prix, score, ml_prob = result
                opportunities.append((sym, sig, rsi_val, prix, score, ml_prob))

        # Trier par score décroissant puis RSI
        opportunities.sort(key=lambda x: (x[4], x[2]), reverse=True)

        for sym, sig, rsi_val, prix, score, ml_prob in opportunities[:max_trades_cycle]:
            # P2-11 : Vérification de la limite sectorielle
            if not check_sector_limit(sym):
                continue

            try:
                positions      = client_trade.get_all_positions()
                total_exposure = sum(float(p.qty) * float(p.current_price) for p in positions)
                equity_now     = float(client_trade.get_account().equity)
                if total_exposure > equity_now * MAX_POS:
                    logger.warning(f"⚠️ Exposition max → skip {sym}")
                    continue
            except Exception as e:
                logger.warning(f"Exposition {sym} : {e}")
                continue

            side       = OrderSide.BUY if sig == 'BUY' else OrderSide.SELL
            close_side = OrderSide.SELL if sig == 'BUY' else OrderSide.BUY

            # Calcul préliminaire de la taille avec le prix indicatif
            stop_p_prelim, _, _ = calc_atr_levels(sym, prix, sig)
            qty      = calc_size(prix, stop_p_prelim, size_multiplier)

            try:
                # ── Soumettre l'ordre d'entrée avec la taille calculée ────
                entry_order = client_trade.submit_order(
                    MarketOrderRequest(symbol=sym, qty=qty, side=side, time_in_force=TimeInForce.GTC)
                )

                # P0-2 : Polling du fill réel (max 15s)
                real_entry = None
                for _ in range(15):
                    time.sleep(1)
                    order_status = client_trade.get_order_by_id(str(entry_order.id))
                    if order_status.status.value == 'filled':
                        real_entry = float(order_status.filled_avg_price)
                        break
                else:
                    # Pas rempli après 15s → annuler et passer au suivant
                    logger.warning(f"⏱️ {sym} ordre non rempli après 15s → annulation")
                    client_trade.cancel_order_by_id(str(entry_order.id))
                    continue

                # Recalcule stop/TP avec le vrai prix de fill
                stop_p, tp_p, atr_val = calc_atr_levels(sym, real_entry, sig)
                half_qty   = qty // 2
                trail_dollar = round(TRAIL_ATR_MULT * atr_val, 2)

                # ── Soumettre stop, TP et trailing stop ──────────
                # P0-4 : Si stop/TP échoue, fermeture immédiate de la position
                try:
                    stop_order = client_trade.submit_order(
                        StopOrderRequest(symbol=sym, qty=half_qty, side=close_side,
                                         stop_price=stop_p, time_in_force=TimeInForce.GTC)
                    )
                    tp_order = client_trade.submit_order(
                        LimitOrderRequest(symbol=sym, qty=half_qty, side=close_side,
                                          limit_price=tp_p, time_in_force=TimeInForce.GTC)
                    )
                    trail_order = client_trade.submit_order(
                        TrailingStopOrderRequest(symbol=sym, qty=half_qty, side=close_side,
                                                 trail_price=trail_dollar, time_in_force=TimeInForce.GTC)
                    )
                except Exception as e:
                    logger.error(f"🚨 Protection échouée pour {sym}, fermeture immédiate : {e}")
                    client_trade.submit_order(
                        MarketOrderRequest(symbol=sym, qty=qty, side=close_side, time_in_force=TimeInForce.GTC)
                    )
                    continue

                open_trades[sym] = {
                    'entry':    real_entry, 'qty': qty, 'side': sig,
                    'stop':     stop_p,     'tp':  tp_p, 'atr': atr_val, 'ml_prob': ml_prob,
                    'stop_id':  str(stop_order.id),  'tp_id':    str(tp_order.id),
                    'trail_id': str(trail_order.id), 'open_time': datetime.now(),
                }
                # P1-5 : Sauvegarde immédiate
                save_open_trades(open_trades)
                logger.info(
                    f"✅ {sig} {qty}x {sym} @{real_entry:.2f} | "
                    f"ATR:{atr_val:.2f} | Stop:{stop_p:.2f} | TP:{tp_p:.2f} | "
                    f"ML:{f'{ml_prob:.0%}' if ml_prob is not None else 'N/A'} | Score:{score}/10"
                )
                trades_count += 1

                # P0 — Alerte Telegram ordre exécuté
                send_telegram(
                    f"✅ <b>ORDRE EXÉCUTÉ</b>\n"
                    f"Symbole : <b>{sym}</b> | Side : {sig}\n"
                    f"Qty : {qty} @ ${real_entry:.2f}\n"
                    f"Stop : ${stop_p:.2f} | TP : ${tp_p:.2f}\n"
                    f"ML score : {f'{ml_prob:.0%}' if ml_prob is not None else 'N/A'} | Score total : {score}/10"
                )

            except Exception as e:
                logger.error(f"❌ Ordre {sym} : {e}")

        update_stats()

        # P0 — Résumé scan toutes les 15 minutes (pas à chaque cycle)
        now_ts = time.time()
        if now_ts - last_scan_report >= 900:  # 15 minutes
            last_scan_report = now_ts
            try:
                equity_now = float(client_trade.get_account().equity)
                pnl_session = equity_now - initial_equity
                positions_info = ""
                if open_trades:
                    positions_info = "\n".join(
                        f"  {sym}: {t['side']} {t['qty']}x @{t['entry']:.2f}"
                        for sym, t in open_trades.items()
                    )
                else:
                    positions_info = "Aucune position ouverte"
                send_telegram(
                    f"📈 <b>RÉSUMÉ SCAN</b> — {now_str}\n"
                    f"Capital : ${equity_now:,.2f} | PnL session : ${pnl_session:+.2f}\n"
                    f"Positions ouvertes ({len(open_trades)}) :\n{positions_info}\n"
                    f"Macro : {macro['details']}",
                    silent=True,
                )
            except Exception as e:
                logger.warning(f"Résumé scan Telegram : {e}")

        logger.info(f"\n📈 Ouverts:{len(open_trades)} | Total:{trades_count} | ⏳ 60s")
        time.sleep(60)

except KeyboardInterrupt:
    logger.info("\n⏹️  ARRÊT MANUEL")

save_session_report()
logger.info("🏁 BOT v6 TERMINÉ")
