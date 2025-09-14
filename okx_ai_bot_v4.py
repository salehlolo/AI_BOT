#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OKX AI Trader - Single File Edition (no .env, no YAML)
- Keys & config embedded below in CONFIG
- Top20 scan, single position, Telegram alerts, hourly PnL report
- TP/SL dynamic using ATR & model confidence
- Capital: 85% with x10 leverage (configurable)

Usage (Windows PowerShell):
  py -3.11 -m pip install ccxt pandas numpy scikit-learn requests joblib
  py -3.11 okx_ai_bot.py download --since_days 120   # optional
  py -3.11 okx_ai_bot.py train
  py -3.11 okx_ai_bot.py backtest
  py -3.11 okx_ai_bot.py trade

Security note: rotate your API keys after testing and keep this file private.
"""

import os, sys, math, argparse, time
from pathlib import Path
from datetime import datetime, timezone

# ------------ USER CONFIG (edit safely) ------------
CONFIG = {
    "exchange": "okx",
    "okx_demo": True,
    "market_type": "swap",         # "spot" أو "swap"
    "td_mode": "isolated",         # "isolated" أو "cross"
    "symbol": "BTC/USDT:USDT",     # يستخدم عند تعطيل Top20
    "timeframe": "5m",
    "limit_bars": 5000,
    "lookback": 100,
    "fee_rate": 0.0005,
    "slippage_bps": 3,
    "paper_trading": False,         # نعتمد تنفيذ ديمو حقيقي وليس محاكاة محلية
    "execute_orders": True,         # تفعيل إنشاء أوامر فعلية على Sandbox
    "starting_balance": 10000.0,
    "model_path": "model.pkl",
    "capital_pct": 0.85,           # 85% of balance
    "leverage": 10,                # x10
    "poll_sec": 10,                 # زمن النوم في الحلقة (ثوانٍ)
    # (خيارات تسريع الدخولات - Aggressive Entries):
    "scan_eval_top": 8,             # قيّم فقط أول 8 رموز من Top20 لتقليل التأخير
    "proba_buy_threshold": 0.60,    # العتبة الأساسية
    "preentry_margin": 0.05,        # سماح مبكر: ادخل لو p_up ≥ (threshold - 0.05) مع زخم
    "enable_momentum_gate": True,   # بوابة زخم لتفادي الإشارات الضعيفة
    "min_rsi": 48,                  # RSI أدنى للزخم
    "macd_rising_bars": 2,          # عدد شموع ارتفاع متتالية في macd_hist
    "risk": {
        "base_atr_stop_mult": 1.8,
        "base_atr_tp_mult": 3.0,
        "max_position_pct": 0.9
    },
    "top_scan": {
        "enabled": True,
        "quote": "USDT",
        "n": 20,
        "sort_by": "volume"        # "volume" or "openInterest"
    },
    "telegram": {
        "enabled": True,
        "smart_message_on_start": True,
        "bot_token": "8367220857:AAHgvPb1pmAqHSwgixb9jBYCT2TTRrDnNL0",
        "chat_id": "1266351161"
    },
    "okx": {
        "api_key": "29809262-8962-4460-b7a0-280131629aea",
        "secret_key": "1EBB409F0B37C9CB936FD6BD510A6C00",
        "passphrase": "Q@BWaG2bf5ybmGZ"
    },
    "reporting": {
        "hourly": True,
        "hour_interval_sec": 3600
    }
}
# ---------------------------------------------------

# --- Friendly imports with hints ---
def _need(mod, pip_name=None):
    if pip_name is None: pip_name = mod
    print(f"[!] Missing dependency: {mod}. Install with:  py -3.11 -m pip install {pip_name}")
    sys.exit(1)

try:
    import requests
except ImportError:
    _need("requests")

try:
    import numpy as np
except ImportError:
    _need("numpy")

try:
    import pandas as pd
except ImportError:
    _need("pandas")

try:
    import ccxt
except ImportError:
    _need("ccxt")

# sklearn & joblib are optional at runtime; we'll handle graceful fallback
try:
    import joblib
except Exception:
    joblib = None

try:
    from sklearn.ensemble import RandomForestClassifier
except Exception:
    RandomForestClassifier = None

# ------------- Logger -------------
import logging
def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        h = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        h.setFormatter(fmt); logger.addHandler(h)
    return logger

log = get_logger("okx_ai_bot")

def now_utc():
    return datetime.now(timezone.utc)

# ------------- Telegram -------------
_default_bot=None; _default_chat=None
def configure_telegram(bot_token=None, chat_id=None):
    global _default_bot,_default_chat
    _default_bot = bot_token or _default_bot
    _default_chat = str(chat_id) if chat_id is not None else _default_chat

def send_telegram(text, bot_token=None, chat_id=None):
    bt = bot_token or _default_bot or CONFIG["telegram"].get("bot_token")
    ci = str(chat_id or _default_chat or CONFIG["telegram"].get("chat_id") or "").strip()
    if not CONFIG["telegram"].get("enabled", True): return False, "Telegram disabled"
    if not bt or not ci: return False, "Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID"
    url = f"https://api.telegram.org/bot{bt}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": ci, "text": text, "parse_mode": "Markdown"})
        return r.ok, r.text
    except Exception as e:
        return False, str(e)

# ------------- Features -------------
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def build_features_from_df(df, return_df=False):
    df = df.copy()
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
        df = df.assign(timestamp=ts).dropna(subset=["timestamp"])
        df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    df["ret_1"] = df["close"].pct_change()
    df["log_ret"] = np.log1p(df["ret_1"])
    df["rsi14"] = rsi(df["close"], 14)
    df["ema_fast"] = ema(df["close"], 12)
    df["ema_slow"] = ema(df["close"], 26)
    macd_line, signal_line, hist = macd(df["close"])
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = hist
    df["atr14"] = atr(df["high"], df["low"], df["close"], 14)
    df = df.dropna().copy()

    features = ["ret_1","log_ret","rsi14","ema_fast","ema_slow","macd","macd_signal","macd_hist","atr14"]
    X_last = df[features].iloc[-1].values
    meta = {"features": features, "n_rows": len(df)}
    if return_df:
        return X_last, meta, df
    return X_last, meta

def build_features_from_csv(csv_path, lookback=100, return_df=False):
    df = pd.read_csv(csv_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
        df.set_index("timestamp", inplace=True)
    else:
        raise RuntimeError("timestamp column missing in downloaded data")
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    df["ret_1"] = df["close"].pct_change()
    df["log_ret"] = np.log1p(df["ret_1"])
    df["rsi14"] = rsi(df["close"], 14)
    df["ema_fast"] = ema(df["close"], 12)
    df["ema_slow"] = ema(df["close"], 26)
    macd_line, signal_line, hist = macd(df["close"])
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = hist
    df["atr14"] = atr(df["high"], df["low"], df["close"], 14)

    horizon = 3
    future_ret = df["close"].shift(-horizon) / df["close"] - 1.0
    threshold = 0.0015
    df["y_up"] = (future_ret > threshold).astype(int)

    features = ["ret_1","log_ret","rsi14","ema_fast","ema_slow","macd","macd_signal","macd_hist","atr14"]
    df = df.dropna().copy()

    X = df[features].values
    y = df["y_up"].values
    meta = {"n_rows": len(df), "features": features, "lookback": lookback}
    if return_df:
        return X, y, meta, df
    return X, y, meta

# ------------- Model -------------
class HeuristicModel:
    """Fallback model if scikit-learn/joblib not available or model file missing."""
    def proba_up(self, X_row):
        # X_row order: ret_1, log_ret, rsi14, ema_fast, ema_slow, macd, macd_signal, macd_hist, atr14
        rsi14 = float(X_row[2])
        macd_hist = float(X_row[8 - 1])  # index 7
        score = 0.0
        if rsi14 < 35: score += 0.15
        if macd_hist > 0: score += 0.25
        score += max(0.0, min(0.2, float(X_row[0])*5))  # ret_1 small positive
        return max(0.05, min(0.95, 0.5 + score - 0.1))

class SignalModel:
    def __init__(self, clf=None):
        self.clf = clf
    def proba_up(self, X_row):
        if self.clf is None:
            return HeuristicModel().proba_up(X_row)
        proba = self.clf.predict_proba([X_row])[0][1]
        return float(proba)
    def save(self, path):
        if joblib is None: raise RuntimeError("joblib not installed")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.clf, path)
    @staticmethod
    def load(path):
        if joblib is None: return SignalModel(None)
        if not Path(path).exists(): return SignalModel(None)
        try:
            clf = joblib.load(path)
            return SignalModel(clf)
        except Exception:
            return SignalModel(None)

class Trainer:
    def __init__(self, cfg): self.cfg = cfg
    def fit(self, X, y):
        if RandomForestClassifier is None:
            raise RuntimeError("scikit-learn not installed. Install with: py -3.11 -m pip install scikit-learn joblib")
        self.clf = RandomForestClassifier(n_estimators=250, min_samples_leaf=3, n_jobs=-1, random_state=42)
        self.clf.fit(X, y)
    def save(self, path): SignalModel(self.clf).save(path)

# ------------- Downloader -------------
class Downloader:
    def __init__(self, cfg):
        self.cfg = cfg
        ex_class = getattr(ccxt, cfg.get("exchange","okx"))
        self.exchange = ex_class({"enableRateLimit": True})
        if cfg.get("exchange")=="okx" and cfg.get("okx_demo", True):
            try: self.exchange.set_sandbox_mode(True)
            except Exception: pass
        self.symbol = cfg["symbol"]
        self.timeframe = cfg["timeframe"]
        self.limit = int(cfg.get("limit_bars", 5000))

    def download(self, since_days=120):
        from datetime import timedelta
        since = int((now_utc() - timedelta(days=since_days)).timestamp()*1000)
        all_candles = []
        while True:
            candles = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, since=since, limit=100)
            if not candles: break
            all_candles += candles; since = candles[-1][0] + 1
            if len(all_candles) >= self.limit: break
        df = pd.DataFrame(all_candles, columns=["timestamp","open","high","low","close","volume"])
        Path("data").mkdir(exist_ok=True, parents=True)
        out = Path(f"data/ohlcv_{self.symbol.replace('/','_')}_{self.timeframe}.csv")
        df.to_csv(out, index=False)
        log.info(f"Saved {len(df)} rows to {out}")

# ------------- Backtester -------------
class Backtester:
    def __init__(self, cfg, df, model):
        self.cfg=cfg; self.df=df.copy(); self.model=model
        self.fee=cfg.get("fee_rate",0.0005); self.slippage=cfg.get("slippage_bps",3)/10000.0
        self.buy_thr=cfg.get("proba_buy_threshold",0.60)
    def run(self):
        feats=["ret_1","log_ret","rsi14","ema_fast","ema_slow","macd","macd_signal","macd_hist","atr14"]
        cash=10000.0; pos=0.0; eq=[]
        for _,row in self.df.iterrows():
            p_up = self.model.proba_up(row[feats].values); price=row["close"]
            if pos==0 and p_up>=self.buy_thr:
                qty=cash/price; fill=price*(1+self.slippage); fee=qty*fill*self.fee; cash-=qty*fill+fee; pos=qty
            elif pos>0 and p_up<self.buy_thr:
                fill=price*(1-self.slippage); fee=pos*fill*self.fee; cash+=pos*fill-fee; pos=0.0
            eq.append(cash+pos*price)
        ret=(eq[-1]/eq[0])-1 if eq else 0; return {"final_equity":eq[-1] if eq else 10000,"return":ret,"n_points":len(eq)}

# ------------- Momentum Helper -------------
def momentum_ok_from_df(fdf, min_rsi=48, rising_bars=2):
    # يفترض fdf يحوي أعمدة: macd_hist, rsi14
    tail = fdf[["macd_hist", "rsi14"]].tail(rising_bars + 1)
    if len(tail) < rising_bars + 1:
        return False
    # ارتفاع متتالي في macd_hist
    diffs = tail["macd_hist"].diff().tail(rising_bars)
    rising = (diffs > 0).all()
    rsi_ok = float(tail["rsi14"].iloc[-1]) >= float(min_rsi)
    return bool(rising and rsi_ok)

# ------------- Trader -------------
class Trader:
    def __init__(self, cfg):
        self.cfg=cfg
        self.paper=cfg.get("paper_trading",True)
        self.execute=bool(cfg.get("execute_orders", False))
        if not cfg.get("okx_demo", True):
            self.execute=False
        self.timeframe=cfg["timeframe"]
        self.buy_thr=cfg.get("proba_buy_threshold",0.6)
        self.risk_cfg=cfg.get("risk",{})
        self.capital_pct=cfg.get("capital_pct",0.85)
        self.leverage=cfg.get("leverage",10)
        self.top_scan=cfg.get("top_scan",{"enabled":False})
        self.reporting_cfg=cfg.get("reporting",{"hourly":True,"hour_interval_sec":3600})
        self.poll_sec=int(cfg.get("poll_sec",10))
        self.scan_eval_top=int(cfg.get("scan_eval_top",8))
        self.preentry_margin=float(cfg.get("preentry_margin",0.05))
        self.enable_momentum_gate=bool(cfg.get("enable_momentum_gate",True))
        self.min_rsi=float(cfg.get("min_rsi",48))
        self.macd_rising_bars=int(cfg.get("macd_rising_bars",2))

        # Telegram
        tel_cfg = cfg.get("telegram", {})
        if tel_cfg.get("enabled"):
            configure_telegram(tel_cfg.get("bot_token"), tel_cfg.get("chat_id"))

        # Exchange
        okx_cfg = cfg.get("okx", {})
        ex_id = cfg.get("exchange", "okx")
        ex_class = getattr(ccxt, ex_id)
        self.exchange = ex_class({
            "apiKey": okx_cfg.get("api_key"),
            "secret": okx_cfg.get("secret_key"),
            "password": okx_cfg.get("passphrase"),
            "enableRateLimit": True,
            "options": {},
        })
        if ex_id=="okx" and cfg.get("okx_demo", True):
            try:
                self.exchange.set_sandbox_mode(True); log.info("OKX sandbox mode enabled")
            except Exception as e:
                log.warning(f"Sandbox mode not supported: {e}")

        self.market_type=cfg.get("market_type","swap")
        self.td_mode=cfg.get("td_mode","isolated")

        self.model = SignalModel.load(cfg["model_path"])
        self.balance = float(cfg.get("starting_balance", 10000.0))
        self.session_start_equity = self.balance
        self.pos=None
        self.last_report_ts=None
        self.last_progress_step=None

        if tel_cfg.get("enabled") and tel_cfg.get("smart_message_on_start"):
            mode_msg = "EXECUTE: ON (Demo orders)" if self.execute and not self.paper else "PAPER: ON (Simulated)"
            send_telegram(
                f"🤖 *Smart Trading Bot* بدأ.\n"
                f"- وضع: {mode_msg}\n"
                f"- فحص Top20\n- إطار: {self.timeframe}\n- رافعة: x{self.leverage}\n"
                f"- استخدام رأس المال: {int(self.capital_pct*100)}%\n"
                f"- دخول سريع: {'ON' if self.enable_momentum_gate else 'OFF'} | preMargin={self.preentry_margin:.02f}"
            )

    def load_top_symbols(self):
        quote = self.top_scan.get("quote","USDT")
        n = int(self.top_scan.get("n",20))
        sort_by = self.top_scan.get("sort_by","volume")
        self.exchange.load_markets()
        symbols = []
        for sym, m in self.exchange.markets.items():
            if "/" not in sym:
                continue
            # Prefer ccxt's parsed quote, fall back to symbol parsing
            q = (m.get("quote") or sym.split("/")[-1]).split(":")[0]
            if q != quote:
                continue
            # Determine type via flags where possible
            is_spot = bool(m.get("spot")) or m.get("type") == "spot"
            is_swap = bool(m.get("swap")) or m.get("type") == "swap" or (":" in sym)
            is_perp = is_swap and ("-" not in sym) and (m.get("expiry") in (None, 0))
            if self.market_type == "swap" and not is_perp:
                continue
            if self.market_type == "spot" and not is_spot:
                continue
            symbols.append(sym)
        # Fallback: if empty on sandbox, seed with common USDT swaps
        if not symbols:
            seed = [
                "BTC/USDT:USDT","ETH/USDT:USDT","SOL/USDT:USDT","BNB/USDT:USDT","XRP/USDT:USDT",
                "ADA/USDT:USDT","DOGE/USDT:USDT","TRX/USDT:USDT","TON/USDT:USDT","SHIB/USDT:USDT",
                "DOT/USDT:USDT","AVAX/USDT:USDT","LINK/USDT:USDT","LTC/USDT:USDT","MATIC/USDT:USDT",
                "BCH/USDT:USDT","APT/USDT:USDT","OP/USDT:USDT","ARB/USDT:USDT","NEAR/USDT:USDT"
            ]
            # Keep only symbols that exist in markets
            symbols = [s for s in seed if s in self.exchange.markets] or seed[:n]
        vols = []
        try:
            tickers = self.exchange.fetch_tickers(symbols)
            for s, t in tickers.items():
                vol = t.get("quoteVolume") or t.get("baseVolume") or 0
                oi = t.get("openInterest") or 0
                metric = oi if sort_by.lower() == "openinterest" else vol
                vols.append((s, float(metric or 0.0)))
        except Exception:
            for s in symbols:
                m = self.exchange.markets.get(s, {})
                vol = 0.0
                info = m.get("info")
                if isinstance(info, dict):
                    vol = info.get("vol24h") or info.get("volCcy24h") or 0
                vols.append((s, float(vol or 0.0)))
        vols.sort(key=lambda x: x[1], reverse=True)
        symbols = [s for s, _ in vols[:n]]
        if symbols:
            symbols = symbols[: self.scan_eval_top]
        return symbols

    def fetch_df(self, symbol, limit=200):
        candles=self.exchange.fetch_ohlcv(symbol, timeframe=self.timeframe, limit=limit)
        return pd.DataFrame(candles, columns=["timestamp","open","high","low","close","volume"])

    def ai_tp_sl(self, p_up, atr_val):
        base_stop=float(self.risk_cfg.get("base_atr_stop_mult",1.8))
        base_tp=float(self.risk_cfg.get("base_atr_tp_mult",3.0))
        conf=max(0.0,min(1.0,(p_up-0.5)/0.5))
        stop_mult=base_stop*(1.0+0.8*conf); tp_mult=base_tp*(1.0+0.8*conf)
        return stop_mult*atr_val, tp_mult*atr_val

    def compute_qty(self, price):
        notional=self.balance*self.capital_pct*self.leverage
        qty=notional/price
        max_pct=float(self.risk_cfg.get("max_position_pct",0.9))
        max_notional=self.balance*max_pct*self.leverage
        return max(min(qty, max_notional/price), 0.0)

    def ensure_margin_leverage(self, symbol):
        try:
            if hasattr(self.exchange, "set_margin_mode"):
                self.exchange.set_margin_mode(self.td_mode, symbol, {"posSide":"long"})
        except Exception:
            pass
        try:
            if hasattr(self.exchange, "set_leverage"):
                self.exchange.set_leverage(self.leverage, symbol, {"mgnMode": self.td_mode, "posSide":"long"})
        except Exception:
            pass

    def compute_qty_contracts(self, symbol, price):
        market = self.exchange.market(symbol)
        contract_size = float(market.get("contractSize") or 1.0)
        notional = self.balance * self.capital_pct * self.leverage
        max_pct = float(self.risk_cfg.get("max_position_pct", 0.9))
        notional = min(notional, self.balance * max_pct * self.leverage)
        contracts = (notional / price) / max(contract_size, 1e-9)
        try:
            contracts = float(self.exchange.amount_to_precision(symbol, contracts))
        except Exception:
            contracts = float(int(max(1, contracts)))
        return max(1.0, contracts), contract_size, notional

    def place_open_market(self, symbol, contracts):
        params = {"tdMode": self.td_mode, "posSide":"long", "reduceOnly": False}
        return self.exchange.create_order(symbol, "market", "buy", contracts, None, params)

    def place_close_market(self, symbol, contracts):
        params = {"tdMode": self.td_mode, "posSide":"long", "reduceOnly": True}
        return self.exchange.create_order(symbol, "market", "sell", contracts, None, params)

    def evaluate_candidates(self, symbols):
        best=None
        for s in symbols:
            try:
                df=self.fetch_df(s, limit=150)
                X_last, meta, fdf=build_features_from_df(df, return_df=True)
                p_up=self.model.proba_up(X_last)
                last=fdf.iloc[-1]
                if best is None or p_up>best[1]:
                    best=(s, p_up, last, fdf)
            except Exception as e:
                log.warning(f"Eval failed for {s}: {e}")
        return best

    def open_position(self, symbol, price, atr, p_up):
        if self.pos is not None: return False
        stop_dist, tp_dist=self.ai_tp_sl(p_up, atr)
        stop_price=max(0.0, price-stop_dist); tp_price=price+tp_dist
        if self.execute and not self.paper:
            try:
                self.ensure_margin_leverage(symbol)
                contracts, csize, notional = self.compute_qty_contracts(symbol, price)
                order = self.place_open_market(symbol, contracts)
                fill = float(order.get("average") or order.get("price") or price)
                self.pos={"symbol":symbol,"contracts":contracts,"contractSize":csize,"entry":fill,"stop":stop_price,"tp":tp_price,"open_ts":now_utc(),"p_up":p_up,"atr":atr,"leverage":self.leverage,"pnl_realized":0.0,"side":"long"}
                send_telegram(f"🟢 *Demo دخول*\nزوج: `{symbol}`\nfill: `{fill:.4f}`\nوقف: `{stop_price:.4f}`\nهدف: `{tp_price:.4f}`\ncontracts: {contracts} | cSize: {csize}\nnotional: {notional:.2f}")
                return True
            except Exception as e:
                log.warning(f"Demo order failed: {e}, switching to PAPER")
                send_telegram(f"⚠️ فشل تنفيذ الأمر (Demo). التحول إلى PAPER. {e}")
                self.paper = True
                self.execute = False
        qty=self.compute_qty(price)
        self.pos={"symbol":symbol,"qty":qty,"entry":price,"stop":stop_price,"tp":tp_price,"open_ts":now_utc(),"p_up":p_up,"atr":atr,"leverage":self.leverage,"pnl_realized":0.0,"side":"long"}
        send_telegram(f"🧠 *Smart إشارة دخول*\nزوج: `{symbol}`\nاحتمال صعود: *{p_up:.2%}*\nدخول: `{price:.4f}`\nوقف: `{stop_price:.4f}`\nهدف: `{tp_price:.4f}`\nرافعة: x{self.leverage}\nحجم تقريبي: {qty:.4f}")
        return True

    def close_position(self, price, reason="signal"):
        if self.pos is None: return
        symbol=self.pos["symbol"]
        if "contracts" in self.pos:
            contracts=self.pos["contracts"]
            csize=self.pos.get("contractSize",1.0)
            entry=self.pos["entry"]
            fill=price
            if self.execute and not self.paper:
                try:
                    order=self.place_close_market(symbol, contracts)
                    fill=float(order.get("average") or order.get("price") or price)
                except Exception as e:
                    log.warning(f"Close order failed: {e}")
            pnl=(fill-entry)*(contracts*csize)*self.leverage
            self.balance+=pnl; self.pos["pnl_realized"]+=pnl
            status="ربح ✅" if pnl>=0 else "خسارة ❌"
            send_telegram(f"⛔ إغلاق الصفقة ({reason}) [Demo]\nزوج: `{symbol}`\nالدخول: `{entry:.4f}`\nالإغلاق: `{fill:.4f}`\nالنتيجة: *{status}*\nالربح/الخسارة: `{pnl:.2f}`\nرصيد الجلسة: `{self.balance:.2f}`")
        else:
            qty=self.pos["qty"]; entry=self.pos["entry"]
            pnl=(price-entry)*qty*self.leverage
            self.balance+=pnl; self.pos["pnl_realized"]+=pnl
            status="ربح ✅" if pnl>=0 else "خسارة ❌"
            send_telegram(f"⛔ إغلاق الصفقة ({reason})\nزوج: `{symbol}`\nالدخول: `{entry:.4f}`\nالإغلاق: `{price:.4f}`\nالنتيجة: *{status}*\nالربح/الخسارة: `{pnl:.2f}`\nرصيد الجلسة: `{self.balance:.2f}`")
        self.pos=None; self.last_progress_step=None

    def progress_update_if_needed(self, price):
        if self.pos is None: return
        change_pct=(price-self.pos["entry"])/self.pos["entry"]*100
        step=int(math.floor(change_pct))
        if self.last_progress_step is None or step!=self.last_progress_step:
            self.last_progress_step=step
            send_telegram(f"📈 تقدم الصفقة: {step:+d}%\nزوج: `{self.pos['symbol']}`\nالسعر الحالي: `{price:.4f}`\nالدخول: `{self.pos['entry']:.4f}`")

    def hourly_report_if_needed(self, price=None):
        if not self.reporting_cfg.get("hourly", True): return
        now = now_utc()
        if self.last_report_ts is None or (now - self.last_report_ts).total_seconds() >= int(self.reporting_cfg.get("hour_interval_sec", 3600)):
            unreal=0.0
            if self.pos is not None and price is not None:
                if "contracts" in self.pos:
                    unreal=(price-self.pos["entry"])*(self.pos["contracts"]*self.pos.get("contractSize",1.0))*self.pos["leverage"]
                else:
                    unreal=(price-self.pos["entry"])*self.pos["qty"]*self.pos["leverage"]
            total_pnl=(self.balance - float(self.cfg.get("starting_balance", 10000.0)))
            send_telegram(f"🕒 تقرير ساعة\nالربح/الخسارة الكلية منذ البداية: `{total_pnl:.2f}`\nالربح/الخسارة غير المحققة: `{unreal:.2f}`\nالرصيد التقريبي: `{self.balance + unreal:.2f}`")
            self.last_report_ts = now

    def run_loop(self):
        mode = "PAPER"
        if self.execute and not self.paper:
            mode = "DEMO EXEC"
        elif not self.paper:
            mode = "LIVE"
        log.info(f"Starting {mode} trading on OKX {self.market_type} @ {self.timeframe}")
        send_telegram(
            f"⚙️ وضع الدخول: Aggressive — tf={self.timeframe}, thr={self.buy_thr:.2f}, preMargin={self.preentry_margin:.2f}, poll={self.poll_sec}s"
        )
        symbols=None
        if self.top_scan.get("enabled"):
            symbols=self.load_top_symbols()
            log.info(f"Top symbols: {symbols}")
            send_telegram("🔎 فحص Top20 تم — عدد الأزواج: {}".format(len(symbols)))
        while True:
            try:
                if self.pos is None:
                    if self.top_scan.get("enabled"):
                        best=self.evaluate_candidates(symbols or [])
                        if best is not None:
                            sym, p_up, last, fdf = best
                            price=float(last["close"]); atr=float(last.get("atr14", 0.0))
                            early_allowed = p_up >= (self.buy_thr - self.preentry_margin)
                            ok_momentum = True
                            if self.enable_momentum_gate:
                                ok_momentum = momentum_ok_from_df(
                                    fdf, self.min_rsi, self.macd_rising_bars
                                )
                            if p_up >= self.buy_thr or (early_allowed and ok_momentum):
                                self.open_position(sym, price, atr, p_up)
                    else:
                        sym=self.cfg["symbol"]
                        df=self.fetch_df(sym, limit=150)
                        X_last, meta, fdf = build_features_from_df(df, return_df=True)
                        p_up=self.model.proba_up(X_last)
                        last=fdf.iloc[-1]
                        price=float(last["close"]); atr=float(last.get("atr14",0.0))
                        early_allowed = p_up >= (self.buy_thr - self.preentry_margin)
                        ok_momentum = True
                        if self.enable_momentum_gate:
                            ok_momentum = momentum_ok_from_df(
                                fdf, self.min_rsi, self.macd_rising_bars
                            )
                        if p_up >= self.buy_thr or (early_allowed and ok_momentum):
                            self.open_position(sym, price, atr, p_up)
                else:
                    sym=self.pos["symbol"]; df=self.fetch_df(sym, limit=5); price=float(df["close"].iloc[-1])
                    self.progress_update_if_needed(price)
                    if price>=self.pos["tp"]: self.close_position(price, reason="TP")
                    elif price<=self.pos["stop"]: self.close_position(price, reason="SL")
                if self.pos is None:
                    self.hourly_report_if_needed()
                else:
                    self.hourly_report_if_needed(price)
                time.sleep(self.poll_sec)
            except Exception as e:
                log.exception(f"Loop error: {e}"); time.sleep(self.poll_sec)

# ------------- Commands -------------
def cmd_download(args):
    Downloader(CONFIG).download(since_days=args.since_days)

def cmd_train(args):
    Path("data").mkdir(exist_ok=True, parents=True)
    csv_path = Path(f"data/ohlcv_{CONFIG['symbol'].replace('/','_')}_{CONFIG['timeframe']}.csv")
    if not csv_path.exists():
        log.info("No data found — downloading 120 days...")
        Downloader(CONFIG).download(since_days=120)
    X, y, meta = build_features_from_csv(csv_path, lookback=CONFIG["lookback"])
    tr = Trainer(CONFIG); tr.fit(X, y); tr.save(CONFIG["model_path"])
    log.info(f"Model trained & saved to {CONFIG['model_path']}. Meta: {meta}")

def cmd_backtest(args):
    csv_path = Path(f"data/ohlcv_{CONFIG['symbol'].replace('/','_')}_{CONFIG['timeframe']}.csv")
    if not csv_path.exists():
        log.info("No data found — downloading 120 days...")
        Downloader(CONFIG).download(since_days=120)
    X, y, meta, df = build_features_from_csv(csv_path, lookback=CONFIG['lookback'], return_df=True)
    model = SignalModel.load(CONFIG["model_path"])
    bt = Backtester(CONFIG, df, model)
    stats = bt.run()
    log.info(f"Backtest: {stats}")

def cmd_trade(args):
    # Auto-train if no model file and sklearn installed
    if not Path(CONFIG["model_path"]).exists():
        try:
            log.info("Model not found. Attempting quick train on recent data...")
            cmd_train(args)
        except Exception as e:
            log.warning(f"Auto-train failed ({e}). Will fallback to heuristic model.")
    Trader(CONFIG).run_loop()

def main():
    p = argparse.ArgumentParser(description="OKX AI Trader (single-file)")
    sub = p.add_subparsers(dest="cmd")
    p_dl = sub.add_parser("download"); p_dl.add_argument("--since_days", type=int, default=120); p_dl.set_defaults(func=cmd_download)
    sub.add_parser("train").set_defaults(func=cmd_train)
    sub.add_parser("backtest").set_defaults(func=cmd_backtest)
    sub.add_parser("trade").set_defaults(func=cmd_trade)
    args = p.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        print("[i] No command given; defaulting to 'trade'. You can also use: download/train/backtest/trade")
        cmd_trade(args)

if __name__ == "__main__":
    main()
