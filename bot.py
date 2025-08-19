import time
import pandas as pd
import numpy as np
import datetime
import ccxt
import uuid
from decimal import Decimal, getcontext

getcontext().prec = 18

# ================== CONFIG ===================
SYMBOL = 'ETH/USDT:USDT'  # only trade ETH/USDT
TIMEFRAME = '15m'
ORDER_SIZE_ETH = 0.06

LEVERAGE = 18
COOLDOWN_PERIOD = 60
VOLATILITY_THRESHOLD_PCT = Decimal('0.02')
FRESH_SIGNAL_MAX_PRICE_DEVIATION = 0.1
PARTIAL_TP_RATIO = Decimal('0.5')
TRAIL_ATR_MULTIPLIER = Decimal('1.2')
TIME_BASED_EXIT_HOURS = 12
MAX_DAILY_LOSS_PCT = Decimal('0.05')
SUPERTREND_PERIOD = 10
SUPERTREND_MULTIPLIER = 3.0
# --- Replaced Stochastic with MACD + ADX ---
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ADX_PERIOD = 14
ADX_THRESHOLD = 20.0
MIN_VOLUME_FACTOR = 0.2
STRONG_TREND_ATR_MULT = 1.8
REENTRY_MIN_DELAY = 300
ATR_CHOPPY_THRESHOLD = Decimal('0.15')  # New filter for choppy/weak markets

exchange = ccxt.bingx({
    'apiKey': 'wGY6iowJ9qdr1idLbKOj81EGhhZe5O8dqqZlyBiSjiEZnuZUDULsAW30m4eFaZOu35n5zQktN7a01wKoeSg',
    'secret': 'tqxcIVDdDJm2GWjinyBJH4EbvJrjIuOVyi7mnKOzhXHquFPNcULqMAOvmSy0pyuoPOAyCzE2zudzEmlwnA',
    'enableRateLimit': True,
    'options': {
        'defaultType': 'swap',
    }
})

last_trade_time = {SYMBOL: 0}
open_trades = {}
daily_pnl = Decimal('0')
daily_loss_stop = False
current_day = datetime.date.today()
last_signal_dir = None

# ================== HELPERS ==================

def generate_client_order_id():
    return "ccbot-" + uuid.uuid4().hex[:16]

def fetch_ohlcv(symbol, timeframe, limit=200):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

# ================== INDICATORS ==================

def compute_atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    return atr

def compute_supertrend(df, period=SUPERTREND_PERIOD, multiplier=SUPERTREND_MULTIPLIER):
    atr = compute_atr(df, period)
    hl2 = (df['high'] + df['low']) / 2
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)

    final_upper = upperband.copy()
    final_lower = lowerband.copy()
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=bool)

    for i in range(len(df)):
        if i == 0:
            final_upper.iat[i] = upperband.iat[i]
            final_lower.iat[i] = lowerband.iat[i]
            supertrend.iat[i] = final_upper.iat[i]
            direction.iat[i] = False
            continue

        final_upper.iat[i] = upperband.iat[i] if (upperband.iat[i] < final_upper.iat[i-1] or df['close'].iat[i-1] > final_upper.iat[i-1]) else final_upper.iat[i-1]
        final_lower.iat[i] = lowerband.iat[i] if (lowerband.iat[i] > final_lower.iat[i-1] or df['close'].iat[i-1] < final_lower.iat[i-1]) else final_lower.iat[i-1]

        if supertrend.iat[i-1] == final_upper.iat[i-1] and df['close'].iat[i] <= final_upper.iat[i]:
            supertrend.iat[i] = final_upper.iat[i]
            direction.iat[i] = False
        elif supertrend.iat[i-1] == final_upper.iat[i-1] and df['close'].iat[i] > final_upper.iat[i]:
            supertrend.iat[i] = final_lower.iat[i]
            direction.iat[i] = True
        elif supertrend.iat[i-1] == final_lower.iat[i-1] and df['close'].iat[i] >= final_lower.iat[i]:
            supertrend.iat[i] = final_lower.iat[i]
            direction.iat[i] = True
        elif supertrend.iat[i-1] == final_lower.iat[i-1] and df['close'].iat[i] < final_lower.iat[i]:
            supertrend.iat[i] = final_upper.iat[i]
            direction.iat[i] = False
        else:
            direction.iat[i] = True if df['close'].iat[i] > final_lower.iat[i] else False
            supertrend.iat[i] = final_lower.iat[i] if direction.iat[i] else final_upper.iat[i]

    return direction, atr

# --- New: MACD ---

def compute_macd(df, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    close = df['close']
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist

# --- New: ADX ---

def compute_adx(df, period=ADX_PERIOD):
    high = df['high']
    low = df['low']
    close = df['close']

    plus_dm = (high.diff())
    minus_dm = (-low.diff())
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr1 = (high - low)
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period, min_periods=1).sum()
    plus_di = 100 * (plus_dm.rolling(window=period, min_periods=1).sum() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(window=period, min_periods=1).sum() / atr.replace(0, np.nan))

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    adx = dx.rolling(window=period, min_periods=1).mean()
    return adx, plus_di, minus_di

# ================== SIGNALS ==================

def is_fresh_or_strong_signal(df):
    global last_signal_dir, last_trade_time

    if len(df) < 50:
        print("📉 Not enough data to generate signals.")
        return None

    st_dir, atr = compute_supertrend(df)
    macd_line, macd_signal, macd_hist = compute_macd(df)
    adx, plus_di, minus_di = compute_adx(df)

    # MACD cross
    cross_up = (macd_line.iloc[-2] < macd_signal.iloc[-2]) and (macd_line.iloc[-1] > macd_signal.iloc[-1])
    cross_down = (macd_line.iloc[-2] > macd_signal.iloc[-2]) and (macd_line.iloc[-1] < macd_signal.iloc[-1])

    price = df['close'].iloc[-1]
    signal_price = df['close'].iloc[-2]
    deviation = abs(price - signal_price) / signal_price if signal_price != 0 else 1

    atr_latest = atr.iloc[-1]
    atr_pct = Decimal(str(atr_latest / price * 100)) if price != 0 else Decimal('0')

    # Avoid weak/choppy markets (ATR) and require trend strength via ADX
    if atr_pct < ATR_CHOPPY_THRESHOLD:
        print(f"⚠️ ATR {atr_pct:.2f}% < {ATR_CHOPPY_THRESHOLD}% — skipping trade" + signal)
        return None

    current_adx = float(adx.iloc[-1])
    if current_adx < ADX_THRESHOLD:
        print(f"⚠️ ADX {current_adx:.1f} < {ADX_THRESHOLD} — skipping trade" + signal)
        return None

    try:
        st_is_up = bool(st_dir.iloc[-1])


    except Exception:
        st_is_up = False

    signal = None
    if cross_up and st_is_up and deviation <= FRESH_SIGNAL_MAX_PRICE_DEVIATION:
        signal = 'buy'
    elif cross_down and (not st_is_up) and deviation <= FRESH_SIGNAL_MAX_PRICE_DEVIATION:
        signal = 'sell'
    else:
        price_change = (price - signal_price) / signal_price if signal_price != 0 else 0
        if st_is_up and price_change > 0.003 and macd_hist.iloc[-1] > 0:
            signal = 'buy'
        if (not st_is_up) and price_change < -0.003 and macd_hist.iloc[-1] < 0:
            signal = 'sell'

    if not signal and last_signal_dir is not None:
        if st_is_up and last_signal_dir == 'buy':
            if price > open_trades.get(SYMBOL, {}).get('entry_price', 0) + (atr_latest * STRONG_TREND_ATR_MULT):
                if time.time() - last_trade_time[SYMBOL] > REENTRY_MIN_DELAY and macd_hist.iloc[-1] > 0:
                    print("⚡ Strong BUY trend — re-entry allowed")
                    signal = 'buy'
        elif (not st_is_up) and last_signal_dir == 'sell':
            if price < open_trades.get(SYMBOL, {}).get('entry_price', 1e9) - (atr_latest * STRONG_TREND_ATR_MULT):
                if time.time() - last_trade_time[SYMBOL] > REENTRY_MIN_DELAY and macd_hist.iloc[-1] < 0:
                    print("⚡ Strong SELL trend — re-entry allowed")
                    signal = 'sell'

    if not signal:
        print("🚫 No valid signal.")
        return None

    last_signal_dir = signal
    return (signal, atr_latest)

# ================== POSITION HELPERS ==================

def in_position(symbol):
    try:
        positions = exchange.fetch_positions([symbol])
        for pos in positions:
            contracts = pos.get('contracts') or pos.get('size') or pos.get('positionAmt') or 0
            if float(contracts) != 0:
                return True
    except Exception as e:
        print(f"[in_position warning] {e}")
    return False

# ================== EXECUTION ==================

def signal_strength_and_execute(df):
    base = is_fresh_or_strong_signal(df)
    if not base:
        return False
    signal, atr = base

    vol_now = df['volume'].iloc[-1]
    vol_avg = df['volume'].rolling(window=20, min_periods=1).mean().iloc[-1]
    if vol_now < (MIN_VOLUME_FACTOR * vol_avg):
        print("[Volume] Below threshold, skipping  " + signal)
        return False

    # Scoring logic preserved; replaced Stochastic magnitude with MACD histogram magnitude
    score = 0.4
    macd_line, macd_signal, macd_hist = compute_macd(df)
    cross_magnitude = abs(macd_line.iloc[-1] - macd_signal.iloc[-1])  # unbounded; capped below
    score += min(0.4, float(min(1.0, cross_magnitude) * 0.4))
    vol_factor = min(1.0, float(vol_now / (vol_avg + 1e-9)))
    score += 0.2 * vol_factor

    if score < 0.25:
        print("[Score] Too low, skipping  "+ signal)
        return False

    if in_position(SYMBOL):
        print("[Position] Active position exists, skipping new trade  "+ signal)
        return False

    trade_meta = place_order(SYMBOL, signal, df['close'].iloc[-1], atr, qty_override=ORDER_SIZE_ETH)
    if trade_meta:
        print(f"✅ {signal.upper()} {SYMBOL} placed")
        return True
    return False

def calculate_tp_sl(entry_price, atr, side):
    entry_price = float(entry_price)
    atr = float(atr)
    tp_price = entry_price + (atr * 1.5 if side == 'buy' else -atr * 1.5)
    sl_price = entry_price - (atr * 1.8 if side == 'buy' else -atr * 1.8)
    return float(round(tp_price, 8)), float(round(sl_price, 8))

def place_order(symbol, side, entry_price, atr, qty_override=None):
    qty = float(qty_override) if qty_override is not None else float(ORDER_SIZE_ETH)
    leverage_side = 'LONG' if side == 'buy' else 'SHORT'

    try:
        exchange.set_leverage(LEVERAGE, symbol, params={'marginMode': 'cross', 'side': leverage_side})
    except Exception as e:
        print(f"[Leverage Warning] {e}")

    order_params = {'positionSide': leverage_side, 'newClientOrderId': generate_client_order_id()}

    try:
        order = exchange.create_order(symbol, 'market', side, qty, None, order_params)
    except ccxt.InsufficientFunds as e:
        print(f"[FAILURE] {e}")
        return False
    except Exception as e:
        print(f"[Order Error] {e}")
        return False

    tp_price, sl_price = calculate_tp_sl(entry_price, atr, side)
    partial_tp_price = entry_price + (float(atr) * 0.8 if side == 'buy' else -float(atr) * 0.8)
    partial_qty = qty / 2

    try:
        exchange.create_order(symbol, 'take_profit_market', 'sell' if side == 'buy' else 'buy', partial_qty, None, {
            'triggerPrice': partial_tp_price,
            'stopPrice': partial_tp_price,
            'positionSide': leverage_side,
            'newClientOrderId': generate_client_order_id()
        })
    except Exception as e:
        print(f"[Partial TP Error] {e}")

    try:
        exchange.create_order(symbol, 'take_profit_market', 'sell' if side == 'buy' else 'buy', qty - partial_qty, None, {
            'triggerPrice': tp_price,
            'stopPrice': tp_price,
            'positionSide': leverage_side,
            'newClientOrderId': generate_client_order_id()
        })
    except Exception as e:
        print(f"[Full TP Error] {e}")

    try:
        exchange.create_order(symbol, 'stop_market', 'sell' if side == 'buy' else 'buy', qty, None, {
            'triggerPrice': sl_price,
            'stopPrice': sl_price,
            'positionSide': leverage_side,
            'newClientOrderId': generate_client_order_id()
        })
    except Exception as e:
        print(f"[SL Error] {e}")

    open_trades[symbol] = {
        'side': side,
        'entry_price': float(entry_price),
        'qty_total': qty,
        'tp1_price': partial_tp_price,
        'tp2_price': tp_price,
        'sl_price': sl_price,
        'atr': float(atr),
        'entry_time': time.time(),
        'leverage_side': leverage_side
    }

    last_trade_time[symbol] = time.time()
    return open_trades[symbol]

# ================== MAIN ==================
if __name__ == '__main__':
    print("🚀 Trading bot started")
    while True:
        try:
            today = datetime.date.today()
            if today != current_day:
                daily_pnl = Decimal('0')
                daily_loss_stop = False
                current_day = today

            if daily_loss_stop:
                print("[Daily] Loss limit hit")
            else:
                df = fetch_ohlcv(SYMBOL, TIMEFRAME)
                try:
                    signal_strength_and_execute(df)
                except Exception as e:
                    print(f"[Signal Error] {e}")
        except Exception as e:
            print(f"[Main Error] {e}")
        time.sleep(20)

        time.sleep(20)










