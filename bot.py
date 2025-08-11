import time
import pandas as pd
import numpy as np
import datetime
import ccxt
import uuid
from decimal import Decimal, getcontext

getcontext().prec = 18

# ================== CONFIG ===================
SYMBOL = 'ETH/USDT:USDT'
TIMEFRAME = '15m'
ORDER_SIZE_ETH = 0.12
LEVERAGE = 10
VOLATILITY_THRESHOLD_PCT = Decimal('0.02')
FRESH_SIGNAL_MAX_PRICE_DEVIATION = 0.02
TIME_BASED_EXIT_HOURS = 12
SUPERTREND_PERIOD = 10
SUPERTREND_MULTIPLIER = 3.0
STOCH_K = 14
STOCH_D = 3
MIN_VOLUME_FACTOR = 0.2

exchange = ccxt.bingx({
    'apiKey': 'wGY6iowJ9qdr1idLbKOj81EGhhZe5O8dqqZlyBiSjiEZnuZUDULsAW30m4eFaZOu35n5zQktN7a01wKoeSg',
    'secret': 'tqxcIVDdDJm2GWjinyBJH4EbvJrjIuOVyi7mnKOzhXHquFPNcULqMAOvmSy0pyuoPOAyCzE2zudzEmlwnA',
    'enableRateLimit': True,
    'options': {
        'defaultType': 'swap',
    }
})

open_trades = {}

# ================== HELPERS ==================

def generate_client_order_id():
    return "ccbot-" + uuid.uuid4().hex[:16]

def fetch_ohlcv(symbol, timeframe, limit=200):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

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

def compute_stochastic(df, k_period=STOCH_K, d_period=STOCH_D):
    low_min = df['low'].rolling(window=k_period, min_periods=1).min()
    high_max = df['high'].rolling(window=k_period, min_periods=1).max()
    denom = (high_max - low_min).replace(0, np.nan)
    k = 100 * (df['close'] - low_min) / denom
    k = k.fillna(50)
    d = k.rolling(window=d_period, min_periods=1).mean()
    return k, d

def is_fresh_signal(df):
    if len(df) < 50:
        return None
    st_dir, atr = compute_supertrend(df)
    k, d = compute_stochastic(df)
    cross_up = (k.iloc[-2] < d.iloc[-2]) and (k.iloc[-1] > d.iloc[-1])
    cross_down = (k.iloc[-2] > d.iloc[-2]) and (k.iloc[-1] < d.iloc[-1])
    price = df['close'].iloc[-1]
    signal_price = df['close'].iloc[-2]
    deviation = abs(price - signal_price) / signal_price if signal_price != 0 else 1
    st_is_up = bool(st_dir.iloc[-1])
    signal = None
    if cross_up and st_is_up and deviation <= FRESH_SIGNAL_MAX_PRICE_DEVIATION:
        signal = 'buy'
    elif cross_down and (not st_is_up) and deviation <= FRESH_SIGNAL_MAX_PRICE_DEVIATION:
        signal = 'sell'
    return (signal, atr.iloc[-1]) if signal else None

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

def calculate_tp_sl(entry_price, atr, side):
    entry_price = float(entry_price)
    atr = float(atr)
    tp_multiplier = 1.5
    sl_multiplier = 1.5
    if side == 'buy':
        tp_price = entry_price + (atr * tp_multiplier)
        sl_price = entry_price - (atr * sl_multiplier)
    else:
        tp_price = entry_price - (atr * tp_multiplier)
        sl_price = entry_price + (atr * sl_multiplier)
    return round(tp_price, 8), round(sl_price, 8)

def place_order(symbol, side, entry_price, atr, qty_override=None):
    qty = float(qty_override) if qty_override else ORDER_SIZE_ETH
    leverage_side = 'LONG' if side == 'buy' else 'SHORT'
    try:
        exchange.set_leverage(LEVERAGE, symbol, params={'marginMode': 'cross', 'side': leverage_side})
    except Exception as e:
        print(f"[Leverage Warning] {e}")
    order_params = {'positionSide': leverage_side, 'newClientOrderId': generate_client_order_id()}
    try:
        exchange.create_order(symbol, 'market', side, qty, None, order_params)
    except Exception as e:
        print(f"[Order Error] {e}")
        return False
    tp_price, sl_price = calculate_tp_sl(entry_price, atr, side)
    try:
        exchange.create_order(symbol, 'take_profit_market', 'sell' if side == 'buy' else 'buy', qty, None, {
            'triggerPrice': tp_price,
            'stopPrice': tp_price,
            'positionSide': leverage_side,
            'newClientOrderId': generate_client_order_id()
        })
    except Exception as e:
        print(f"[TP Error] {e}")
    try:
        exchange.create_order(symbol, 'stop_market', 'sell' if side == 'buy' else 'buy', qty, None, {
            'triggerPrice': sl_price,
            'stopPrice': sl_price,
            'positionSide': leverage_side,
            'newClientOrderId': generate_client_order_id()
        })
    except Exception as e:
        print(f"[SL Error] {e}")
    open_trades[symbol] = {'side': side, 'entry_price': entry_price, 'qty_total': qty, 'tp_price': tp_price, 'sl_price': sl_price, 'atr': atr, 'entry_time': time.time(), 'leverage_side': leverage_side}
    return True

def signal_strength_and_execute(df):
    base = is_fresh_signal(df)
    if not base:
        return False
    signal, atr = base
    vol_now = df['volume'].iloc[-1]
    vol_avg = df['volume'].rolling(window=20, min_periods=1).mean().iloc[-1]
    if vol_now < (MIN_VOLUME_FACTOR * vol_avg):
        return False
    if in_position(SYMBOL):
        print("[Position] Already in position, skipping new entry.")
        return False
    price = df['close'].iloc[-1]
    return place_order(SYMBOL, signal, price, atr, qty_override=ORDER_SIZE_ETH)

if __name__ == '__main__':
    print("🚀 Trading bot started — ATR-based TP/SL, single position logic")
    while True:
        try:
            df = fetch_ohlcv(SYMBOL, TIMEFRAME)
            signal_strength_and_execute(df)
        except Exception as e:
            print(f"[Main loop error] {e}")
        time.sleep(20)





