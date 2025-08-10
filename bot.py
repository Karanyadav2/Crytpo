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
ORDER_SIZE_ETH = 0.1  # fixed trade size in ETH
LEVERAGE = 15
COOLDOWN_PERIOD = 60 * 30  # 30 minutes
VOLATILITY_THRESHOLD_PCT = Decimal('0.08')  # minimum ATR% to allow trading
FRESH_SIGNAL_MAX_PRICE_DEVIATION = 0.006
PARTIAL_TP_RATIO = Decimal('0.5')  # take 50% at first TP
TRAIL_ATR_MULTIPLIER = Decimal('1.5')
TIME_BASED_EXIT_HOURS = 6
MAX_DAILY_LOSS_PCT = Decimal('0.03')

# Exchange (set your API keys)
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

# ================== HELPERS ==================

def generate_client_order_id():
    return "ccbot-" + uuid.uuid4().hex[:16]


def fetch_ohlcv(symbol, timeframe, limit=150):
    print(f"📈 Fetching OHLCV for {symbol} timeframe={timeframe} limit={limit}...")
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


def compute_supertrend(df, period=10, multiplier=3):
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


def compute_stochastic(df, k_period=14, d_period=3):
    low_min = df['low'].rolling(window=k_period, min_periods=1).min()
    high_max = df['high'].rolling(window=k_period, min_periods=1).max()
    denom = (high_max - low_min).replace(0, np.nan)
    k = 100 * (df['close'] - low_min) / denom
    k = k.fillna(50)
    d = k.rolling(window=d_period, min_periods=1).mean()
    return k, d


# ================== TP/SL CALC ==================

def calculate_tp_sl(entry_price, atr, side):
    price = Decimal(str(entry_price))
    atr_dec = Decimal(str(atr))
    # use simple multiples for TP/SL
    sl_mult = Decimal('1.5')
    tp_mult = Decimal('3.0')

    sl_distance = (atr_dec * sl_mult)
    tp_distance = (atr_dec * tp_mult)

    if side == 'buy':
        sl_price = float((price - sl_distance).quantize(Decimal('0.01')))
        tp_price = float((price + tp_distance).quantize(Decimal('0.01')))
    else:
        sl_price = float((price + sl_distance).quantize(Decimal('0.01')))
        tp_price = float((price - tp_distance).quantize(Decimal('0.01')))

    print(f"[TP/SL] TP={tp_price}, SL={sl_price}")
    return tp_price, sl_price


# ================== SIGNALS ==================

def is_fresh_signal(df):
    if len(df) < 50:
        print("📉 Not enough data to generate signals.")
        return None

    st_dir, atr = compute_supertrend(df)
    k, d = compute_stochastic(df)

    cross_up = (k.iloc[-2] < d.iloc[-2]) and (k.iloc[-1] > d.iloc[-1])
    cross_down = (k.iloc[-2] > d.iloc[-2]) and (k.iloc[-1] < d.iloc[-1])

    price = df['close'].iloc[-1]
    signal_price = df['close'].iloc[-2]
    deviation = abs(price - signal_price) / signal_price if signal_price != 0 else 1

    atr_latest = atr.iloc[-1]
    atr_pct = Decimal(str(atr_latest / price * 100)) if price != 0 else Decimal('0')

    print(f"[DEBUG] ATR (price): {atr_latest:.6f}, ATR%: {atr_pct:.6f}")
    if atr_pct < VOLATILITY_THRESHOLD_PCT:
        print("🔇 Skipping due to low volatility (ATR% below threshold)")
        return None

    print(f"[DEBUG] Stochastic: K={k.iloc[-1]:.2f}, D={d.iloc[-1]:.2f}, cross_up={cross_up}, cross_down={cross_down}")
    print(f"[DEBUG] Price deviation: {deviation:.6f}")

    signal = None
    try:
        st_is_up = bool(st_dir.iloc[-1])
    except Exception:
        st_is_up = False

    if cross_up and st_is_up and deviation <= FRESH_SIGNAL_MAX_PRICE_DEVIATION:
        signal = 'buy'
    elif cross_down and (not st_is_up) and deviation <= FRESH_SIGNAL_MAX_PRICE_DEVIATION:
        signal = 'sell'

    if not signal:
        print("🚫 Conditions not met for signal.")
        return None

    # Freshness check (ensures the cross happened recently)
    age_candles = 1
    if age_candles > 1:
        print("⌛ Signal too old")
        return None

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
    return symbol in open_trades


def close_position_market(symbol, reason='manual'):
    try:
        if symbol not in open_trades:
            print(f"[close] No open trade metadata for {symbol}")
            return False
        meta = open_trades[symbol]
        side = meta['side']
        qty = meta['qty_total']
        close_side = 'sell' if side == 'buy' else 'buy'
        order = exchange.create_order(symbol, 'market', close_side, qty, None, {'positionSide': meta.get('leverage_side'), 'newClientOrderId': generate_client_order_id()})
        print(f"[close] Market close order placed: {order}")
        try:
            fill_price = float(order.get('average', order.get('price', meta['entry_price'])))
            pnl_per_unit = (fill_price - meta['entry_price']) if side == 'buy' else (meta['entry_price'] - fill_price)
            pnl = Decimal(str(pnl_per_unit)) * Decimal(str(qty))
            global daily_pnl
            daily_pnl += pnl
            print(f"[close] PnL estimated: {pnl}")
        except Exception:
            pass
        del open_trades[symbol]
        return True
    except Exception as e:
        print(f"[close error] {e}")
        return False


def monitor_open_trades():
    for symbol, meta in list(open_trades.items()):
        try:
            df_15 = fetch_ohlcv(symbol, TIMEFRAME, limit=50)
            st_dir_15, atr_15 = compute_supertrend(df_15)
            st_is_up_15 = bool(st_dir_15.iloc[-1])

            side = meta['side']
            entry_price = meta['entry_price']
            current_price = df_15['close'].iloc[-1]
            atr_now = float(atr_15.iloc[-1])

            # Supertrend flip exit
            if side == 'buy' and (not st_is_up_15):
                print(f"[Exit] Supertrend flipped against BUY on {symbol}. Closing position.")
                close_position_market(symbol, reason='supertrend_flip')
                continue
            if side == 'sell' and st_is_up_15:
                print(f"[Exit] Supertrend flipped against SELL on {symbol}. Closing position.")
                close_position_market(symbol, reason='supertrend_flip')
                continue

            # Time-based exit
            age_seconds = time.time() - meta['entry_time']
            if age_seconds > TIME_BASED_EXIT_HOURS * 3600:
                print(f"[Exit] Time-based exit hit for {symbol}. Closing position.")
                close_position_market(symbol, reason='time_exit')
                continue

            # Trailing stop: move SL to breakeven after +1 ATR and trail by TRAIL_ATR_MULTIPLIER
            profit_move = (current_price - entry_price) if side == 'buy' else (entry_price - current_price)
            if profit_move > 0 and abs(profit_move) >= atr_now:
                if side == 'buy':
                    new_sl = current_price - float(Decimal(str(atr_now)) * TRAIL_ATR_MULTIPLIER)
                else:
                    new_sl = current_price + float(Decimal(str(atr_now)) * TRAIL_ATR_MULTIPLIER)
                try:
                    remaining_qty = meta.get('qty_rest', meta.get('qty_total'))
                    exchange.create_order(symbol, 'stop_market', 'sell' if side == 'buy' else 'buy', remaining_qty, None, {
                        'triggerPrice': new_sl,
                        'positionSide': meta.get('leverage_side'),
                        'newClientOrderId': generate_client_order_id(),
                        'stopPrice': new_sl
                    })
                    meta['sl_price'] = new_sl
                    print(f"[Trail] Updated SL for {symbol} to {new_sl}")
                except Exception as e:
                    print(f"[Trail Error] {e}")

        except Exception as e:
            print(f"[monitor error] {symbol}: {e}")


# ================== SIGNAL STRENGTH & EXECUTION ==================

def signal_strength_and_execute(df):
    base = is_fresh_signal(df)
    if not base:
        return False
    signal, atr = base

    # Volume confirmation
    vol_now = df['volume'].iloc[-1]
    vol_avg = df['volume'].rolling(window=20, min_periods=1).mean().iloc[-1]
    if vol_now < vol_avg:
        print("[Volume] Current volume below 20-period average. Skipping.")
        return False

    # Strength score (supertrend alignment + stochastic cross magnitude + volume)
    score = 0.6
    k, d = compute_stochastic(df)
    cross_magnitude = abs(k.iloc[-1] - d.iloc[-1]) / 100.0
    score += min(0.3, float(cross_magnitude * 0.3))
    vol_factor = min(1.0, float(vol_now / (vol_avg + 1e-9)))
    score += 0.1 * vol_factor

    print(f"[Signal Score] {score:.3f}")
    if score < 0.35:
        print("[Score] Signal strength below threshold. Skipping.")
        return False

    # final ATR% check
    price = df['close'].iloc[-1]
    atr_val = compute_atr(df).iloc[-1]
    atr_pct = Decimal(str(atr_val / price * 100)) if price != 0 else Decimal('0')
    if atr_pct < VOLATILITY_THRESHOLD_PCT:
        print("🔇 Skipping due to low volatility (post-check)")
        return False

    # place fixed-size order (0.1 ETH)
    trade_meta = place_order(SYMBOL, signal, price, atr_val, qty_override=ORDER_SIZE_BY_SYMBOL.get(SYMBOL, 0.1))
    if trade_meta:
        print(f"✅ {signal.upper()} {SYMBOL} placed")
        return True
    return False


# ================== ORDER EXECUTION ==================

def place_order(symbol, side, entry_price, atr, qty_override=None):
    print(f"🛒 Placing {side.upper()} order on {symbol}...")
    try:
        entry_price = float(entry_price)
        atr = float(atr)
    except Exception as e:
        print(f"[Qty/ATR Error] {e}")
        return False

    qty = float(qty_override) if qty_override is not None else float(ORDER_SIZE_BY_SYMBOL.get(symbol, 0.1))
    # force fixed 0.1 ETH if symbol matches
    if symbol == SYMBOL:
        qty = 0.1

    print(f"[DEBUG] Qty: {qty}")

    try:
        if hasattr(exchange, 'set_position_mode'):
            try:
                exchange.set_position_mode(True)
            except Exception as e:
                print(f"[Mode Warning] Could not set position mode: {e}")
    except Exception:
        pass

    try:
        leverage_side = 'LONG' if side == 'buy' else 'SHORT'
        try:
            exchange.set_leverage(LEVERAGE, symbol, params={'marginMode': 'cross', 'side': leverage_side})
        except Exception as e:
            print(f"[Leverage Warning] set_leverage failed or unsupported: {e}")
    except Exception as e:
        print(f"[Leverage Error] {e}")

    order_params = {
        'positionSide': leverage_side,
        'newClientOrderId': generate_client_order_id()
    }

    try:
        order = exchange.create_order(symbol, 'market', side, qty, None, order_params)
        print(f"[Order] Market order placed: {order}")
    except ccxt.InsufficientFunds as e:
        print(f"[FAILURE] Order rejected: {str(e)}")
        return False
    except Exception as e:
        print(f"[Order Error] {e}")
        return False

    tp_price, sl_price = calculate_tp_sl(entry_price, atr, side)

    try:
        qty_total = qty
        qty_first = float(Decimal(str(qty_total)) * PARTIAL_TP_RATIO)
        qty_rest = float(Decimal(str(qty_total)) - Decimal(str(qty_first)))
    except Exception:
        qty_first = qty * float(PARTIAL_TP_RATIO)
        qty_rest = qty - qty_first

    try:
        tp_order_1 = exchange.create_order(symbol, 'limit', 'sell' if side == 'buy' else 'buy', qty_first, tp_price, {
            'positionSide': leverage_side,
            'newClientOrderId': generate_client_order_id()
        })
        print(f"[TP1 Order] Created for {qty_first} at {tp_price}")
    except Exception as e:
        print(f"[TP1 Error] {e}")

    try:
        sl_order = exchange.create_order(symbol, 'stop_market', 'sell' if side == 'buy' else 'buy', qty, None, {
            'triggerPrice': sl_price,
            'positionSide': leverage_side,
            'newClientOrderId': generate_client_order_id(),
            'stopPrice': sl_price
        })
        print(f"[SL Order] Created at {sl_price}")
    except Exception as e:
        print(f"[SL Error] {e}")

    open_trades[symbol] = {
        'side': side,
        'entry_price': float(entry_price),
        'qty_total': qty_total,
        'qty_first': qty_first,
        'qty_rest': qty_rest,
        'tp1_price': tp_price,
        'sl_price': sl_price,
        'atr': float(atr),
        'entry_time': time.time(),
        'leverage_side': leverage_side
    }

    last_trade_time[symbol] = time.time()
    return open_trades[symbol]


# ================== MAIN ==================
if __name__ == '__main__':
    print("🚀 Trading bot started with 15m Supertrend+Stochastic (fixed 0.1 ETH size)")
    while True:
        try:
            # reset daily trackers at midnight UTC
            today = datetime.date.today()
            if today != current_day:
                daily_pnl = Decimal('0')
                daily_loss_stop = False
                current_day = today

            if daily_loss_stop:
                print("[Daily] Trading suspended due to daily loss limit.")
            else:
                df = fetch_ohlcv(SYMBOL, TIMEFRAME)
                try:
                    trade_executed = signal_strength_and_execute(df)
                except Exception as e:
                    print(f"[signal exec error] {e}")

                # monitor open trades
                if open_trades:
                    monitor_open_trades()

        except Exception as e:
            print(f"[Main loop error] {e}")

        time.sleep(30)
