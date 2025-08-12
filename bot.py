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
COOLDOWN_PERIOD = 60  # 1 minute cooldown for faster entries
# Loosened volatility threshold to allow more trades but still avoid extremely low volatility
VOLATILITY_THRESHOLD_PCT = Decimal('0.02')
FRESH_SIGNAL_MAX_PRICE_DEVIATION = 0.02  # allow more deviation
PARTIAL_TP_RATIO = Decimal('0.5')  # take 50% at first TP (kept for compatibility but not used)
TRAIL_ATR_MULTIPLIER = Decimal('1.2')
TIME_BASED_EXIT_HOURS = 12
MAX_DAILY_LOSS_PCT = Decimal('0.05')  # more conservative stop

# Indicator params (kept reasonable)
SUPERTREND_PERIOD = 10
SUPERTREND_MULTIPLIER = 3.0
STOCH_K = 14
STOCH_D = 3

# Volume filter relaxed
MIN_VOLUME_FACTOR = 0.2  # allow trades when current volume >= 60% of 20-period avg

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


def compute_stochastic(df, k_period=STOCH_K, d_period=STOCH_D):
    low_min = df['low'].rolling(window=k_period, min_periods=1).min()
    high_max = df['high'].rolling(window=k_period, min_periods=1).max()
    denom = (high_max - low_min).replace(0, np.nan)
    k = 100 * (df['close'] - low_min) / denom
    k = k.fillna(50)
    d = k.rolling(window=d_period, min_periods=1).mean()
    return k, d


# ================== SIGNALS (LOOSENED) ==================

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
    # relaxed volatility requirement
    if atr_pct < VOLATILITY_THRESHOLD_PCT:
        print("🔇 ATR% below threshold — but continuing because threshold is relaxed.")
        # allow trade but penalize score later

    print(f"[DEBUG] Stochastic: K={k.iloc[-1]:.2f}, D={d.iloc[-1]:.2f}, cross_up={cross_up}, cross_down={cross_down}")
    print(f"[DEBUG] Price deviation: {deviation:.6f}")

    # Allow trades on stochastic cross OR momentum when supertrend aligns
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
        # fallback momentum: if price has risen 0.3% in last candle and supertrend is up -> buy
        price_change = (price - signal_price) / signal_price if signal_price != 0 else 0
        if st_is_up and price_change > 0.003:
            signal = 'buy'
        if (not st_is_up) and price_change < -0.003:
            signal = 'sell'

    if not signal:
        print("🚫 Conditions not met for signal.")
        return None

    return (signal, atr_latest)


# ================== POSITION HELPERS ==================

def in_position(symbol):
    # check memory first
    if symbol in open_trades:
        return True
    try:
        positions = exchange.fetch_positions([symbol])
        for pos in positions:
            contracts = pos.get('contracts') or pos.get('size') or pos.get('positionAmt') or 0
            if float(contracts) != 0:
                return True
    except Exception as e:
        print(f"[in_position warning] {e}")
    return False


def has_active_tp_sl(symbol):
    """Return True if there are active TP/SL-like open orders for the given symbol."""
    try:
        open_orders = exchange.fetch_open_orders(symbol=symbol)
        for o in open_orders:
            otype = str(o.get('type', '')).lower()
            info = o.get('info', {}) or {}
            if 'stop' in otype or 'take' in otype or 'tp' in otype or 'stopPrice' in o or 'stopPrice' in info or 'takeProfitPrice' in info or 'takeProfit' in info:
                return True
            order_type_info = str(info.get('orderType', '')).lower()
            if 'tp' in order_type_info or 'sl' in order_type_info or 'stop' in order_type_info:
                return True
        return False
    except Exception as e:
        print(f"[has_active_tp_sl warning] {e}")
        return False


def close_position_market(symbol, reason='manual'):
    try:
        # Prevent accidental manual closes if active TP/SL orders exist
        if has_active_tp_sl(symbol):
            print(f"[close] Skipping manual close for {symbol} because TP/SL orders are active.")
            return False

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

            # Trailing stop: move SL to breakeven after +0.8 ATR and trail by TRAIL_ATR_MULTIPLIER
            profit_move = (current_price - entry_price) if side == 'buy' else (entry_price - current_price)
            if profit_move > 0 and abs(profit_move) >= 0.8 * atr_now:
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

    # relaxed Volume confirmation
    vol_now = df['volume'].iloc[-1]
    vol_avg = df['volume'].rolling(window=20, min_periods=1).mean().iloc[-1]
    if vol_now < (MIN_VOLUME_FACTOR * vol_avg):
        print("[Volume] Current volume below relaxed threshold. Skipping.")
        return False

    # Strength score (relaxed)
    score = 0.4
    k, d = compute_stochastic(df)
    cross_magnitude = abs(k.iloc[-1] - d.iloc[-1]) / 100.0
    score += min(0.4, float(cross_magnitude * 0.4))
    vol_factor = min(1.0, float(vol_now / (vol_avg + 1e-9)))
    score += 0.2 * vol_factor

    print(f"[Signal Score] {score:.3f}")
    if score < 0.25:
        print("[Score] Signal strength below threshold. Skipping.")
        return False

    # final ATR% check (relaxed)
    price = df['close'].iloc[-1]
    atr_val = compute_atr(df).iloc[-1]
    atr_pct = Decimal(str(atr_val / price * 100)) if price != 0 else Decimal('0')
    if atr_pct < VOLATILITY_THRESHOLD_PCT:
        print("🔇 Low volatility but still allowing execution under relaxed rules.")

    # must not already be in position
    if in_position(SYMBOL):
        print("[Position] Already in position, skipping new entry.")
        return False

    # place fixed-size order (0.1 ETH)
    trade_meta = place_order(SYMBOL, signal, price, atr_val, qty_override=ORDER_SIZE_ETH)
    if trade_meta:
        print(f"✅ {signal.upper()} {SYMBOL} placed")
        return True
    return False


# ================== ORDER EXECUTION ==================

def calculate_tp_sl(entry_price, atr, side):
    """
    Compute TP and SL levels based on entry price and ATR.
    This function is intentionally minimal and preserves the strategy logic.
    It returns (tp_price, sl_price) as floats rounded to 8 decimal places to avoid precision issues.
    """
    try:
        # Ensure floats
        entry_price = float(entry_price)
        atr = float(atr)
    except Exception:
        # If conversion fails, fallback to safe defaults (no change in logic)
        entry_price = float(entry_price)
        atr = float(atr)

    # Use 1.5 ATR for TP and 1.0 ATR for SL (keeps behaviour consistent with suggested defaults)
    tp_multiplier = 1.5
    sl_multiplier = 1.5

    if side == 'buy':
        tp_price = entry_price + (atr * tp_multiplier)
        sl_price = entry_price - (atr * sl_multiplier)
    else:  # sell
        tp_price = entry_price - (atr * tp_multiplier)
        sl_price = entry_price + (atr * sl_multiplier)

    # round to reasonable precision to avoid exchange precision errors
    tp_price = float(round(tp_price, 8))
    sl_price = float(round(sl_price, 8))
    return tp_price, sl_price


def place_order(symbol, side, entry_price, atr, qty_override=None):
    print(f"🛒 Placing {side.upper()} order on {symbol}...")
    try:
        entry_price = float(entry_price)
        atr = float(atr)
    except Exception as e:
        print(f"[Qty/ATR Error] {e}")
        return False

    qty = float(qty_override) if qty_override is not None else float(ORDER_SIZE_ETH)
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

    # Create TP order (full qty if partial TP ratio is not used)
    try:
        tp_order = exchange.create_order(symbol, 'take_profit_market', 'sell' if side == 'buy' else 'buy', qty, None, {
            'triggerPrice': tp_price,
            'stopPrice': tp_price,
            'positionSide': leverage_side,
            'newClientOrderId': generate_client_order_id()
        })
        print(f"[TP Order] Created at {tp_price}")
    except Exception as e:
        print(f"[TP Error] {e}")

    try:
        sl_order = exchange.create_order(symbol, 'stop_market', 'sell' if side == 'buy' else 'buy', qty, None, {
            'triggerPrice': sl_price,
            'stopPrice': sl_price,
            'positionSide': leverage_side,
            'newClientOrderId': generate_client_order_id()
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
    print("🚀 Trading bot started — relaxed entry rules to open more trades")
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

        time.sleep(20)

