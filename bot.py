imimport os
import time
import pandas as pd
import numpy as np
import datetime
import ccxt
import uuid
from decimal import Decimal, getcontext

getcontext().prec = 18

# ================== CONFIG ===================
SYMBOLS = ['ETH/USDT:USDT']
TIMEFRAME = '15m'
CONFIRM_TIMEFRAME = '1h'  # higher timeframe confirmation
ORDER_SIZE_BY_SYMBOL = {
    'ETH/USDT:USDT': Decimal('0.1')  # fallback fixed size (contracts/base/quote interpretation left to user)
}

# Volatility thresholds (ATR as percentage of price)
VOLATILITY_THRESHOLD_PCT = Decimal('0.08')  # minimum ATR% to allow trading
VOL_LOW_PCT = Decimal('0.20')
VOL_HIGH_PCT = Decimal('0.60')

# ATR-based multipliers for SL and TP depending on volatility bucket
SL_MULT_LOW = Decimal('1.0')
TP_MULT_LOW = Decimal('2.0')
SL_MULT_MID = Decimal('1.5')
TP_MULT_MID = Decimal('3.0')
SL_MULT_HIGH = Decimal('2.5')
TP_MULT_HIGH = Decimal('4.0')

# Other risk / trade management
COOLDOWN_PERIOD = 60 * 30
FRESH_SIGNAL_MAX_AGE_CANDLES = 1
FRESH_SIGNAL_MAX_PRICE_DEVIATION = 0.006
RISK_PER_TRADE_PCT = Decimal('0.005')  # risk 0.5% of account equity per trade when using dynamic sizing
MAX_DAILY_LOSS_PCT = Decimal('0.03')  # stop trading for the day after 3% loss
PARTIAL_TP_RATIO = Decimal('0.5')  # take 50% at first TP
TRAIL_ATR_MULTIPLIER = Decimal('1.5')  # trail by 1.5 ATR
TIME_BASED_EXIT_HOURS = 6

# Exchange credentials (consider moving to environment variables)
exchange = ccxt.bingx({
    'apiKey': 'wGY6iowJ9qdr1idLbKOj81EGhhZe5O8dqqZlyBiSjiEZnuZUDULsAW30m4eFaZOu35n5zQktN7a01wKoeSg',
    'secret': 'tqxcIVDdDJm2GWjinyBJH4EbvJrjIuOVyi7mnKOzhXHquFPNcULqMAOvmSy0pyuoPOAyCzE2zudzEmlwnA',
    'enableRateLimit': True,
    'options': {
        'defaultType': 'swap',
    }
})

last_trade_time = {symbol: 0 for symbol in SYMBOLS}
open_trades = {}  # store active trade metadata for monitoring
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


# ================== ORDER EXECUTION =======================

def place_order(symbol, side, entry_price, atr, qty_override=None):
    """Place market order and create ATR-based TP and SL orders.
    Returns a dictionary with order metadata on success, or False on failure.
    """
    print(f"🛒 Placing {side.upper()} order on {symbol}...")
    try:
        entry_price = float(entry_price)
        atr = float(atr)
    except Exception as e:
        print(f"[Qty/ATR Error] {e}")
        return False

    # determine qty: if qty_override provided use it, else use adaptive sizing + fallback
    qty = None
    try:
        if qty_override is not None:
            qty = float(qty_override)
        else:
            # try dynamic sizing based on account equity and ATR
            try:
                balance = fetch_account_equity()
                # position size = (risk_pct * equity) / (atr * price_unit)
                risk_amount = (RISK_PER_TRADE_PCT * Decimal(balance)).quantize(Decimal('0.00000001'))
                # convert to float for calculation
                price = Decimal(str(entry_price))
                atr_dec = Decimal(str(atr))
                # avoid division by zero
                if atr_dec == 0:
                    raise Exception('ATR is zero')
                raw_qty = (risk_amount / atr_dec)
                # raw_qty may not match exchange contract units; use it as guide and cap by configured ORDER_SIZE_BY_SYMBOL
                fallback = ORDER_SIZE_BY_SYMBOL.get(symbol, Decimal('0'))
                qty = float(min(raw_qty, fallback)) if fallback > 0 else float(raw_qty)
                if qty <= 0:
                    qty = float(fallback)
            except Exception as e:
                print(f"[Sizing Warning] dynamic sizing failed: {e}")
                qty = float(ORDER_SIZE_BY_SYMBOL.get(symbol, Decimal('0')))
    except Exception as e:
        print(f"[Qty compute error] {e}")
        return False

    print(f"[DEBUG] Qty: {qty}")

    # attempt to set leverage / position mode (best-effort)
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
            exchange.set_leverage(15, symbol, params={'marginMode': 'cross', 'side': leverage_side})
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

    # Calculate dynamic TP/SL based on ATR & current price
    tp_price, sl_price = calculate_tp_sl(entry_price, atr, side)

    # split qty for partial TP
    try:
        qty_total = qty
        qty_first = float(Decimal(str(qty_total)) * PARTIAL_TP_RATIO)
        qty_rest = float(Decimal(str(qty_total)) - Decimal(str(qty_first)))
    except Exception:
        qty_first = qty * float(PARTIAL_TP_RATIO)
        qty_rest = qty - qty_first

    # place first TP for partial profit (best-effort)
    try:
        tp_order_1 = exchange.create_order(symbol, 'limit', 'sell' if side == 'buy' else 'buy', qty_first, tp_price, {
            'positionSide': leverage_side,
            'newClientOrderId': generate_client_order_id()
        })
        print(f"[TP1 Order] Created for {qty_first} at {tp_price}")
    except Exception as e:
        print(f"[TP1 Error] {e}")

    # place stop loss for full size initially
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

    # store open trade metadata for monitoring & trailing
    open_trades[symbol] = {
        'side': side,
        'entry_price': float(entry_price),
        'qty_total': qty_total,
        'qty_first': qty_first,
        'qty_rest': qty_rest,
        'tp1_price': tp_price,
        'tp2_price': None,  # second TP not used; trailing will manage remaining
        'sl_price': sl_price,
        'atr': float(atr),
        'entry_time': time.time(),
        'leverage_side': leverage_side
    }

    last_trade_time[symbol] = time.time()
    return open_trades[symbol]


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
    atr_pct = (atr_dec / price) * Decimal('100') if price != 0 else Decimal('0')

    # Determine volatility bucket
    if atr_pct <= VOL_LOW_PCT:
        sl_mult = SL_MULT_LOW
        tp_mult = TP_MULT_LOW
    elif atr_pct <= VOL_HIGH_PCT:
        sl_mult = SL_MULT_MID
        tp_mult = TP_MULT_MID
    else:
        sl_mult = SL_MULT_HIGH
        tp_mult = TP_MULT_HIGH

    sl_distance = (atr_dec * sl_mult)
    tp_distance = (atr_dec * tp_mult)

    if side == 'buy':
        sl_price = float((price - sl_distance).quantize(Decimal('0.01')))
        tp_price = float((price + tp_distance).quantize(Decimal('0.01')))
    else:
        sl_price = float((price + sl_distance).quantize(Decimal('0.01')))
        tp_price = float((price - tp_distance).quantize(Decimal('0.01')))

    print(f"[TP/SL] ATR%={atr_pct:.4f}%, SL_mult={sl_mult}, TP_mult={tp_mult}, TP={tp_price}, SL={sl_price}")
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

    # Freshness check: ensure the cross happened on the previous candle (age = 1)
    age_candles = 1
    if age_candles > FRESH_SIGNAL_MAX_AGE_CANDLES:
        print("⌛ Signal too old")
        return None

    return (signal, atr_latest)


# ================== ACCOUNT HELPERS ==================

def fetch_account_equity():
    """Best-effort fetch of account equity (quote currency) from exchange. Returns Decimal string of equity.
    If fetch fails, returns a conservative default (e.g., 1000.0)."""
    try:
        bal = exchange.fetch_balance()
        # try common keys
        total_equity = None
        if 'total' in bal and isinstance(bal['total'], dict):
            # try USDT or quote currency
            for k in ['USDT', 'USD', 'usdt', 'usd']:
                if k in bal['total'] and bal['total'][k] is not None:
                    total_equity = Decimal(str(bal['total'][k]))
                    break
        if total_equity is None:
            # fallback to free+used sum if present
            total = 0
            if 'info' in bal:
                # exchange-specific; best-effort
                total = sum([v for v in bal['total'].values() if isinstance(v, (int, float))])
                total_equity = Decimal(str(total))
        if total_equity is None:
            raise Exception('could not parse balance')
        return total_equity
    except Exception as e:
        print(f"[Balance Warning] Could not fetch equity: {e}. Using fallback 1000.0")
        return Decimal('1000.0')


# ================== POSITION & MONITORING HELPERS ==================

def in_position(symbol):
    try:
        positions = exchange.fetch_positions([symbol])
        for pos in positions:
            contracts = pos.get('contracts') or pos.get('size') or pos.get('positionAmt') or 0
            if float(contracts) != 0:
                return True
    except Exception as e:
        print(f"[in_position warning] {e}")
    # fallback: check open_trades memory
    return symbol in open_trades


def close_position_market(symbol, reason='manual'):
    """Attempt to close entire position at market and update daily pnl estimate (best-effort)."""
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
        # update daily_pnl estimate crudely using entry price vs market fill price if available
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
    """Run routine checks for trailing stop adjustments, supertrend flip exit, and time-based exit."""
    for symbol, meta in list(open_trades.items()):
        try:
            df_15 = fetch_ohlcv(symbol, TIMEFRAME, limit=50)
            st_dir_15, atr_15 = compute_supertrend(df_15)
            st_is_up_15 = bool(st_dir_15.iloc[-1])

            # check higher timeframe supertrend flip as additional safety
            df_1h = fetch_ohlcv(symbol, CONFIRM_TIMEFRAME, limit=50)
            st_dir_1h, atr_1h = compute_supertrend(df_1h)
            st_is_up_1h = bool(st_dir_1h.iloc[-1])

            side = meta['side']
            entry_price = meta['entry_price']
            current_price = df_15['close'].iloc[-1]
            atr_now = float(atr_15.iloc[-1])

            # Supertrend flip exit: if either timeframe flips against us, close
            if side == 'buy' and (not st_is_up_15 or not st_is_up_1h):
                print(f"[Exit] Supertrend flipped against BUY on {symbol}. Closing position.")
                close_position_market(symbol, reason='supertrend_flip')
                continue
            if side == 'sell' and (st_is_up_15 or st_is_up_1h):
                print(f"[Exit] Supertrend flipped against SELL on {symbol}. Closing position.")
                close_position_market(symbol, reason='supertrend_flip')
                continue

            # Time-based exit
            age_seconds = time.time() - meta['entry_time']
            if age_seconds > TIME_BASED_EXIT_HOURS * 3600:
                print(f"[Exit] Time-based exit hit for {symbol}. Closing position.")
                close_position_market(symbol, reason='time_exit')
                continue

            # Trailing stop management: move stop to breakeven after +1 ATR and trail by TRAIL_ATR_MULTIPLIER
            profit_move = (current_price - entry_price) if side == 'buy' else (entry_price - current_price)
            if profit_move > 0 and abs(profit_move) >= atr_now:
                # new stop at current_price - trail_multiplier * atr (for buys)
                if side == 'buy':
                    new_sl = current_price - float(Decimal(str(atr_now)) * TRAIL_ATR_MULTIPLIER)
                else:
                    new_sl = current_price + float(Decimal(str(atr_now)) * TRAIL_ATR_MULTIPLIER)
                # only update if tighter for the position
                try:
                    # best-effort: place a new stop_market to replace previous one (exchange semantics vary)
                    print(f"[Trail] Updating SL for {symbol} to {new_sl}")
                    # we attempt to place a new stop order for remaining qty
                    remaining_qty = meta.get('qty_rest', meta.get('qty_total'))
                    exchange.create_order(symbol, 'stop_market', 'sell' if side == 'buy' else 'buy', remaining_qty, None, {
                        'triggerPrice': new_sl,
                        'positionSide': meta.get('leverage_side'),
                        'newClientOrderId': generate_client_order_id(),
                        'stopPrice': new_sl
                    })
                    meta['sl_price'] = new_sl
                except Exception as e:
                    print(f"[Trail Error] {e}")

        except Exception as e:
            print(f"[monitor error] {symbol}: {e}")


# ================== SIGNAL STRENGTH & MULTITF CHECK ==================

def signal_strength_and_confirm(symbol, df_15):
    """Return (signal, atr_value) only if higher timeframe confirms. Also compute a simple strength score."""
    # primary 15m signal
    base = is_fresh_signal(df_15)
    if not base:
        return None
    signal, atr_15 = base

    # fetch 1h for confirmation
    df_1h = fetch_ohlcv(symbol, CONFIRM_TIMEFRAME, limit=150)
    st_dir_1h, atr_1h = compute_supertrend(df_1h)
    try:
        st_is_up_1h = bool(st_dir_1h.iloc[-1])
    except Exception:
        st_is_up_1h = False

    # require supertrend direction to match 15m
    # compute 15m supertrend direction
    st_dir_15, _ = compute_supertrend(df_15)
    try:
        st_is_up_15 = bool(st_dir_15.iloc[-1])
    except Exception:
        st_is_up_15 = False

    if signal == 'buy' and not st_is_up_1h:
        print("[MTF] 1H Supertrend does not confirm BUY. Skipping.")
        return None
    if signal == 'sell' and st_is_up_1h:
        print("[MTF] 1H Supertrend does not confirm SELL. Skipping.")
        return None

    # Volume confirmation on 15m
    vol_now = df_15['volume'].iloc[-1]
    vol_avg = df_15['volume'].rolling(window=20, min_periods=1).mean().iloc[-1]
    if vol_now < vol_avg:
        print("[Volume] Current volume below 20-period average. Skipping.")
        return None

    # Simple strength score: weights -> supertrend alignment (0.6), stochastic cross magnitude (0.3), volume (0.1)
    score = 0.0
    # supertrend alignment -> full points if 1h and 15m agree
    score += 0.6
    # stochastic cross magnitude
    k, d = compute_stochastic(df_15)
    cross_magnitude = abs(k.iloc[-1] - d.iloc[-1]) / 100.0
    score += min(0.3, float(cross_magnitude * 0.3))
    # volume factor (relative to avg)
    vol_factor = min(1.0, float(vol_now / (vol_avg + 1e-9)))
    score += 0.1 * vol_factor

    print(f"[Signal Score] {score:.3f} (higher is stronger)")
    # set minimal score threshold for execution
    if score < 0.35:
        print("[Score] Signal strength below threshold. Skipping.")
        return None

    return (signal, atr_15)


# ================== LOGIC ======================

def trade_logic(symbol):
    global daily_loss_stop, current_day, daily_pnl
    print(f"🔍 Analyzing {symbol}...")

    # reset daily trackers at midnight UTC
    today = datetime.date.today()
    if today != current_day:
        print("[Daily] New day - resetting daily PnL and stop flag")
        daily_pnl = Decimal('0')
        daily_loss_stop = False
        current_day = today

    if daily_loss_stop:
        print("[Daily] Trading suspended due to daily loss limit.")
        return False

    if in_position(symbol):
        print(f"⛔️ Already in position for {symbol}")
        return False

    if symbol in last_trade_time:
        since_last = time.time() - last_trade_time[symbol]
        if since_last < COOLDOWN_PERIOD:
            print(f"⏳ Cooling down ({int((COOLDOWN_PERIOD - since_last) / 60)} min left)...")
            return False

    df_15 = fetch_ohlcv(symbol, TIMEFRAME)

    # multi-timeframe confirmation + scoring
    signal_result = signal_strength_and_confirm(symbol, df_15)
    if not signal_result:
        return False

    signal, atr = signal_result
    price = df_15['close'].iloc[-1]

    # final check: ATR% threshold
    atr_pct = Decimal(str(atr / price * 100)) if price != 0 else Decimal('0')
    if atr_pct < VOLATILITY_THRESHOLD_PCT:
        print("🔇 Skipping due to low volatility (post-check)")
        return False

    # place order with dynamic sizing
    trade_meta = place_order(symbol, signal, price, atr)
    if trade_meta:
        print(f"✅ {signal.upper()} {symbol} placed")
        # update daily pnl stop if new pnl pushes us beyond limit
        try:
            if daily_pnl < -MAX_DAILY_LOSS_PCT * fetch_account_equity():
                daily_loss_stop = True
                print("[Daily] Max daily loss reached. Suspending trading for today.")
        except Exception:
            pass
        return True

    print("❌ Order placement failed")
    return False


# ================== MAIN =====================
if __name__ == '__main__':
    print("🚀 Upgraded trading bot started (MTF confirm, ATR trailing, dynamic sizing)...")
    while True:
        try:
            # primary cycle: run analysis & place new trades
            for symbol in SYMBOLS:
                try:
                    trade_logic(symbol)
                except Exception as e:
                    print(f"[Unhandled Error] trade_logic {symbol}: {e}")

            # secondary cycle: monitor existing trades for trailing / exits
            if open_trades:
                monitor_open_trades()

        except Exception as e:
            print(f"[Main loop error] {e}")

        print("⏰ Cycle complete, sleeping 30 seconds...")
        time.sleep(30)
