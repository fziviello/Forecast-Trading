from config import MAX_MARGIN, MIN_MARGIN, LOT_SIZE, CONTRACT_SIZE

def get_profit(predicted_take_profit, predicted_entry_price, exchange_rate):
    price_difference = predicted_take_profit - predicted_entry_price
    profit = price_difference * CONTRACT_SIZE * LOT_SIZE * exchange_rate
    return round(abs(profit), 2)

def get_loss(predicted_stop_loss, predicted_entry_price, exchange_rate):
    price_difference = predicted_entry_price - predicted_stop_loss
    loss = price_difference * CONTRACT_SIZE * LOT_SIZE * exchange_rate
    return round(abs(loss), 2)

def get_dynamic_margin(entry_price, volatility, atr, reward_ratio=1, recovery_ratio=2, atr_weight=0.5):
    weighted_volatility = (volatility * (1 - atr_weight)) + (atr * atr_weight)
    tp_margin = weighted_volatility * reward_ratio
    sl_margin = tp_margin * recovery_ratio
    max_margin = entry_price * MAX_MARGIN
    min_margin = entry_price * MIN_MARGIN
    tp_margin = min(max(tp_margin, min_margin), max_margin)
    sl_margin = min(max(sl_margin, min_margin * recovery_ratio), max_margin * recovery_ratio)
    return tp_margin, sl_margin