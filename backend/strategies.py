"""Example trading strategies for the paper-trading simulator.

Each class subclasses BaseStrategy and implements on_tick().
Pass an instance to Simulator() to activate it.
"""

from paper_trading import BaseStrategy


class PassiveStrategy(BaseStrategy):
    """Does nothing — useful as a no-op baseline / placeholder."""

    def on_tick(self, state: dict) -> list[dict]:
        asset_id= state['asset_id']
        mid     = state['mid_price']
        portfolio = state['portfolio']
        cash    = portfolio['cash']
        best_ask= state['features']['best_ask']
        best_bid= state['features']['best_bid']
        position= portfolio['positions'].get(asset_id, 0.0)

        orders: list[dict] = []
        if mid <= 0.4 and cash > 0:
            qty = (cash * 0.25) / best_ask
            orders.append({
                'asset_id': asset_id, 'side': 'BUY', 'order_type': 'limit',
                'quantity': qty, 'price': best_ask + 0.01, 'time_in_force': 'IOC',
            })
            return orders
        elif mid >= 0.6 and position > 0:
            orders.append({
                'asset_id': asset_id, 'side': 'SELL', 'order_type': 'market',
                'quantity': position, 'price': None, 'time_in_force': 'IOC',
            })
            return orders

        return []


class QuantileMomentumStrategy(BaseStrategy):
    """Trades based on the model's median (q50) prediction.

    Logic
    -----
    - If q50 > mid + entry_threshold  → BUY  (expects price to rise)
    - If q50 < mid − entry_threshold  → SELL (expects price to fall)
    - Close an open long  when expected_return < −exit_threshold
    - Close an open short when expected_return >  exit_threshold
    - Sizes each trade as *position_frac* of available cash.
    - Skips entry when predictions are unavailable.
    """

    def __init__(
        self,
        entry_threshold: float = 0.01,
        exit_threshold: float  = 0.005,
        position_frac: float   = 0.1,
        min_cash: float        = 10.0,
    ) -> None:
        self.entry_threshold = entry_threshold
        self.exit_threshold  = exit_threshold
        self.position_frac   = position_frac
        self.min_cash        = min_cash

    def on_tick(self, state: dict) -> list[dict]:
        preds = state.get('predictions')
        if preds is None:
            return []

        asset_id = state['asset_id']
        mid      = state['mid_price']
        q50      = preds.get('q50', mid)
        portfolio = state['portfolio']
        cash     = portfolio['cash']
        position = portfolio['positions'].get(asset_id, 0.0)

        expected_return = q50 - mid
        orders: list[dict] = []

        # ── Close existing position if signal has reversed ─────────────
        if position > 0 and expected_return < -self.exit_threshold:
            orders.append({
                'asset_id': asset_id, 'side': 'SELL', 'order_type': 'market',
                'quantity': position, 'price': None, 'time_in_force': 'IOC',
            })
            return orders

        if position < 0 and expected_return > self.exit_threshold:
            orders.append({
                'asset_id': asset_id, 'side': 'BUY', 'order_type': 'market',
                'quantity': abs(position), 'price': None, 'time_in_force': 'IOC',
            })
            return orders

        # ── Enter new position only when flat ──────────────────────────
        if position == 0.0 and cash > self.min_cash:
            qty = (cash * self.position_frac) / (mid + 1e-9)
            if expected_return > self.entry_threshold:
                orders.append({
                    'asset_id': asset_id, 'side': 'BUY', 'order_type': 'market',
                    'quantity': qty, 'price': None, 'time_in_force': 'IOC',
                })
            elif expected_return < -self.entry_threshold:
                orders.append({
                    'asset_id': asset_id, 'side': 'SELL', 'order_type': 'market',
                    'quantity': qty, 'price': None, 'time_in_force': 'IOC',
                })

        return orders


class BandReversionStrategy(BaseStrategy):
    """Fades moves that breach the model's q10–q90 prediction band.

    Logic
    -----
    - Buy  when mid < q10  (market oversold relative to model's lower bound)
    - Sell when mid > q90  (market overbought relative to model's upper bound)
    - Exit long  when mid rises back above q10.
    - Exit short when mid falls back below q90.
    """

    def __init__(
        self,
        position_frac: float = 0.1,
        min_cash: float      = 10.0,
    ) -> None:
        self.position_frac = position_frac
        self.min_cash      = min_cash

    def on_tick(self, state: dict) -> list[dict]:
        preds = state.get('predictions')
        if preds is None:
            return []

        asset_id = state['asset_id']
        mid      = state['mid_price']
        q10      = preds.get('q10', mid)
        q90      = preds.get('q90', mid)
        portfolio = state['portfolio']
        cash     = portfolio['cash']
        position = portfolio['positions'].get(asset_id, 0.0)

        orders: list[dict] = []

        # ── Exit conditions (mid returned inside band) ─────────────────
        if position > 0 and mid > q10:
            orders.append({
                'asset_id': asset_id, 'side': 'SELL', 'order_type': 'market',
                'quantity': position, 'price': None, 'time_in_force': 'IOC',
            })
            return orders

        if position < 0 and mid < q90:
            orders.append({
                'asset_id': asset_id, 'side': 'BUY', 'order_type': 'market',
                'quantity': abs(position), 'price': None, 'time_in_force': 'IOC',
            })
            return orders

        # ── Entry conditions ───────────────────────────────────────────
        if position == 0.0 and cash > self.min_cash:
            qty = (cash * self.position_frac) / (mid + 1e-9)
            if mid < q10:
                orders.append({
                    'asset_id': asset_id, 'side': 'BUY', 'order_type': 'market',
                    'quantity': qty, 'price': None, 'time_in_force': 'IOC',
                })
            elif mid > q90:
                orders.append({
                    'asset_id': asset_id, 'side': 'SELL', 'order_type': 'market',
                    'quantity': qty, 'price': None, 'time_in_force': 'IOC',
                })

        return orders
