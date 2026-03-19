"""Paper trading engine for simulating trades against live Polymarket order books.

Classes
-------
BaseStrategy    Abstract interface; subclass and implement on_tick().
ExecutionEngine Matches orders against an order-book snapshot, returns fills.
Portfolio       Tracks cash and positions; applies fills to update state.
Simulator       Orchestrates the other three; call on_tick() each second.

Order dict shape
----------------
    asset_id      : str
    side          : 'BUY' | 'SELL'
    order_type    : 'market' | 'limit'
    quantity      : float
    price         : float | None  (required for limit orders)
    time_in_force : 'IOC' | 'FOK' | 'GTC'

Fill dict shape
---------------
    asset_id   : str
    side       : 'BUY' | 'SELL'
    filled_qty : float
    avg_price  : float
    fee        : float
    remaining  : float
    status     : 'filled' | 'partial' | 'rejected'
    timestamp  : int  (Unix ms)
"""

import logging
import math
import random
import time
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# BaseStrategy
# ──────────────────────────────────────────────────────────────────────────────

class BaseStrategy(ABC):
    """Abstract base for all trading strategies.

    Subclass this and implement ``on_tick``.  The Simulator calls it once per
    1-second bar and passes the latest market state.
    """

    @abstractmethod
    def on_tick(self, state: dict) -> list[dict]:
        """Generate orders given the current market state.

        Parameters
        ----------
        state : dict
            {
                'timestamp'   : int          – Unix ms
                'asset_id'    : str          – Polymarket token id
                'order_book'  : dict         – {'bids': [...], 'asks': [...]}
                'mid_price'   : float        – best-bid/ask midpoint in [0, 1]
                'predictions' : dict | None  – {'mid','q10','q50','q90'} or None
                'features'    : dict         – current 1-second feature row
                'portfolio'   : dict         – see Portfolio.snapshot()
            }

        Returns
        -------
        list[dict]  (may be empty)
            Each element is an order dict with keys:
                asset_id, side, order_type, quantity, price, time_in_force
        """


# ──────────────────────────────────────────────────────────────────────────────
# ExecutionEngine
# ──────────────────────────────────────────────────────────────────────────────

class ExecutionEngine:
    """Simulates order execution against a live order-book snapshot.

    Handles market and limit orders (IOC / FOK / GTC) with configurable
    slippage and per-level size constraints.  Outstanding GTC orders are
    stored internally and re-evaluated on every subsequent call to
    ``process_orders``.
    """

    def __init__(
        self,
        taker_fee: float = 0.001,
        maker_fee: float = 0.0,
        slippage_pct: float = 0.002,
        limit_fill_prob: float = 0.5,
    ) -> None:
        self.taker_fee      = taker_fee
        self.maker_fee      = maker_fee
        self.slippage_pct   = slippage_pct
        self.limit_fill_prob = limit_fill_prob  # fill prob when order is right at limit price
        self._open_gtc: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_orders(self, orders: list[dict], order_book: dict) -> list[dict]:
        """Execute *orders* against *order_book* and return fills.

        Also re-attempts any outstanding GTC limit orders against the same
        snapshot.

        Parameters
        ----------
        orders     : New orders emitted by the strategy this tick.
        order_book : Current book ``{'bids': [...], 'asks': [...]}``.

        Returns
        -------
        list[dict]  Fill dicts for every (partial or full) execution.
        """
        bids, asks = self._sort_book(order_book)
        fills: list[dict] = []

        # ── New orders ────────────────────────────────────────────────
        for order in orders:
            fill = self._execute(order, bids, asks, is_gtc_retry=False)
            if fill is not None:
                fills.append(fill)

        # ── Re-attempt outstanding GTC orders ─────────────────────────
        remaining_gtc: list[dict] = []
        for gtc in self._open_gtc:
            fill = self._execute(gtc, bids, asks, is_gtc_retry=True)
            if fill is not None:
                fills.append(fill)
                if fill['status'] == 'partial' and fill.get('remaining', 0.0) > 0:
                    leftover = dict(gtc)
                    leftover['quantity'] = fill['remaining']
                    remaining_gtc.append(leftover)
                # 'filled' status → order fully done, drop from list
            else:
                remaining_gtc.append(gtc)   # still waiting for liquidity

        self._open_gtc = remaining_gtc
        return fills

    @property
    def open_orders(self) -> list[dict]:
        """Read-only view of outstanding GTC orders."""
        return list(self._open_gtc)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _sort_book(self, order_book: dict) -> tuple[list, list]:
        bids = sorted(
            [{'price': float(b['price']), 'size': float(b['size'])} for b in order_book.get('bids', [])],
            key=lambda x: -x['price'],
        )
        asks = sorted(
            [{'price': float(a['price']), 'size': float(a['size'])} for a in order_book.get('asks', [])],
            key=lambda x: x['price'],
        )
        return bids, asks

    def _limit_fill_prob(self, limit_px: float, level_px: float, side: str) -> float:
        """Probability of receiving a fill at *level_px* given a limit order at *limit_px*.

        Models queue-position uncertainty: the more aggressively the limit price
        overshoots the book level, the more volume must trade before us in the
        queue, so the higher the chance we are reached.

        At rel_overshoot = 0  (level exactly at limit):  prob = limit_fill_prob
        As rel_overshoot → ∞  (very aggressive limit):   prob → 1.0

        Formula: prob = 1 − (1 − base) × exp(−k × rel_overshoot)
        where k = 20 gives saturation around a 15–20% overshoot.
        """
        if side == 'BUY':
            rel_overshoot = (limit_px - level_px) / (limit_px + 1e-9)
        else:
            rel_overshoot = (level_px - limit_px) / (level_px + 1e-9)

        rel_overshoot = max(0.0, rel_overshoot)
        return 1.0 - (1.0 - self.limit_fill_prob) * math.exp(-20.0 * rel_overshoot)

    def _slip_price(self, base: float, side: str, level_size: float, take: float) -> float:
        """Apply proportional slippage with a small random component."""
        pct   = self.slippage_pct * (1.0 + 3.0 * take / (level_size + 1e-9))
        noise = (random.random() - 0.5) * 0.2   # ±10% noise
        if side == 'BUY':
            return base * (1.0 + pct * (1.0 + noise))
        return base * (1.0 - pct * (1.0 + noise))

    def _execute(
        self,
        order: dict,
        bids_sorted: list[dict],
        asks_sorted: list[dict],
        is_gtc_retry: bool,
    ) -> Optional[dict]:
        """Attempt to fill one order.  Returns a fill dict or None."""
        asset_id   = order.get('asset_id', '')
        side       = order['side']
        order_type = order.get('order_type', 'market')
        quantity   = float(order['quantity'])
        tif        = order.get('time_in_force', 'IOC')
        limit_px   = order.get('price')
        ts         = int(time.time() * 1000)

        opp = asks_sorted if side == 'BUY' else bids_sorted

        # Filter to levels accessible at the limit price
        if order_type == 'limit':
            if limit_px is None:
                raise ValueError('limit orders require a price')
            levels = (
                [lv for lv in opp if lv['price'] <= limit_px]
                if side == 'BUY'
                else [lv for lv in opp if lv['price'] >= limit_px]
            )
        else:
            levels = opp   # market order: sweep everything

        available = sum(lv['size'] for lv in levels)

        # FOK: all-or-nothing
        if tif == 'FOK' and available < quantity:
            return {
                'asset_id': asset_id, 'side': side,
                'filled_qty': 0.0, 'avg_price': 0.0, 'fee': 0.0,
                'remaining': quantity, 'status': 'rejected', 'timestamp': ts,
            }

        # ── Classify the order's execution mode ───────────────────────
        #
        # Marketable limit (crosses the spread):
        #   guaranteed fill (no queue draw) + slippage (taker)
        # Passive / resting limit (GTC retries included):
        #   queue-position uncertainty (Bernoulli draw per level) + no slippage (maker)
        # Market order:
        #   guaranteed fill + slippage (always taker)
        # FOK:
        #   deterministic everywhere (handled by the available-check above)
        if order_type == 'market':
            use_prob_fill  = False
            apply_slippage = True
        else:  # limit
            if is_gtc_retry:
                # Was already resting in the book → passive treatment regardless
                is_marketable = False
            elif side == 'BUY':
                best_opp      = asks_sorted[0]['price'] if asks_sorted else float('inf')
                is_marketable = limit_px >= best_opp
            else:
                best_opp      = bids_sorted[0]['price'] if bids_sorted else 0.0
                is_marketable = limit_px <= best_opp
            use_prob_fill  = not is_marketable and (tif != 'FOK')
            apply_slippage = is_marketable

        # Walk the book best-first
        filled = 0.0
        cost   = 0.0
        for lv in levels:
            if filled >= quantity:
                break
            if use_prob_fill:
                prob = self._limit_fill_prob(limit_px, lv['price'], side)
                if random.random() > prob:
                    continue   # queue didn't reach our order at this level
            take       = min(quantity - filled, lv['size'])
            fill_price = (
                self._slip_price(lv['price'], side, lv['size'], take)
                if apply_slippage
                else lv['price']   # passive maker: fill at exact book price
            )
            filled += take
            cost   += take * fill_price

        remaining = max(0.0, quantity - filled)

        if filled == 0.0:
            # Queue as GTC if this is a fresh order (not a retry)
            if tif == 'GTC' and not is_gtc_retry:
                self._open_gtc.append(dict(order))
            return None

        avg_price = cost / filled
        fee_rate  = self.maker_fee if (order_type == 'limit' and tif == 'GTC') else self.taker_fee
        fee       = avg_price * filled * fee_rate
        status    = 'filled' if remaining == 0.0 else 'partial'

        # Queue unfilled remainder as GTC
        if tif == 'GTC' and remaining > 0.0 and not is_gtc_retry:
            leftover             = dict(order)
            leftover['quantity'] = remaining
            self._open_gtc.append(leftover)

        return {
            'asset_id':   asset_id,
            'side':       side,
            'filled_qty': filled,
            'avg_price':  avg_price,
            'fee':        fee,
            'remaining':  remaining,
            'status':     status,
            'timestamp':  ts,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Portfolio
# ──────────────────────────────────────────────────────────────────────────────

class Portfolio:
    """Tracks cash and positions; applies fills to update state.

    Uses a weighted-average cost basis for PnL accounting.
    """

    def __init__(self, starting_cash: float = 1000.0) -> None:
        self.cash: float                   = float(starting_cash)
        self.positions: dict[str, float]   = {}   # asset_id → net qty
        self._cost_basis: dict[str, float] = {}   # asset_id → avg cost per unit
        self.realized_pnl: float           = 0.0
        self.trade_history: list[dict]     = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_fills(self, fills: list[dict]) -> None:
        """Update cash, positions, and realized PnL from a list of fills."""
        for fill in fills:
            self._apply(fill)

    def unrealized_pnl(self, current_prices: dict[str, float]) -> float:
        """Sum of (current_price − avg_cost) × qty for all open positions."""
        total = 0.0
        for asset, qty in self.positions.items():
            price = current_prices.get(asset)
            if price is None or qty == 0.0:
                continue
            avg_cost = self._cost_basis.get(asset, price)
            total += (price - avg_cost) * qty
        return total

    def total_value(self, current_prices: dict[str, float]) -> float:
        """Cash + mark-to-market value of all open positions."""
        pos_value = sum(
            qty * current_prices.get(asset, 0.0)
            for asset, qty in self.positions.items()
        )
        return self.cash + pos_value

    def snapshot(self, current_prices: Optional[dict[str, float]] = None) -> dict:
        """Return a JSON-serialisable portfolio snapshot."""
        prices = current_prices or {}
        return {
            'cash':           round(self.cash, 6),
            'positions':      dict(self.positions),
            'cost_basis':     dict(self._cost_basis),
            'realized_pnl':   round(self.realized_pnl, 6),
            'unrealized_pnl': round(self.unrealized_pnl(prices), 6),
            'total_value':    round(self.total_value(prices), 6),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _apply(self, fill: dict) -> None:
        if fill['status'] == 'rejected' or fill.get('filled_qty', 0.0) == 0.0:
            return

        asset    = fill['asset_id']
        side     = fill['side']
        qty      = fill['filled_qty']
        price    = fill['avg_price']
        fee      = fill['fee']
        cost     = price * qty
        prev_qty = self.positions.get(asset, 0.0)
        prev_cb  = self._cost_basis.get(asset, 0.0)

        if side == 'BUY':
            self.cash -= (cost + fee)
            new_qty = prev_qty + qty
            self._cost_basis[asset] = (
                (prev_qty * prev_cb + qty * price) / new_qty if new_qty else price
            )
            self.positions[asset] = new_qty
        else:  # SELL
            self.cash += (cost - fee)
            self.realized_pnl += (price - prev_cb) * qty
            new_qty = prev_qty - qty
            if new_qty <= 1e-9:
                self.positions.pop(asset, None)
                self._cost_basis.pop(asset, None)
            else:
                self.positions[asset] = new_qty

        self.trade_history.append({
            'timestamp': fill['timestamp'],
            'asset_id':  asset,
            'side':      side,
            'qty':       qty,
            'price':     price,
            'fee':       fee,
        })


# ──────────────────────────────────────────────────────────────────────────────
# Simulator
# ──────────────────────────────────────────────────────────────────────────────

class Simulator:
    """Orchestrates BaseStrategy, ExecutionEngine, and Portfolio.

    Call ``on_tick`` once per 1-second bar from the live data feed in
    ``main.py``.  Strategy errors are caught so they never interrupt the
    stream.

    Example
    -------
    ::
        from paper_trading import Simulator
        from strategies import QuantileMomentumStrategy

        sim = Simulator(strategy=QuantileMomentumStrategy(), starting_cash=1000.0)

        # Inside run_polymarket_stream, after aggregating a 1-second bar:
        result = sim.on_tick(
            asset_id    = current_token_ids[0],
            order_book  = _curr_books[current_token_ids[0]],
            predictions = preds,
            features    = row_1s,
            timestamp   = int(now * 1000),
        )
        # result: {'fills': [...], 'portfolio': {...}, 'pnl': float}
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        starting_cash: float = 1000.0,
        taker_fee: float = 0.001,
        maker_fee: float = 0.0,
        slippage_pct: float = 0.002,
        limit_fill_prob: float = 0.5,
    ) -> None:
        self.strategy       = strategy
        self.engine         = ExecutionEngine(
            taker_fee=taker_fee, maker_fee=maker_fee,
            slippage_pct=slippage_pct, limit_fill_prob=limit_fill_prob,
        )
        self.portfolio      = Portfolio(starting_cash=starting_cash)
        self._starting_cash = starting_cash
        self._tick_count    = 0

    # ------------------------------------------------------------------
    # Main entry point (called from main.py)
    # ------------------------------------------------------------------

    def on_tick(
        self,
        asset_id: str,
        order_book: dict,
        predictions: Optional[dict],
        features: dict,
        timestamp: int,
    ) -> dict:
        """Process one 1-second bar.

        Parameters
        ----------
        asset_id    : Polymarket token id for the subscribed market.
        order_book  : Current book snapshot ``{bids: [...], asks: [...]}``.
        predictions : Model output ``{mid, q10, q50, q90}`` or None.
        features    : 1-second aggregated feature row from main.py.
        timestamp   : Unix milliseconds.

        Returns
        -------
        dict
            {
                'fills'     : list[dict]  – fills executed this tick
                'portfolio' : dict        – portfolio snapshot
                'pnl'       : float       – realized + unrealized PnL
            }
        """
        self._tick_count += 1

        bids      = order_book.get('bids', [])
        asks      = order_book.get('asks', [])
        best_bid  = max((b['price'] for b in bids), default=0.0)
        best_ask  = min((a['price'] for a in asks), default=1.0)
        mid_price = (best_bid + best_ask) / 2.0
        current_prices = {asset_id: mid_price}

        state = {
            'timestamp':   timestamp,
            'asset_id':    asset_id,
            'order_book':  order_book,
            'mid_price':   mid_price,
            'predictions': predictions,
            'features':    features,
            'portfolio':   self.portfolio.snapshot(current_prices),
        }

        try:
            orders = self.strategy.on_tick(state)
        except Exception as exc:
            logger.warning('[Simulator] strategy error on tick %d: %s', self._tick_count, exc)
            orders = []

        fills = self.engine.process_orders(orders, order_book)
        self.portfolio.apply_fills(fills)

        snap = self.portfolio.snapshot(current_prices)
        return {
            'fills':     fills,
            'portfolio': snap,
            'pnl':       snap['realized_pnl'] + snap['unrealized_pnl'],
        }

    def reset(self) -> None:
        """Reset portfolio and engine to initial state; strategy is preserved."""
        self.engine = ExecutionEngine(
            taker_fee=self.engine.taker_fee,
            maker_fee=self.engine.maker_fee,
            slippage_pct=self.engine.slippage_pct,
            limit_fill_prob=self.engine.limit_fill_prob,
        )
        self.portfolio   = Portfolio(starting_cash=self._starting_cash)
        self._tick_count = 0
