import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import time
import requests
import json
import random
import math
import csv
import os

GAMMA_API = 'https://gamma-api.polymarket.com'
CLOB_API = 'https://clob.polymarket.com'


# ------- HELPERS -----------#
def get_token_ids_from_slug(slug):
    # Gamma API handles market discovery
    url = f"{GAMMA_API}/markets"
    response = requests.get(url, params={'slug': slug}).json()
    
    print(response[0]['question'])
    # clobTokenIds is a list: index 0 is YES, index 1 is NO
    token_ids = json.loads(response[0]['clobTokenIds'])

    return token_ids


def get_polymarket_book(token_id):
    """
    Fetches the live Order Book (Bids and Asks) for a specific token.
    """
    url = f"{CLOB_API}/book"
    params = {"token_id": token_id}
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        print(f"--- Order Book for {token_id} ---")
        print(f"Lowest Ask: {data['asks'][-1]['price']} (Size: {data['asks'][-1]['size']})")
        print(f"Highest Bid: {data['bids'][-1]['price']} (Size: {data['bids'][-1]['size']})")
        
        # Return the full data for your RL agent's observation space
        return data
    
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None
    

# TODO: execute both get_polymarket_book calls concurrently in ThreadPoolExecutor()
def get_btc_5m_polymarket_book_by_timestamp():
    # Calculate the current 5-minute interval timestamp
    # 300 seconds = 5 minutes
    current_time = int(time.time())
    current_interval = (current_time // 300) * 300
    slug = f"btc-updown-5m-{current_interval}"

    # Pass slug to get_token_ids_from_slug to retrieve the YES and NO token IDs
    yes_token_id, no_token_id = get_token_ids_from_slug(slug)

    # Fetch the Order Book for the 'Yes' token
    book_data_yes = get_polymarket_book(yes_token_id)
    book_data_no = get_polymarket_book(no_token_id)
    return book_data_yes, book_data_no


# ---------------- Paper Trading Engine -----------------
class PaperTradingEngine:
    """Simple realistic paper trading engine that uses order books
    returned by `get_polymarket_book` and supports market/limit orders
    with time-in-force: GTC, FOK, IOC. Models slippage and fees and
    respects level sizes when filling orders.
    """
    def __init__(self, starting_cash=1000.0, maker_fee=0.0, taker_fee=0.001, slippage_pct=0.002, trades_csv='paper_trades.csv', orders_csv='paper_orders.csv'):
        self.cash = float(starting_cash)
        self.positions = {}  # token_id -> shares (float)
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage_pct = slippage_pct
        self.open_orders = []  # stored GTC limit orders
        self.trade_history = []
        self.trades_csv = trades_csv
        self.orders_csv = orders_csv
        # ensure CSV files exist with headers
        if not os.path.exists(self.trades_csv):
            with open(self.trades_csv, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['timestamp', 'token', 'side', 'qty', 'price', 'fee'])
        if not os.path.exists(self.orders_csv):
            with open(self.orders_csv, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['timestamp', 'token', 'side', 'qty', 'price', 'tif', 'order_type', 'status', 'filled_qty', 'remaining'])

    def _normalize_book(self, book):
        # Convert price/size to floats and return sorted levels
        asks = [{'price': float(x['price']), 'size': float(x['size'])} for x in book.get('asks', [])]
        bids = [{'price': float(x['price']), 'size': float(x['size'])} for x in book.get('bids', [])]
        # Best ask: lowest price; Best bid: highest price
        asks_sorted = sorted(asks, key=lambda x: x['price'])
        bids_sorted = sorted(bids, key=lambda x: x['price'], reverse=True)
        return asks_sorted, bids_sorted

    def _slip_price(self, base_price, side, level_size, take_size):
        # Simple slippage model: slippage grows with proportion of level taken
        pct = self.slippage_pct * (1.0 + 3.0 * (take_size / (level_size + 1e-9)))
        rnd = (random.random() - 0.5) * 0.2  # small randomness
        if side == 'BUY':
            return base_price * (1.0 + pct * (1.0 + rnd))
        else:
            return base_price * (1.0 - pct * (1.0 + rnd))

    def _apply_fill(self, token_id, side, filled_qty, avg_price, fee_rate):
        fee = avg_price * filled_qty * fee_rate
        cost = avg_price * filled_qty
        if side == 'BUY':
            self.cash -= (cost + fee)
            self.positions[token_id] = self.positions.get(token_id, 0.0) + filled_qty
        else:
            self.cash += (cost - fee)
            self.positions[token_id] = self.positions.get(token_id, 0.0) - filled_qty
        self.trade_history.append({'token': token_id, 'side': side, 'qty': filled_qty, 'price': avg_price, 'fee': fee})
        # append to trades CSV
        try:
            with open(self.trades_csv, 'a', newline='') as f:
                w = csv.writer(f)
                w.writerow([time.time(), token_id, side, filled_qty, avg_price, fee])
        except Exception:
            pass

    def submit_order(self, token_id, side, quantity, order_type='market', price=None, time_in_force='IOC'):
        """Submit order to the engine.
        - side: 'BUY' or 'SELL'
        - order_type: 'market' or 'limit'
        - time_in_force: 'IOC', 'FOK', 'GTC'
        Returns dict: filled_qty, avg_price, remaining, status
        """
        book = get_polymarket_book(token_id)
        if book is None:
            return {'filled_qty': 0.0, 'avg_price': None, 'remaining': quantity, 'status': 'book_unavailable'}

        asks, bids = self._normalize_book(book)
        opp = asks if side == 'BUY' else bids

        filled = 0.0
        cost = 0.0

        if order_type == 'market':
            # consume opposite book best-first until filled or exhausted
            for lvl in opp:
                if filled >= quantity:
                    break
                take = min(quantity - filled, lvl['size'])
                if take <= 0:
                    continue
                fill_price = self._slip_price(lvl['price'], side, lvl['size'], take)
                filled += take
                cost += take * fill_price
            if filled == 0:
                return {'filled_qty': 0.0, 'avg_price': None, 'remaining': quantity, 'status': 'rejected'}
            avg_price = cost / filled
            fee_rate = self.taker_fee
            self._apply_fill(token_id, side, filled, avg_price, fee_rate)
            return {'filled_qty': filled, 'avg_price': avg_price, 'remaining': max(0.0, quantity - filled), 'status': 'filled' if filled >= quantity else 'partial'}

        # limit order handling
        assert order_type == 'limit', 'unsupported order_type'
        assert price is not None, 'limit orders require price'

        # determine available at price or better
        available = 0.0
        levels = []
        for lvl in opp:
            if (side == 'BUY' and lvl['price'] <= price) or (side == 'SELL' and lvl['price'] >= price):
                available += lvl['size']
                levels.append(lvl)

        # FOK: all-or-nothing. FAK: fill-and-kill (partial fills allowed, cancel remainder)
        if time_in_force == 'FOK' and available < quantity:
            result = {'filled_qty': 0.0, 'avg_price': None, 'remaining': quantity, 'status': 'rejected_fok'}
            # log order
            try:
                with open(self.orders_csv, 'a', newline='') as f:
                    w = csv.writer(f)
                    w.writerow([time.time(), token_id, side, quantity, price, time_in_force, 'limit', result['status'], 0.0, quantity])
            except Exception:
                pass
            return result

        to_fill = min(available, quantity) if time_in_force in ('FAK', 'FOK') else min(available, quantity)
        for lvl in levels:
            if filled >= to_fill:
                break
            take = min(quantity - filled, lvl['size'])
            if take <= 0:
                continue
            fill_price = self._slip_price(lvl['price'], side, lvl['size'], take)
            filled += take
            cost += take * fill_price

        if filled > 0:
            avg_price = cost / filled
            fee_rate = self.taker_fee if time_in_force in ('IOC', 'FOK') else self.maker_fee
            self._apply_fill(token_id, side, filled, avg_price, fee_rate)
        else:
            avg_price = None

        remaining = max(0.0, quantity - filled)
        status = 'filled' if remaining == 0.0 else ('partial' if filled > 0 else 'open')

        if time_in_force == 'GTC' and remaining > 0.0:
            # store as open limit order
            self.open_orders.append({'token': token_id, 'side': side, 'qty': remaining, 'price': price, 'created_at': time.time()})
        # log order outcome
        try:
            with open(self.orders_csv, 'a', newline='') as f:
                w = csv.writer(f)
                w.writerow([time.time(), token_id, side, quantity, price, time_in_force, 'limit', status, filled, remaining])
        except Exception:
            pass

        return {'filled_qty': filled, 'avg_price': avg_price, 'remaining': remaining, 'status': status}

    def process_open_orders(self):
        # Attempt to fill GTC orders against the latest book snapshot
        remaining_orders = []
        for o in self.open_orders:
            # use FAK (fill-and-kill) to attempt immediate fill of whatever liquidity exists
            res = self.submit_order(o['token'], o['side'], o['qty'], order_type='limit', price=o['price'], time_in_force='FAK')
            if res['remaining'] > 0.0:
                # keep remainder as GTC
                remaining_orders.append({'token': o['token'], 'side': o['side'], 'qty': res['remaining'], 'price': o['price'], 'created_at': o['created_at']})
        self.open_orders = remaining_orders

    def account(self):
        return {'cash': self.cash, 'positions': dict(self.positions), 'open_orders': list(self.open_orders)}




# ------ DEMO --------- #
def paper_trade_demo(polls=6, sleep_sec=5):
    """Simple demo that polls the market a few times and takes simple actions."""
    engine = PaperTradingEngine(starting_cash=500.0, taker_fee=0.001, slippage_pct=0.003)

    yes_token, no_token = 'yes', 'no'

    for i in range(polls):
        yes_book, no_book = get_btc_5m_polymarket_book_by_timestamp()

        # get top-of-book buy price (what you'd pay to buy = best ask)
        asks, bids = engine._normalize_book(yes_book)
        if len(asks) == 0:
            print('no asks available')
            time.sleep(sleep_sec)
            continue
        top_ask = asks[0]['price']
        mid = (asks[0]['price'] + (bids[0]['price'] if bids else asks[0]['price'])) / 2.0
        print(f'Poll {i+1}: top_ask={top_ask:.4f} mid={mid:.4f} cash={engine.cash:.2f}')

        # Very simple strategy: if probability (mid) < 0.40 and we have cash, try FOK buy for 25% of cash
        if mid < 0.40 and engine.cash > 1.0:
            qty = (engine.cash * 0.25) / top_ask
            # use FAK (fill-and-kill) which allows partial fills and cancels the remainder
            res = engine.submit_order(yes_token, 'BUY', qty, order_type='limit', price=top_ask, time_in_force='FAK')
            print('  FAK BUY result:', res)

        # If price > 0.60 and we hold position, sell with IOC market-ish behavior
        pos = engine.positions.get(yes_token, 0.0)
        if mid > 0.60 and pos > 0.0:
            res2 = engine.submit_order(yes_token, 'SELL', pos, order_type='market')
            print('  MARKET SELL result:', res2)

        # occasionally place a GTC limit to try to buy cheaper
        if i == 2 and engine.cash > 1.0:
            gtc_price = top_ask * 0.98
            qty = (engine.cash * 0.3) / gtc_price
            res3 = engine.submit_order(yes_token, 'BUY', qty, order_type='limit', price=gtc_price, time_in_force='GTC')
            print('  Placed GTC limit buy:', res3)

        engine.process_open_orders()
        print('  account:', engine.account())
        time.sleep(sleep_sec)

    print('Demo complete:', engine.account())


if __name__ == '__main__':
    # Run demo when module executed directly
    paper_trade_demo()


