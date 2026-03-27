import asyncio
import json
import websockets
import requests
import time
from datetime import datetime, timezone
import os
import asyncpg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
GAMMA_API = "https://gamma-api.polymarket.com"

N_LEVELS = 5
orderbooks: dict = {}       # {'UP': {'bids': {price: size}, 'asks': {price: size}}, ...}
snapshot_buffer: list = []  # feature snapshot dicts ready for DB insert
last_ts_s: int = 0          # last integer second that was snapshotted


def apply_update(market: str, event_type: str, data: dict) -> None:
    if event_type == 'book':
        orderbooks[market] = {
            'bids': {float(r['price']): float(r['size']) for r in data.get('bids', [])},
            'asks': {float(r['price']): float(r['size']) for r in data.get('asks', [])},
        }
    elif event_type == 'price_change':
        if market not in orderbooks:
            return
        for ch in data.get('price_changes', []):
            side = 'bids' if ch['side'] == 'BUY' else 'asks'
            price, size = float(ch['price']), float(ch['size'])
            if size == 0.0:
                orderbooks[market][side].pop(price, None)
            else:
                orderbooks[market][side][price] = size


def compute_features(ts: datetime, market: str) -> dict | None:
    book = orderbooks.get(market)
    if not book:
        return None
    bids = sorted(book['bids'].items(), key=lambda x: -x[0])
    asks = sorted(book['asks'].items(), key=lambda x:  x[0])
    if not bids or not asks:
        return None

    bid_vol_all = sum(sz for _, sz in bids)
    ask_vol_all = sum(sz for _, sz in asks)
    vwap = (sum(p * s for p, s in bids) + sum(p * s for p, s in asks)) / (bid_vol_all + ask_vol_all + 1e-9)

    row = {
        'timestamp':    ts,
        'market':       market,
        'best_bid':     bids[0][0],
        'best_ask':     asks[0][0],
        'bid_vol_all':  bid_vol_all,
        'ask_vol_all':  ask_vol_all,
        'bid_n_levels': len(bids),
        'ask_n_levels': len(asks),
        'vwap':         vwap,
    }
    for i in range(N_LEVELS):
        row[f'bid_size_l{i+1}'] = bids[i][1] if i < len(bids) else None
        row[f'ask_size_l{i+1}'] = asks[i][1] if i < len(asks) else None
    return row


async def flush(pool: asyncpg.Pool):
    global snapshot_buffer
    if not snapshot_buffer:
        return
    rows = snapshot_buffer.copy()
    snapshot_buffer.clear()
    try:
        async with pool.acquire() as conn:
            await conn.executemany("""
                INSERT INTO orderbook_features (
                    timestamp, market,
                    best_bid, best_ask,
                    bid_size_l1, bid_size_l2, bid_size_l3, bid_size_l4, bid_size_l5,
                    ask_size_l1, ask_size_l2, ask_size_l3, ask_size_l4, ask_size_l5,
                    bid_vol_all, ask_vol_all, bid_n_levels, ask_n_levels, vwap
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19)
                ON CONFLICT (timestamp, market) DO UPDATE SET
                    best_bid     = EXCLUDED.best_bid,
                    best_ask     = EXCLUDED.best_ask,
                    bid_size_l1  = EXCLUDED.bid_size_l1,
                    bid_size_l2  = EXCLUDED.bid_size_l2,
                    bid_size_l3  = EXCLUDED.bid_size_l3,
                    bid_size_l4  = EXCLUDED.bid_size_l4,
                    bid_size_l5  = EXCLUDED.bid_size_l5,
                    ask_size_l1  = EXCLUDED.ask_size_l1,
                    ask_size_l2  = EXCLUDED.ask_size_l2,
                    ask_size_l3  = EXCLUDED.ask_size_l3,
                    ask_size_l4  = EXCLUDED.ask_size_l4,
                    ask_size_l5  = EXCLUDED.ask_size_l5,
                    bid_vol_all  = EXCLUDED.bid_vol_all,
                    ask_vol_all  = EXCLUDED.ask_vol_all,
                    bid_n_levels = EXCLUDED.bid_n_levels,
                    ask_n_levels = EXCLUDED.ask_n_levels,
                    vwap         = EXCLUDED.vwap
            """, [(
                r['timestamp'], r['market'],
                r['best_bid'], r['best_ask'],
                r['bid_size_l1'], r['bid_size_l2'], r['bid_size_l3'],
                r['bid_size_l4'], r['bid_size_l5'],
                r['ask_size_l1'], r['ask_size_l2'], r['ask_size_l3'],
                r['ask_size_l4'], r['ask_size_l5'],
                r['bid_vol_all'], r['ask_vol_all'],
                r['bid_n_levels'], r['ask_n_levels'],
                r['vwap'],
            ) for r in rows])
    except Exception:
        snapshot_buffer[:0] = rows  # restore rows to front of buffer on failure
        raise


async def periodic_flush(pool: asyncpg.Pool, interval: int = 5):
    while True:
        await asyncio.sleep(interval)
        await flush(pool)


def get_token_ids_by_slug(slug, return_res=False):
    # Gamma API handles market discovery
    url = f"{GAMMA_API}/markets"
    response = requests.get(url, params={'slug': slug}).json()
    # clobTokenIds is a list: index 0 is YES, index 1 is NO
    token_ids = json.loads(response[0]['clobTokenIds'])
    if return_res:
        return token_ids, response
    return token_ids


# -----------------------------
# Connect to websocket
# -----------------------------
async def heartbeat(ws, interval=10):
    while True:
        await asyncio.sleep(interval)
        await ws.send("PING")


async def stream_market_data(token_ids, pool: asyncpg.Pool):
    global orderbooks, snapshot_buffer, last_ts_s
    end_of_curr_interval = (int(time.time()) // 300) * 300 + 300
    asset_id_token_map = {token_ids[0]: 'UP', token_ids[1]: 'DOWN'}

    while True:
        try:
            async with websockets.connect(WS_URL, ping_interval=None) as ws:
                sub_msg = {
                    "action": "subscribe",
                    'type': "market",
                    'assets_ids': token_ids,
                    'custom_feature_enabled': True
                }
                await ws.send(json.dumps(sub_msg))
                print("Subscribed to:", token_ids)
                hb_task = asyncio.create_task(heartbeat(ws))

                try:
                    while True:
                        msg = await ws.recv()
                        if msg == "PONG":
                            continue
                        data = json.loads(msg)
                        data = data[0] if isinstance(data, list) else data
                        print(f"{data['timestamp']}: {data['event_type']} ")

                        curr_ts_s = int(data['timestamp']) // 1000

                        # Snapshot end-of-second state BEFORE applying current message
                        if last_ts_s != 0 and curr_ts_s > last_ts_s:
                            snap_ts = datetime.fromtimestamp(last_ts_s, tz=timezone.utc)
                            for mkt in list(asset_id_token_map.values()):
                                row = compute_features(snap_ts, mkt)
                                if row is not None:
                                    snapshot_buffer.append(row)
                        last_ts_s = curr_ts_s

                        # Apply update to in-memory orderbook
                        if data['event_type'] == 'book':
                            if data['asset_id'] in asset_id_token_map:
                                apply_update(asset_id_token_map[data['asset_id']], 'book', data)
                        elif data['event_type'] == 'price_change':
                            for ch in data.get('price_changes', []):
                                if ch['asset_id'] in asset_id_token_map:
                                    apply_update(asset_id_token_map[ch['asset_id']], 'price_change', {'price_changes': [ch]})

                        if curr_ts_s >= end_of_curr_interval:
                            print('Switching intervals:')
                            t = asyncio.create_task(flush(pool))
                            t.add_done_callback(lambda t: print(f"Interval flush error: {t.exception()}") if not t.cancelled() and t.exception() else None)
                            new_slug = f"btc-updown-5m-{end_of_curr_interval}"
                            new_token_ids = get_token_ids_by_slug(new_slug)
                            # Unsubscribe from current market
                            await ws.send(json.dumps({"assets_ids": token_ids, "operation": "unsubscribe"}))
                            await ws.send(json.dumps({"assets_ids": new_token_ids, "operation": "subscribe"}))
                            # Clear stale orderbook state for the new market
                            orderbooks.clear()
                            last_ts_s = 0
                            # Update interval + token_ids
                            end_of_curr_interval += 300
                            token_ids = new_token_ids
                            asset_id_token_map = dict(zip(token_ids, ['UP', 'DOWN']))

                except asyncio.CancelledError:
                    print("Stream cancelled")
                    # Flush the current partial second before closing
                    if last_ts_s != 0:
                        snap_ts = datetime.fromtimestamp(last_ts_s, tz=timezone.utc)
                        for mkt in list(asset_id_token_map.values()):
                            row = compute_features(snap_ts, mkt)
                            if row is not None:
                                snapshot_buffer.append(row)
                    try:
                        await asyncio.shield(flush(pool))
                    except (asyncio.CancelledError, Exception) as e:
                        print(f"Final flush error: {e}")
                    raise
                finally:
                    hb_task.cancel()
                    await asyncio.gather(hb_task, return_exceptions=True)

        except websockets.exceptions.ConnectionClosedError as e:
            print(f"WebSocket connection closed unexpectedly: {e}, reconnecting in 2s...")
            orderbooks.clear()
            last_ts_s = 0
            await asyncio.sleep(2)


async def main():
    pool = await asyncpg.create_pool(DATABASE_URL, ssl='require')
    # Compute event slug based on current time (for 5m intervals)
    current_time = int(time.time())
    current_interval = (current_time // 300) * 300
    slug = f"btc-updown-5m-{current_interval}"
    token_ids = get_token_ids_by_slug(slug)
    print("Fetched token IDs for market ", slug, ":", token_ids)
    market_task = asyncio.create_task(stream_market_data(token_ids, pool))
    flush_task = asyncio.create_task(periodic_flush(pool))
    try:
        await asyncio.gather(market_task)
    finally:
        flush_task.cancel()
        await asyncio.gather(flush_task, return_exceptions=True)
        await pool.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Keyboard interrupt received")
