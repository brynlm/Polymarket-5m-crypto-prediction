import asyncio
import json
import websockets
import requests
import time
import pandas as pd
import numpy as np
import os

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
GAMMA_API = "https://gamma-api.polymarket.com"
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@aggTrade"

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
# Data formatting helpers
# -----------------------------

def format_book_data_to_dataframe(data):
    bids_snapshot = pd.DataFrame(data['bids'], columns=['price', 'size'], dtype='float64').sort_values('price', ascending=False)
    bids_snapshot['timestamp'] = int(data['timestamp'])
    bids_snapshot['order_type'] = 'bid'
    asks_snapshot = pd.DataFrame(data['asks'], columns=['price', 'size'], dtype='float64')
    asks_snapshot['timestamp'] = int(data['timestamp'])
    asks_snapshot['order_type'] = 'ask'
    # df['price'] = df['price'].astype('float16')
    # df['size'] = df['size'].astype('float16')
    return pd.concat([bids_snapshot, asks_snapshot])

def save_book_market_data(books, slug):
    df = pd.concat(books.values())
    df.to_pickle(f"raw_book_data/book-{slug}.pkl")

def fetch_btc_price_rest():
    """Fetches the current BTC/USD price from Binance REST API."""
    resp = requests.get("https://api.binance.com/api/v3/ticker/price", params={"symbol": "BTCUSDT"})
    return float(resp.json()['price'])


# -----------------------------
# BTC price stream (Binance)
# -----------------------------
async def stream_btc_price():
    """Streams BTC/USD price from Binance and keeps btc_prices updated."""
    global btc_prices
    async with websockets.connect(BINANCE_WS_URL) as ws:
        try:
            while True:
                msg = await ws.recv()
                data = json.loads(msg)
                # aggTrade messages include 'T' (trade time ms) and 'p' (price)
                ts_ms = int(data['T'])
                price = float(data['p'])
                btc_prices[ts_ms] = price
        except asyncio.CancelledError:
            pass
        finally:
            await ws.close()

# -----------------------------
# Connect to websocket
# -----------------------------
async def stream_market_data(token_ids, price_to_beat):
    global book_snapshots
    curr_snapshot = None
    async with websockets.connect(WS_URL, ping_interval=5) as ws:
        # subscription message
        sub_msg = {
            "action": "subscribe",
            'type': "market",
            'assets_ids': [token_ids[0]],  # subscribe only to YES token for simplicity
            'custom_feature_enabled': True
        }
        await ws.send(json.dumps(sub_msg))
        print("Subscribed to:", token_ids)

        try:
            end_of_curr_interval = (int(time.time()) // 300) * 300 + 300
            while True:
                msg = await ws.recv()
                data = json.loads(msg)
                data = data[0] if isinstance(data, list) else data
                ts = int(data['timestamp'])
                print(f"{data['timestamp']}: {data['event_type']} ")
                if data['event_type'] == 'book':
                    curr_snapshot = format_book_data_to_dataframe(data)

                elif data['event_type'] == 'price_change':
                    for prx_change in data['price_changes']:
                        order_type = 'bid' if prx_change['side'] == 'BUY' else 'ask'
                        price = float(prx_change['price'])
                        size = float(prx_change['size'])
                        if size == 0:
                            # Remove price level from snapshot
                            mask = ~((curr_snapshot['price'] == price) & (curr_snapshot['order_type'] == order_type))
                            curr_snapshot = curr_snapshot.loc[mask]
                        else:
                            idx_arr = (curr_snapshot['price'] == price) & (curr_snapshot['order_type'] == order_type)
                            if idx_arr.any():
                                # Update existing price level
                                curr_snapshot.loc[idx_arr, 'size'] = size
                            else:
                                # Add new price level
                                new_row = pd.DataFrame({'price': pd.array([price], dtype='float64'), 'size': pd.array([size], dtype='float64'), 'timestamp': [ts], 'order_type': [order_type]})
                                curr_snapshot = pd.concat([curr_snapshot, new_row], ignore_index=True)
                curr_snapshot['timestamp'] = ts
                snapshot = curr_snapshot.copy()
                # Attach the most recent BTC price at this timestamp
                if btc_prices:
                    nearest_btc_ts = min(btc_prices.keys(), key=lambda t: abs(t - ts))
                    snapshot['btc_price'] = btc_prices[nearest_btc_ts]
                else:
                    snapshot['btc_price'] = float('nan')
                snapshot['price_to_beat'] = price_to_beat
                book_snapshots[ts] = snapshot

                curr_time = int(data['timestamp']) // 1000
                if curr_time >= end_of_curr_interval:
                    print('Switching intervals:')
                    old_slug = f"btc-updown-5m-{end_of_curr_interval - 300}"
                    new_slug = f"btc-updown-5m-{end_of_curr_interval}"
                    new_token_ids = get_token_ids_by_slug(new_slug)
                    # # Re-query closed interval to get priceToBeat (only available after close)
                    # # Retry with backoff since metadata may not be immediately available
                    # for delay in [2, 5, 10, 20]:
                    #     await asyncio.sleep(delay)
                    #     try:
                    #         _, old_response = get_token_ids_by_slug(old_slug, return_res=True)
                    #         ptb = old_response[0]['events'][0]['eventMetadata']['priceToBeat']
                    #         price_to_beat = float(ptb)
                    #         print(f"Price to beat for {old_slug}: {price_to_beat}")
                    #         for snap in book_snapshots.values():
                    #             snap['price_to_beat'] = price_to_beat
                    #         break
                    #     except (KeyError, IndexError, TypeError):
                    #         print(f"priceToBeat not yet available for {old_slug}, retrying...")
                    # else:
                    #     print(f"Warning: could not fetch priceToBeat for {old_slug} after all retries")
                    # Unsubscribe to current market
                    await ws.send(json.dumps({"assets_ids": [token_ids[0]], "operation": "unsubscribe"}))
                    await ws.send(json.dumps({"assets_ids": [new_token_ids[0]], "operation": "subscribe"}))
                    end_of_curr_interval += 300
                    token_ids = new_token_ids
                    save_book_market_data(book_snapshots, old_slug)
                    book_snapshots.clear()
                    btc_prices.clear()
                    btc_prices[int(time.time() * 1000)] = fetch_btc_price_rest()
                    price_to_beat = float('nan')

        except asyncio.CancelledError:
            print("Stream cancelled")
        finally:
            print("Closing websocket...")
            await ws.close()


# -----------------------------
# MAIN
# -----------------------------
async def main():
    # Compute event slug based on current time (for 5m intervals)
    current_time = int(time.time())
    current_interval = (current_time // 300) * 300
    slug = f"btc-updown-5m-{current_interval}"
    token_ids = get_token_ids_by_slug(slug)
    print("Fetched token IDs for market ", slug, ":", token_ids)
    btc_prices[int(time.time() * 1000)] = fetch_btc_price_rest()
    market_task = asyncio.create_task(stream_market_data(token_ids, float('nan')))
    btc_task = asyncio.create_task(stream_btc_price())
    try:
        await asyncio.gather(market_task, btc_task)
    except KeyboardInterrupt:
        print("Keyboard interrupt received")
        market_task.cancel()
        btc_task.cancel()
        await asyncio.gather(market_task, btc_task, return_exceptions=True)
    # finally:
        # # Save book snapshots on close
        # global book_snapshots
        # df_new_data = pd.concat(book_snapshots)
        # df_new_data.to_pickle(f"raw_book_data/book-{slug}.pkl")


if __name__ == "__main__":
    book_snapshots = {}  # {timestamp_ms: snapshot_df}
    btc_prices = {}      # {binance_timestamp_ms: btc_usd_price}
    asyncio.run(main())