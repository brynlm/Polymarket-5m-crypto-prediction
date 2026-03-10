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

def get_token_ids_by_slug(slug):
    # Gamma API handles market discovery
    url = f"{GAMMA_API}/markets"
    response = requests.get(url, params={'slug': slug}).json()

    # clobTokenIds is a list: index 0 is YES, index 1 is NO
    token_ids = json.loads(response[0]['clobTokenIds'])
    return token_ids

# -----------------------------
# Data formatting helpers
# -----------------------------

def format_book_data_to_dataframe(data):
    bids_snapshot = pd.DataFrame(data['bids'], columns=['price', 'size'], dtype='float32').sort_values('price', ascending=False)
    bids_snapshot['timestamp'] = int(data['timestamp'])
    bids_snapshot['order_type'] = 'bid'
    asks_snapshot = pd.DataFrame(data['asks'], columns=['price', 'size'])
    asks_snapshot['timestamp'] = int(data['timestamp'])
    asks_snapshot['order_type'] = 'ask'
    # df['price'] = df['price'].astype('float16')
    # df['size'] = df['size'].astype('float16')
    return pd.concat([bids_snapshot, asks_snapshot])

def save_book_market_data(books, slug):
    df = pd.concat(books.values())
    df.to_pickle(f"raw_book_data/book-{slug}.pkl")

# -----------------------------
# Connect to websocket
# -----------------------------
async def stream_market_data(token_ids):
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
                    # last_appended_timestamp = int(data['timestamp'])
                    # book_snapshots.append(curr_snapshot.copy())

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
                                new_row = pd.DataFrame({'price': [price], 'size': [size], 'timestamp': [int(data['timestamp'])], 'order_type': [order_type]})
                                curr_snapshot = pd.concat([curr_snapshot, new_row], ignore_index=True)
                    # ts = int(data['timestamp'])
                curr_snapshot['timestamp'] = ts
                book_snapshots[ts] = curr_snapshot.copy()

                curr_time = int(data['timestamp']) // 1000
                if curr_time >= end_of_curr_interval:
                    print('Switching intervals:')
                    new_slug = f"btc-updown-5m-{end_of_curr_interval}"
                    new_token_ids = get_token_ids_by_slug(new_slug)
                    # Unsubscribe to current market
                    await ws.send(json.dumps({"assets_ids": [token_ids[0]], "operation": "unsubscribe"}))
                    await ws.send(json.dumps({"assets_ids": [new_token_ids[0]], "operation": "subscribe"}))
                    end_of_curr_interval += 300
                    token_ids = new_token_ids
                    save_book_market_data(book_snapshots, f"btc-updown-5m-{end_of_curr_interval-300}")
                    book_snapshots.clear() # Clear old snapshots for old market

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
    task = asyncio.create_task(stream_market_data(token_ids))
    try:
        await task
    except KeyboardInterrupt:
        print("Keyboard interrupt received")
        task.cancel()
        await task
    # finally:
        # # Save book snapshots on close
        # global book_snapshots
        # df_new_data = pd.concat(book_snapshots)
        # df_new_data.to_pickle(f"raw_book_data/book-{slug}.pkl")


if __name__ == "__main__":
    book_snapshots = {}  # {timestamp_ms: snapshot_df}
    asyncio.run(main())