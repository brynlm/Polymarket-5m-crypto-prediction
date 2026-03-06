import asyncio
import json
import websockets
import requests
import time

# -----------------------------
# CONFIG
# -----------------------------
EVENT_SLUG = "btc-updown-15m-1772757900"  # change to your event
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
# STEP 2: connect to websocket
# -----------------------------
async def stream_market_data(token_ids):
    async with websockets.connect(WS_URL, ping_interval=5) as ws:
        # subscription message

        sub_msg = {
            "action": "subscribe",
            'type': "market",
            'assets_ids': token_ids,
            'custom_feature_enabled': True
        }

        await ws.send(json.dumps(sub_msg))
        print("Subscribed to:", token_ids)

        try:
            while True:
                msg = await ws.recv()
                data = json.loads(msg)
                print(data)
                # if isinstance(data, dict):
                #     print(data['event_type'])
                # else:
                #     print(data)

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
    print("Found tokens:", token_ids)
    task = asyncio.create_task(stream_market_data(token_ids))
    try:
        await task
    except KeyboardInterrupt:
        print("Keyboard interrupt received")
        task.cancel()
        await task


if __name__ == "__main__":
    asyncio.run(main())