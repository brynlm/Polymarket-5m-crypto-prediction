import asyncio
import json
import time
import logging
from typing import Optional

import websockets
import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Polymarket Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

POLYMARKET_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
GAMMA_API = "https://gamma-api.polymarket.com"

# Shared state
active_clients: set[WebSocket] = set()
stream_task: Optional[asyncio.Task] = None
monitor_task: Optional[asyncio.Task] = None
current_token_ids: list[str] = []
current_slug: Optional[str] = None


def get_token_ids(slug: str) -> tuple[list[str], dict]:
    try:
        resp = requests.get(f"{GAMMA_API}/markets", params={"slug": slug}, timeout=5).json()
        if not resp:
            return [], {}
        market = resp[0]
        token_ids = json.loads(market["clobTokenIds"])
        return token_ids, market
    except Exception as e:
        logger.error(f"Error fetching market {slug}: {e}")
        return [], {}


def is_5m_slug(slug: str) -> bool:
    return slug.startswith("btc-updown-5m-")


async def broadcast(message: str):
    dead = set()
    for client in active_clients:
        try:
            await client.send_text(message)
        except Exception:
            dead.add(client)
    active_clients.difference_update(dead)


async def run_polymarket_stream(token_ids: list[str]):
    logger.info(f"Starting Polymarket stream for {len(token_ids)} tokens")
    try:
        async with websockets.connect(POLYMARKET_WS, ping_interval=5) as ws:
            sub_msg = {
                "action": "subscribe",
                "type": "market",
                "assets_ids": [token_ids[0]],  # Subscribe to only YES token market data for simplicity (hot fix, needs to be refactored later on)
                "custom_feature_enabled": True,
            }
            await ws.send(json.dumps(sub_msg))
            while True:
                msg = await ws.recv()
                await broadcast(msg)
    except asyncio.CancelledError:
        logger.info("Stream cancelled")
    except Exception as e:
        logger.error(f"Stream error: {e}")
        await broadcast(json.dumps({"event_type": "error", "message": str(e)}))


async def monitor_5m_market():
    """Sleep until the next 5-minute boundary, then auto-switch to the new market."""
    global stream_task, current_token_ids, current_slug

    while active_clients:
        # Derive the next boundary from the tracked slug's timestamp, not time.time().
        # This ensures the next slug is always exactly +300s from the one we're on.
        try:
            slug_time = int(current_slug.split("-")[-1])  # type: ignore[union-attr]
            next_boundary = slug_time + 300
        except (ValueError, AttributeError, TypeError):
            next_boundary = ((int(time.time()) // 300) + 1) * 300

        sleep_for = max(next_boundary - time.time() + 3, 1)
        logger.info(f"5m monitor: sleeping {sleep_for:.1f}s until {next_boundary}")
        await asyncio.sleep(sleep_for)

        new_slug = f"btc-updown-5m-{next_boundary}"

        # Retry fetching token IDs — new market can take up to ~60s to appear
        token_ids: list[str] = []
        for attempt in range(12):
            token_ids, _ = get_token_ids(new_slug)
            if token_ids:
                break
            logger.info(f"Waiting for {new_slug} (attempt {attempt + 1}/12)...")
            await asyncio.sleep(10)

        if not token_ids:
            logger.warning(f"Could not find new 5m market after retries: {new_slug}")
            continue

        logger.info(f"Auto-switching stream to {new_slug}")

        if stream_task and not stream_task.done():
            stream_task.cancel()
            await asyncio.sleep(0.1)

        current_token_ids = token_ids
        current_slug = new_slug

        # Broadcast market_changed BEFORE starting the new stream so the frontend
        # resets its state before any book events from the new market arrive.
        await broadcast(json.dumps({
            "event_type": "market_changed",
            "slug": new_slug,
            "token_ids": token_ids,
        }))

        stream_task = asyncio.create_task(run_polymarket_stream(token_ids))


@app.get("/api/market/{slug}")
def get_market(slug: str):
    token_ids, market = get_token_ids(slug)
    if not token_ids:
        return {"error": "Market not found"}
    return {
        "slug": slug,
        "token_ids": token_ids,
        "question": market.get("question"),
        "end_date": market.get("endDate"),
    }


@app.get("/api/markets/active")
def get_active_markets():
    """Detect currently active BTC updown markets (5m and 15m intervals)."""
    current_time = int(time.time())
    markets = []
    for interval, label in [(300, "5m"), (900, "15m")]:
        slug_time = (current_time // interval) * interval
        slug = f"btc-updown-{label}-{slug_time}"
        token_ids, market = get_token_ids(slug)
        if token_ids:
            markets.append({
                "slug": slug,
                "label": f"BTC {label}",
                "token_ids": token_ids,
                "question": market.get("question"),
            })
    return markets


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global stream_task, monitor_task, current_token_ids, current_slug
    await websocket.accept()
    active_clients.add(websocket)
    logger.info(f"Client connected. Total: {len(active_clients)}")

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg.get("action") == "subscribe":
                slug = msg.get("slug", "").strip()
                if not slug:
                    continue

                token_ids, _ = get_token_ids(slug)
                if not token_ids:
                    await websocket.send_text(json.dumps({
                        "event_type": "error",
                        "message": f"Market not found: {slug}",
                    }))
                    continue

                # Restart stream if switching to a different market
                if token_ids != current_token_ids:
                    if stream_task and not stream_task.done():
                        stream_task.cancel()
                        await asyncio.sleep(0.1)
                    current_token_ids = token_ids
                    current_slug = slug
                    stream_task = asyncio.create_task(run_polymarket_stream(token_ids))

                elif not stream_task or stream_task.done():
                    stream_task = asyncio.create_task(run_polymarket_stream(token_ids))

                # Start the 5m monitor if subscribing to a 5m market and not already running
                if is_5m_slug(slug) and (not monitor_task or monitor_task.done()):
                    monitor_task = asyncio.create_task(monitor_5m_market())

                await websocket.send_text(json.dumps({
                    "event_type": "subscribed",
                    "slug": slug,
                    "token_ids": token_ids,
                }))

    except WebSocketDisconnect:
        active_clients.discard(websocket)
        logger.info(f"Client disconnected. Total: {len(active_clients)}")
        if not active_clients:
            if stream_task and not stream_task.done():
                stream_task.cancel()
            if monitor_task and not monitor_task.done():
                monitor_task.cancel()
            current_token_ids = []
            current_slug = None
