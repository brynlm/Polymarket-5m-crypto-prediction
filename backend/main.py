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
current_token_ids: list[str] = []


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
                "assets_ids": token_ids,
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
    global stream_task, current_token_ids
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
                    stream_task = asyncio.create_task(run_polymarket_stream(token_ids))

                elif not stream_task or stream_task.done():
                    stream_task = asyncio.create_task(run_polymarket_stream(token_ids))

                await websocket.send_text(json.dumps({
                    "event_type": "subscribed",
                    "slug": slug,
                    "token_ids": token_ids,
                }))

    except WebSocketDisconnect:
        active_clients.discard(websocket)
        logger.info(f"Client disconnected. Total: {len(active_clients)}")
        if not active_clients and stream_task and not stream_task.done():
            stream_task.cancel()
            current_token_ids = []
