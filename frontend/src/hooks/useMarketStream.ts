import { useState, useEffect, useRef, useCallback } from 'react'
import type { MarketState, OrderBook, PricePoint, MarketEvent, OrderBookEntry } from '../types'

const WS_URL = 'ws://localhost:8000/ws'
const MAX_PRICE_HISTORY = 300
const MAX_EVENTS = 100

function getMidPrice(books: Record<string, OrderBook>): PricePoint | null {
  const book = Object.values(books)[0]
  if (!book || !book.bids.length || !book.asks.length) return null
  const bestBid = book.bids[0].price
  const bestAsk = book.asks[0].price
  return {
    time: Date.now(),
    midPrice: (bestBid + bestAsk) / 2,
    bestBid,
    bestAsk,
  }
}

function parseEntries(raw: { price: string; size: string }[], descending: boolean): OrderBookEntry[] {
  return raw
    .map(e => ({ price: parseFloat(e.price), size: parseFloat(e.size) }))
    .sort((a, b) => descending ? b.price - a.price : a.price - b.price)
}

export function useMarketStream(slug: string | null) {
  const [state, setState] = useState<MarketState>({
    orderBooks: {},
    priceHistory: [],
    events: [],
    status: 'disconnected',
  })

  const wsRef = useRef<WebSocket | null>(null)
  const eventIdRef = useRef(0)
  // Keep a ref to current order books so onmessage handler stays current
  const booksRef = useRef<Record<string, OrderBook>>({})

  const connect = useCallback((targetSlug: string) => {
    wsRef.current?.close()
    booksRef.current = {}

    setState({
      orderBooks: {},
      priceHistory: [],
      events: [],
      status: 'connecting',
    })

    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => {
      setState(s => ({ ...s, status: 'connected' }))
      ws.send(JSON.stringify({ action: 'subscribe', slug: targetSlug }))
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      const messages: Record<string, unknown>[] = Array.isArray(data) ? data : [data]

      setState(prev => {
        const newBooks = { ...booksRef.current }
        const newEvents: MarketEvent[] = [...prev.events]
        let priceUpdated = false

        for (const msg of messages) {
          const eventType = msg.event_type as string ?? 'unknown'
          const assetId = msg.asset_id as string | undefined

          newEvents.unshift({
            id: ++eventIdRef.current,
            event_type: eventType,
            asset_id: assetId,
            raw: msg,
            timestamp: Date.now(),
          })

          if (eventType === 'book' && assetId) {
            newBooks[assetId] = {
              asset_id: assetId,
              bids: parseEntries(msg.bids as { price: string; size: string }[] ?? [], true),
              asks: parseEntries(msg.asks as { price: string; size: string }[] ?? [], false),
              timestamp: Date.now(),
            }
            priceUpdated = true
          } else if (eventType === 'price_change' && assetId && newBooks[assetId]) {
            const existing = newBooks[assetId]
            const bids = [...existing.bids]
            const asks = [...existing.asks]

            for (const change of msg.changes as { price: string; size: string; side: string }[] ?? []) {
              const price = parseFloat(change.price)
              const size = parseFloat(change.size)
              const arr = change.side === 'BUY' ? bids : asks
              const idx = arr.findIndex(e => e.price === price)
              if (size === 0) {
                if (idx >= 0) arr.splice(idx, 1)
              } else {
                if (idx >= 0) arr[idx] = { price, size }
                else arr.push({ price, size })
              }
            }

            newBooks[assetId] = {
              ...existing,
              bids: bids.sort((a, b) => b.price - a.price),
              asks: asks.sort((a, b) => a.price - b.price),
              timestamp: Date.now(),
            }
            priceUpdated = true
          }
        }

        booksRef.current = newBooks

        let newPriceHistory = prev.priceHistory
        if (priceUpdated) {
          const point = getMidPrice(newBooks)
          if (point) {
            newPriceHistory = [...prev.priceHistory, point].slice(-MAX_PRICE_HISTORY)
          }
        }

        return {
          ...prev,
          orderBooks: newBooks,
          priceHistory: newPriceHistory,
          events: newEvents.slice(0, MAX_EVENTS),
        }
      })
    }

    ws.onerror = () => {
      setState(s => ({ ...s, status: 'error', error: 'Connection failed' }))
    }

    ws.onclose = () => {
      setState(s => ({ ...s, status: 'disconnected' }))
    }
  }, [])

  useEffect(() => {
    if (!slug) return
    connect(slug)
    return () => {
      wsRef.current?.close()
    }
  }, [slug, connect])

  return state
}
