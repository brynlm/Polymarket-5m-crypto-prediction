import { useState, useEffect, useRef, useCallback } from 'react'
import type { MarketState, OrderBook, PricePoint, PredictionPoint, OrderBookEntry } from '../types'

const WS_URL            = 'ws://localhost:8000/ws'
const MAX_PRICE_HISTORY = 300
const MAX_PRED_HISTORY  = 300
const FLUSH_INTERVAL_MS = 1000   // flush refs → React state at 1 Hz

function getMidPrice(books: Record<string, OrderBook>): PricePoint | null {
  const book = Object.values(books)[0]
  if (!book || !book.bids.length || !book.asks.length) return null
  const bestBid = book.bids[0].price
  const bestAsk = book.asks[0].price
  return { time: Date.now(), midPrice: (bestBid + bestAsk) / 2, bestBid, bestAsk }
}

function parseEntries(raw: { price: string; size: string }[], descending: boolean): OrderBookEntry[] {
  return raw
    .map(e => ({ price: parseFloat(e.price), size: parseFloat(e.size) }))
    .sort((a, b) => descending ? b.price - a.price : a.price - b.price)
}

const INITIAL_STATE: MarketState = {
  orderBooks: {}, priceHistory: [], predictionHistory: [],
  latestPredictions: null, status: 'disconnected',
}

export function useMarketStream(slug: string | null) {
  const [state, setState] = useState<MarketState>(INITIAL_STATE)

  const wsRef = useRef<WebSocket | null>(null)

  // All mutable data lives in refs — onmessage writes here, interval flushes to React state
  const booksRef       = useRef<Record<string, OrderBook>>({})
  const priceHistRef   = useRef<PricePoint[]>([])
  const predHistRef    = useRef<PredictionPoint[]>([])
  const latestPredsRef = useRef<Record<string, number> | null>(null)
  const statusRef      = useRef<MarketState['status']>('disconnected')

  // 1-second flush: copy refs into React state (the only setState call in the hot path)
  useEffect(() => {
    const timer = setInterval(() => {
      setState({
        orderBooks:        booksRef.current,
        priceHistory:      priceHistRef.current,
        predictionHistory: predHistRef.current,
        latestPredictions: latestPredsRef.current,
        status:            statusRef.current,
      })
    }, FLUSH_INTERVAL_MS)
    return () => clearInterval(timer)
  }, [])

  const connect = useCallback((targetSlug: string) => {
    wsRef.current?.close()

    // Reset all refs and immediately show connecting state
    booksRef.current       = {}
    priceHistRef.current   = []
    predHistRef.current    = []
    latestPredsRef.current = null
    statusRef.current      = 'connecting'
    setState({ ...INITIAL_STATE, status: 'connecting' })

    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => {
      statusRef.current = 'connected'
      ws.send(JSON.stringify({ action: 'subscribe', slug: targetSlug }))
    }

    // No setState here — just update refs
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      const messages: Record<string, unknown>[] = Array.isArray(data) ? data : [data]

      for (const msg of messages) {
        const eventType = (msg.event_type as string) ?? 'unknown'
        const assetId   = msg.asset_id as string | undefined

        if (eventType === 'prediction') {
          const predPoint: PredictionPoint = {
            time:        msg.timestamp as number,
            predictions: msg.predictions as Record<string, number>,
          }
          predHistRef.current    = [...predHistRef.current, predPoint].slice(-MAX_PRED_HISTORY)
          latestPredsRef.current = predPoint.predictions
          continue
        }

        let priceUpdated = false

        if (eventType === 'book' && assetId) {
          booksRef.current = {
            ...booksRef.current,
            [assetId]: {
              asset_id:  assetId,
              bids:      parseEntries(msg.bids as { price: string; size: string }[] ?? [], true),
              asks:      parseEntries(msg.asks as { price: string; size: string }[] ?? [], false),
              timestamp: Date.now(),
            },
          }
          priceUpdated = true
        } else if (eventType === 'price_change' && assetId && booksRef.current[assetId]) {
          const existing = booksRef.current[assetId]
          const bids = [...existing.bids]
          const asks = [...existing.asks]

          for (const change of msg.changes as { price: string; size: string; side: string }[] ?? []) {
            const price = parseFloat(change.price)
            const size  = parseFloat(change.size)
            const arr   = change.side === 'BUY' ? bids : asks
            const idx   = arr.findIndex(e => e.price === price)
            if (size === 0) {
              if (idx >= 0) arr.splice(idx, 1)
            } else {
              if (idx >= 0) arr[idx] = { price, size }
              else arr.push({ price, size })
            }
          }

          booksRef.current = {
            ...booksRef.current,
            [assetId]: {
              ...existing,
              bids: bids.sort((a, b) => b.price - a.price),
              asks: asks.sort((a, b) => a.price - b.price),
              timestamp: Date.now(),
            },
          }
          priceUpdated = true
        }

        if (priceUpdated) {
          const point = getMidPrice(booksRef.current)
          if (point) {
            priceHistRef.current = [...priceHistRef.current, point].slice(-MAX_PRICE_HISTORY)
          }
        }
      }
    }

    ws.onerror = () => { statusRef.current = 'error' }
    ws.onclose = () => { statusRef.current = 'disconnected' }
  }, [])

  useEffect(() => {
    if (!slug) return
    connect(slug)
    return () => { wsRef.current?.close() }
  }, [slug, connect])

  return state
}
