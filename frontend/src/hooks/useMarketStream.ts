import { useState, useEffect, useRef, useCallback } from 'react'
import type { MarketState, OrderBook, PricePoint, PredictionPoint, MarketEvent, OrderBookEntry } from '../types'

const WS_URL = 'ws://localhost:8000/ws'
const MAX_PRICE_HISTORY  = 300
const MAX_PRED_HISTORY   = 300
const MAX_EVENTS         = 100
const PRED_HORIZON_MS    = 5000  // target_5s prediction horizon

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
    predictionHistory: [],
    latestPredictions: null,
    events: [],
    status: 'disconnected',
  })

  const wsRef = useRef<WebSocket | null>(null)
  const eventIdRef = useRef(0)
  const booksRef = useRef<Record<string, OrderBook>>({})
  const predHistoryRef = useRef<PredictionPoint[]>([])

  const connect = useCallback((targetSlug: string) => {
    wsRef.current?.close()
    booksRef.current = {}
    predHistoryRef.current = []

    setState({
      orderBooks: {},
      priceHistory: [],
      predictionHistory: [],
      latestPredictions: null,
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
        let newPredHistory = prev.predictionHistory
        let newLatestPreds = prev.latestPredictions

        for (const msg of messages) {
          const eventType = msg.event_type as string ?? 'unknown'
          const assetId = msg.asset_id as string | undefined

          // Handle prediction events separately — don't add to event feed
          if (eventType === 'prediction') {
            const predPoint: PredictionPoint = {
              time: msg.timestamp as number,
              predictions: msg.predictions as Record<string, number>,
            }
            predHistoryRef.current = [
              ...predHistoryRef.current,
              predPoint,
            ].slice(-MAX_PRED_HISTORY)
            newPredHistory = predHistoryRef.current
            newLatestPreds = predPoint.predictions
            continue
          }

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
              bids: bids.sort((b1, b2) => b2.price - b1.price),
              asks: asks.sort((a1, a2) => a1.price - a2.price),
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
            // Find prediction made ~PRED_HORIZON_MS ago to compare against this moment
            const targetTime = point.time - PRED_HORIZON_MS
            const match = predHistoryRef.current.find(
              p => Math.abs(p.time - targetTime) < 1500
            )
            const enriched: PricePoint = match
              ? {
                  ...point,
                  predQ10: match.predictions['q10'],
                  predQ50: match.predictions['q50'],
                  predQ90: match.predictions['q90'],
                }
              : point
            newPriceHistory = [...prev.priceHistory, enriched].slice(-MAX_PRICE_HISTORY)
          }
        }

        return {
          ...prev,
          orderBooks: newBooks,
          predictionHistory: newPredHistory,
          latestPredictions: newLatestPreds,
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
