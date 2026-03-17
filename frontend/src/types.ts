export interface OrderBookEntry {
  price: number
  size: number
}

export interface OrderBook {
  asset_id: string
  bids: OrderBookEntry[]
  asks: OrderBookEntry[]
  timestamp: number
}

export interface PricePoint {
  time: number
  midPrice: number
  bestBid: number
  bestAsk: number
}

export interface PredictionPoint {
  time: number
  predictions: Record<string, number>
}

export interface MarketEvent {
  id: number
  event_type: string
  asset_id?: string
  raw: unknown
  timestamp: number
}

export interface MarketState {
  orderBooks: Record<string, OrderBook>
  priceHistory: PricePoint[]
  predictionHistory: PredictionPoint[]
  latestPredictions: Record<string, number> | null
  events: MarketEvent[]
  status: 'disconnected' | 'connecting' | 'connected' | 'error'
  error?: string
}
