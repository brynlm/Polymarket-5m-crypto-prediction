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
  predictions: Record<string, Record<string, number>>  // { UP: { q10, q50, q90, mid }, DOWN: { ... } }
}

export interface MarketState {
  orderBooks: Record<string, OrderBook>
  upTokenId: string | null
  priceHistory: PricePoint[]
  predictionHistory: PredictionPoint[]
  latestPredictions: Record<string, Record<string, number>> | null
  status: 'disconnected' | 'connecting' | 'connected' | 'error'
  error?: string
}
