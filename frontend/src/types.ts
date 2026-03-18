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

export interface SimulationPortfolio {
  cash: number
  positions: Record<string, number>
  realized_pnl: number
  unrealized_pnl: number
  total_value: number
}

export interface SimulationFill {
  asset_id: string
  side: 'BUY' | 'SELL'
  filled_qty: number
  avg_price: number
  fee: number
  status: string
  timestamp: number
}

export interface SimulationState {
  portfolio: SimulationPortfolio
  latestFills: SimulationFill[]
  pnl: number
}

export interface MarketState {
  orderBooks: Record<string, OrderBook>
  priceHistory: PricePoint[]
  predictionHistory: PredictionPoint[]
  latestPredictions: Record<string, number> | null
  simulation: SimulationState | null
  status: 'disconnected' | 'connecting' | 'connected' | 'error'
  error?: string
}
