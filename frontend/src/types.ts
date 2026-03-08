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

export interface MarketState {
  orderBooks: Record<string, OrderBook>
  priceHistory: PricePoint[]
  status: 'disconnected' | 'connecting' | 'connected' | 'error'
  currentSlug: string | null
  yesTokenId: string | null
  error?: string
}
