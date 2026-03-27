import { useState, useEffect } from 'react'
import { useMarketStream } from './hooks/useMarketStream'
import { PriceChart } from './components/PriceChart'
import { OrderBook } from './components/OrderBook'

const INTERVAL_SECS = 5 * 60
const INTERVAL_MS   = INTERVAL_SECS * 1000

function currentSlug() {
  return `btc-updown-5m-${Math.floor(Date.now() / 1000 / INTERVAL_SECS) * INTERVAL_SECS}`
}

function msUntilNextInterval() {
  return INTERVAL_MS - (Date.now() % INTERVAL_MS)
}

const STATUS_DOT: Record<string, string> = {
  connected: 'bg-green-400',
  connecting: 'bg-yellow-400 animate-pulse',
  disconnected: 'bg-gray-500',
  error: 'bg-red-500',
}

export default function App() {
  const [slug, setSlug] = useState(currentSlug)

  // Auto-switch at each 5-minute interval boundary
  useEffect(() => {
    let timerId: ReturnType<typeof setTimeout>
    function scheduleNext() {
      timerId = setTimeout(() => {
        setSlug(currentSlug())
        scheduleNext()
      }, msUntilNextInterval())
    }
    scheduleNext()
    return () => clearTimeout(timerId)
  }, [])

  const market = useMarketStream(slug)

  const bestBook = (market.upTokenId && market.orderBooks[market.upTokenId]) || Object.values(market.orderBooks)[0]
  const bestBid = bestBook?.bids[0]?.price
  const bestAsk = bestBook?.asks[0]?.price
  const midPrice = bestBid != null && bestAsk != null ? (bestBid + bestAsk) / 2 : null

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-4 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex items-start justify-between mb-5">
        <div>
          <h1 className="text-lg font-bold tracking-tight text-white">Polymarket Dashboard</h1>
          <p className="text-xs text-gray-500 mt-0.5">{slug}</p>
        </div>
        <div className="flex items-center gap-2 mt-1">
          <div className={`w-2 h-2 rounded-full ${STATUS_DOT[market.status]}`} />
          <span className="text-xs text-gray-400 capitalize">{market.status}</span>
        </div>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-3 gap-3 mb-4">
        <div className="bg-gray-900 rounded-lg p-3 border border-gray-800">
          <div className="text-xs text-gray-500 mb-1">Best Bid</div>
          <div className="text-2xl font-mono font-semibold text-green-400">
            {bestBid != null ? bestBid.toFixed(4) : '—'}
          </div>
        </div>
        <div className="bg-gray-900 rounded-lg p-3 border border-gray-800">
          <div className="text-xs text-gray-500 mb-1">Mid Price</div>
          <div className="text-2xl font-mono font-semibold text-blue-400">
            {midPrice != null ? midPrice.toFixed(4) : '—'}
          </div>
        </div>
        <div className="bg-gray-900 rounded-lg p-3 border border-gray-800">
          <div className="text-xs text-gray-500 mb-1">Best Ask</div>
          <div className="text-2xl font-mono font-semibold text-red-400">
            {bestAsk != null ? bestAsk.toFixed(4) : '—'}
          </div>
        </div>
      </div>

      {/* Main grid */}
      <div className="grid grid-cols-3 gap-4">
        <div className="col-span-2 bg-gray-900 rounded-lg p-4 border border-gray-800">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-medium text-gray-300">Price History</h2>
            <span className="text-xs text-gray-600">{market.priceHistory.length} points</span>
          </div>
          <PriceChart data={market.priceHistory} predictionHistory={market.predictionHistory} latestPredictions={market.latestPredictions} />
        </div>

        <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <h2 className="text-sm font-medium text-gray-300 mb-3">Order Book</h2>
          <OrderBook books={market.orderBooks} yesTokenId={market.upTokenId} />
        </div>

      </div>
    </div>
  )
}
