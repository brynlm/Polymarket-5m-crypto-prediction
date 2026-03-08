import { useState, useEffect } from 'react'
import { useMarketStream } from './hooks/useMarketStream'
import { PriceChart } from './components/PriceChart'
import { OrderBook } from './components/OrderBook'

const API_URL = 'http://localhost:8000'

interface ActiveMarket {
  slug: string
  label: string
  question?: string
}

const STATUS_DOT: Record<string, string> = {
  connected: 'bg-green-400',
  connecting: 'bg-yellow-400 animate-pulse',
  disconnected: 'bg-gray-500',
  error: 'bg-red-500',
}

export default function App() {
  const [activeMarkets, setActiveMarkets] = useState<ActiveMarket[]>([])
  const [selectedSlug, setSelectedSlug] = useState<string | null>(null)
  const [customSlug, setCustomSlug] = useState('')

  const market = useMarketStream(selectedSlug)

  useEffect(() => {
    fetch(`${API_URL}/api/markets/active`)
      .then(r => r.json())
      .then((data: ActiveMarket[]) => {
        setActiveMarkets(data)
        if (data.length > 0 && !selectedSlug) {
          setSelectedSlug(data[0].slug)
        }
      })
      .catch(console.error)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  function handleCustomSlug() {
    const slug = customSlug.trim()
    if (slug) {
      setSelectedSlug(slug)
      setCustomSlug('')
    }
  }

  const bestBook = Object.values(market.orderBooks)[0]
  const bestBid = bestBook?.bids[0]?.price
  const bestAsk = bestBook?.asks[0]?.price
  const midPrice = bestBid != null && bestAsk != null ? (bestBid + bestAsk) / 2 : null

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-4 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex items-start justify-between mb-5">
        <div>
          <h1 className="text-lg font-bold tracking-tight text-white">Polymarket Dashboard</h1>
          <p className="text-xs text-gray-500 mt-0.5">{selectedSlug ?? 'No market selected'}</p>
        </div>
        <div className="flex items-center gap-2 mt-1">
          <div className={`w-2 h-2 rounded-full ${STATUS_DOT[market.status]}`} />
          <span className="text-xs text-gray-400 capitalize">{market.status}</span>
        </div>
      </div>

      {/* Market selector */}
      <div className="flex flex-wrap gap-2 mb-5">
        {activeMarkets.map(m => (
          <button
            key={m.slug}
            onClick={() => setSelectedSlug(m.slug)}
            className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
              selectedSlug === m.slug
                ? 'bg-blue-600 text-white'
                : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
            }`}
          >
            {m.label}
          </button>
        ))}
        <div className="flex gap-1.5">
          <input
            type="text"
            placeholder="Custom slug..."
            value={customSlug}
            onChange={e => setCustomSlug(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleCustomSlug()}
            className="px-3 py-1.5 bg-gray-800 border border-gray-700 rounded text-xs text-gray-300 placeholder-gray-600 focus:outline-none focus:border-blue-500 w-48"
          />
          <button
            onClick={handleCustomSlug}
            className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-xs transition-colors"
          >
            Go
          </button>
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
          <PriceChart data={market.priceHistory} />
        </div>

        <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <h2 className="text-sm font-medium text-gray-300 mb-3">Order Book</h2>
          <OrderBook books={market.orderBooks} />
        </div>

      </div>
    </div>
  )
}
