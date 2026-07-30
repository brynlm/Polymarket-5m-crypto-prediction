import type { OrderBook, OrderBookEntry } from '../types'

interface Props {
  books: Record<string, OrderBook>
  yesTokenId?: string | null
  tokenLabels?: Record<string, string>
}

function DepthRow({ entry, maxSize, side }: { entry: OrderBookEntry; maxSize: number; side: 'bid' | 'ask' }) {
  const pct = Math.min((entry.size / maxSize) * 100, 100)
  const isBid = side === 'bid'

  return (
    <div className="relative flex items-center px-2 py-0.5 text-xs">
      <div
        className={`absolute top-0 bottom-0 opacity-10 ${isBid ? 'bg-green-500 right-0' : 'bg-red-500 left-0'}`}
        style={{ width: `${pct}%` }}
      />
      <span className={`relative flex-1 font-mono ${isBid ? 'text-green-400' : 'text-red-400'}`}>
        {entry.price.toFixed(3)}
      </span>
      <span className="relative text-gray-400 font-mono">{entry.size.toFixed(1)}</span>
    </div>
  )
}

export function OrderBook({ books, yesTokenId, tokenLabels = {} }: Props) {
  // Only show the YES token. If yesTokenId is known but the book hasn't arrived yet
  // (e.g. NO event arrived first after a market switch), wait rather than show wrong data.
  const book = yesTokenId ? books[yesTokenId] : Object.values(books)[0]
  const assetId = yesTokenId ?? Object.keys(books)[0] ?? ''
  const label = tokenLabels[assetId] ?? 'YES token'

  if (!book) {
    return (
      <div className="flex items-center justify-center h-32 text-gray-500 text-sm">
        Waiting for order book...
      </div>
    )
  }
  const topBids = book.bids.slice(0, 8)
  const topAsks = book.asks.slice(0, 8)
  const maxBidSize = Math.max(...topBids.map(e => e.size), 1)
  const maxAskSize = Math.max(...topAsks.map(e => e.size), 1)

  const spread =
    book.bids.length > 0 && book.asks.length > 0
      ? book.asks[0].price - book.bids[0].price
      : null

  return (
    <div>
      <div className="text-xs text-gray-500 mb-2 truncate" title={assetId}>
        {label}
      </div>

      <div className="grid grid-cols-2 gap-2">
        <div>
          <div className="flex justify-between px-2 mb-1 text-xs text-gray-500">
            <span>Price</span>
            <span>Size</span>
          </div>
          {topBids.map((e, i) => (
            <DepthRow key={i} entry={e} maxSize={maxBidSize} side="bid" />
          ))}
        </div>
        <div>
          <div className="flex justify-between px-2 mb-1 text-xs text-gray-500">
            <span>Price</span>
            <span>Size</span>
          </div>
          {topAsks.map((e, i) => (
            <DepthRow key={i} entry={e} maxSize={maxAskSize} side="ask" />
          ))}
        </div>
      </div>

      {spread !== null && (
        <div className="mt-2 px-2 text-xs text-gray-500">
          Spread: <span className="text-gray-300">{spread.toFixed(4)}</span>
        </div>
      )}
    </div>
  )
}
