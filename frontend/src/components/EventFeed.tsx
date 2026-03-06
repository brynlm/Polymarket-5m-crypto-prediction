import type { MarketEvent } from '../types'

interface Props {
  events: MarketEvent[]
}

const EVENT_COLORS: Record<string, string> = {
  book: 'text-blue-400',
  price_change: 'text-yellow-400',
  last_trade_price: 'text-green-400',
  tick_size_change: 'text-purple-400',
  subscribed: 'text-cyan-400',
  error: 'text-red-400',
}

export function EventFeed({ events }: Props) {
  return (
    <div className="overflow-y-auto h-48 space-y-px font-mono text-xs">
      {events.length === 0 ? (
        <div className="text-gray-500 text-center py-4">No events yet...</div>
      ) : (
        events.map(event => (
          <div
            key={event.id}
            className="flex items-center gap-3 px-2 py-0.5 hover:bg-gray-800 rounded"
          >
            <span className="text-gray-600 shrink-0 w-20">
              {new Date(event.timestamp).toLocaleTimeString([], {
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit',
              })}
            </span>
            <span className={`shrink-0 w-28 ${EVENT_COLORS[event.event_type] ?? 'text-gray-400'}`}>
              {event.event_type}
            </span>
            <span className="text-gray-500 truncate">
              {event.asset_id ? `${event.asset_id.slice(0, 12)}...` : ''}
            </span>
          </div>
        ))
      )}
    </div>
  )
}
