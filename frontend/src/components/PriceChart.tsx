import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'
import type { PricePoint } from '../types'

interface Props {
  data: PricePoint[]
}

function formatTime(ts: number) {
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

export function PriceChart({ data }: Props) {
  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500 text-sm">
        Waiting for price data...
      </div>
    )
  }

  const formatted = data.map(p => ({
    time: formatTime(p.time),
    Mid: +p.midPrice.toFixed(4),
    Bid: +p.bestBid.toFixed(4),
    Ask: +p.bestAsk.toFixed(4),
  }))

  const allPrices = data.flatMap(p => [p.bestBid, p.bestAsk])
  const min = Math.min(...allPrices)
  const max = Math.max(...allPrices)
  const pad = (max - min) * 0.15 || 0.02
  const domain: [number, number] = [+(min - pad).toFixed(4), +(max + pad).toFixed(4)]

  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={formatted}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
        <XAxis
          dataKey="time"
          tick={{ fill: '#6b7280', fontSize: 10 }}
          interval="preserveStartEnd"
        />
        <YAxis
          domain={domain}
          tick={{ fill: '#6b7280', fontSize: 10 }}
          width={55}
        />
        <Tooltip
          contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151', borderRadius: 6 }}
          labelStyle={{ color: '#9ca3af', marginBottom: 4 }}
          itemStyle={{ fontSize: 12 }}
        />
        <Legend iconType="plainline" />
        <Line type="monotone" dataKey="Mid" stroke="#60a5fa" dot={false} strokeWidth={2} />
        <Line type="monotone" dataKey="Bid" stroke="#34d399" dot={false} strokeWidth={1} strokeDasharray="5 3" />
        <Line type="monotone" dataKey="Ask" stroke="#f87171" dot={false} strokeWidth={1} strokeDasharray="5 3" />
      </LineChart>
    </ResponsiveContainer>
  )
}
