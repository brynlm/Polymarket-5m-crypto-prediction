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
  latestPredictions: Record<string, number> | null
}

function formatTime(ts: number) {
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

export function PriceChart({ data, latestPredictions }: Props) {
  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500 text-sm">
        Waiting for price data...
      </div>
    )
  }

  const formatted = data.map(p => ({
    time:      formatTime(p.time),
    Mid:       +p.midPrice.toFixed(4),
    Bid:       +p.bestBid.toFixed(4),
    Ask:       +p.bestAsk.toFixed(4),
    'Q50 (5s)': p.predQ50 != null ? +p.predQ50.toFixed(4) : null,
    'Q10':      p.predQ10 != null ? +p.predQ10.toFixed(4) : null,
    'Q90':      p.predQ90 != null ? +p.predQ90.toFixed(4) : null,
  }))

  const allPrices = data.flatMap(p => [p.bestBid, p.bestAsk])
  const min = Math.min(...allPrices)
  const max = Math.max(...allPrices)
  const pad = (max - min) * 0.15 || 0.02
  const domain: [number, number] = [+(min - pad).toFixed(4), +(max + pad).toFixed(4)]

  const hasPreds = latestPredictions != null &&
    (latestPredictions['q10'] != null || latestPredictions['q50'] != null)

  return (
    <div>
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
          <Line type="monotone" dataKey="Mid"       stroke="#60a5fa" dot={false} strokeWidth={2} />
          <Line type="monotone" dataKey="Bid"       stroke="#34d399" dot={false} strokeWidth={1} strokeDasharray="5 3" />
          <Line type="monotone" dataKey="Ask"       stroke="#f87171" dot={false} strokeWidth={1} strokeDasharray="5 3" />
          <Line type="monotone" dataKey="Q50 (5s)"  stroke="#f59e0b" dot={false} strokeWidth={1.5} strokeDasharray="4 2" connectNulls={false} />
          <Line type="monotone" dataKey="Q10"       stroke="#a78bfa" dot={false} strokeWidth={1} strokeDasharray="2 3" connectNulls={false} />
          <Line type="monotone" dataKey="Q90"       stroke="#fb923c" dot={false} strokeWidth={1} strokeDasharray="2 3" connectNulls={false} />
        </LineChart>
      </ResponsiveContainer>

      {hasPreds && (
        <div className="mt-3 grid grid-cols-3 gap-2">
          {(['q10', 'q50', 'q90'] as const).map((key, i) => {
            const labels = ['10th %ile', 'Median', '90th %ile']
            const colors = ['text-violet-400', 'text-amber-400', 'text-orange-400']
            const val = latestPredictions![key]
            return (
              <div key={key} className="bg-gray-800 rounded px-2 py-1.5 text-center">
                <div className="text-xs text-gray-500 mb-0.5">{labels[i]} +5s</div>
                <div className={`text-sm font-mono ${colors[i]}`}>
                  {val != null ? val.toFixed(4) : '—'}
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
