import { useState } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import type { PricePoint } from '../types'

interface Props {
  data: PricePoint[]
  latestPredictions: Record<string, number> | null
}

const SERIES = [
  { key: 'Mid',      color: '#60a5fa', width: 2,   dash: undefined },
  { key: 'Bid',      color: '#34d399', width: 1,   dash: '5 3' },
  { key: 'Ask',      color: '#f87171', width: 1,   dash: '5 3' },
  { key: 'Q50 (5s)', color: '#f59e0b', width: 1.5, dash: '4 2' },
  { key: 'Q10',      color: '#a78bfa', width: 1,   dash: '2 3' },
  { key: 'Q90',      color: '#fb923c', width: 1,   dash: '2 3' },
] as const

type SeriesKey = typeof SERIES[number]['key']

const DEFAULT_VISIBLE: Record<SeriesKey, boolean> = {
  'Mid': true, 'Bid': true, 'Ask': true, 'Q50 (5s)': true, 'Q10': true, 'Q90': true,
}

function formatTime(ts: number) {
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

export function PriceChart({ data, latestPredictions }: Props) {
  const [visible, setVisible] = useState<Record<SeriesKey, boolean>>(DEFAULT_VISIBLE)

  const toggleSeries = (key: SeriesKey) =>
    setVisible(v => ({ ...v, [key]: !v[key] }))

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500 text-sm">
        Waiting for price data...
      </div>
    )
  }

  const formatted = data.map(p => ({
    time:       formatTime(p.time),
    Mid:        +p.midPrice.toFixed(4),
    Bid:        +p.bestBid.toFixed(4),
    Ask:        +p.bestAsk.toFixed(4),
    'Q50 (5s)': p.predQ50 != null ? +p.predQ50.toFixed(4) : null,
    Q10:        p.predQ10 != null ? +p.predQ10.toFixed(4) : null,
    Q90:        p.predQ90 != null ? +p.predQ90.toFixed(4) : null,
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
      {/* Custom toggle legend */}
      <div className="flex flex-wrap gap-1.5 mb-3">
        {SERIES.map(({ key, color }) => {
          const on = visible[key]
          return (
            <button
              key={key}
              onClick={() => toggleSeries(key)}
              className="flex items-center gap-1 px-2 py-0.5 rounded text-xs transition-opacity"
              style={{ opacity: on ? 1 : 0.35 }}
            >
              <span className="w-5 h-0.5 inline-block rounded" style={{ backgroundColor: color }} />
              <span style={{ color: on ? color : '#6b7280' }}>{key}</span>
            </button>
          )
        })}
      </div>

      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={formatted}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis dataKey="time" tick={{ fill: '#6b7280', fontSize: 10 }} interval="preserveStartEnd" />
          <YAxis domain={domain} tick={{ fill: '#6b7280', fontSize: 10 }} width={55} />
          <Tooltip
            contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151', borderRadius: 6 }}
            labelStyle={{ color: '#9ca3af', marginBottom: 4 }}
            itemStyle={{ fontSize: 12 }}
          />
          {SERIES.map(({ key, color, width, dash }) =>
            visible[key] ? (
              <Line
                key={key}
                type="monotone"
                dataKey={key}
                stroke={color}
                strokeWidth={width}
                strokeDasharray={dash}
                dot={false}
                connectNulls={false}
              />
            ) : null
          )}
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
