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
import type { PricePoint, PredictionPoint } from '../types'

interface Props {
  data: PricePoint[]
  predictionHistory: PredictionPoint[]
  latestPredictions: Record<string, number> | null
}

/** Binary-search predictionHistory for the entry nearest to `time`, within maxDeltaMs. */
function findNearestPreds(
  preds: PredictionPoint[],
  time: number,
  maxDeltaMs = 5000,
): Record<string, number> | null {
  if (preds.length === 0) return null
  let lo = 0, hi = preds.length - 1
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (preds[mid].time < time) lo = mid + 1
    else hi = mid
  }
  const candidates: PredictionPoint[] = [preds[lo]]
  if (lo > 0) candidates.push(preds[lo - 1])
  const best = candidates.reduce((a, b) =>
    Math.abs(a.time - time) < Math.abs(b.time - time) ? a : b
  )
  return Math.abs(best.time - time) <= maxDeltaMs ? best.predictions : null
}

const PRED_HORIZON_MS = 5000

const SERIES = [
  { key: 'Mid',      color: '#60a5fa', width: 2,   dash: undefined },
  { key: 'Bid',      color: '#34d399', width: 1,   dash: '5 3' },
  { key: 'Ask',      color: '#f87171', width: 1,   dash: '5 3' },
  { key: 'Q50 (5s)', color: '#f59e0b', width: 1.5, dash: '4 2' },
  { key: 'Q10',      color: '#a78bfa', width: 1,   dash: '2 3' },
  { key: 'Q90',      color: '#fb923c', width: 1,   dash: '2 3' },
] as const

type SeriesKey = typeof SERIES[number]['key']

type ChartPoint = {
  time: string
  Mid:        number | null
  Bid:        number | null
  Ask:        number | null
  'Q50 (5s)': number | null
  Q10:        number | null
  Q90:        number | null
}

const DEFAULT_VISIBLE: Record<SeriesKey, boolean> = {
  'Mid': true, 'Bid': true, 'Ask': true, 'Q50 (5s)': true, 'Q10': true, 'Q90': true,
}

const WINDOW_PRESETS = [15, 30, 60, 300] as const
const DEFAULT_WINDOW_SECS = 30

function formatTime(ts: number) {
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

function formatWindowLabel(secs: number) {
  return secs >= 60 ? `${secs / 60}m` : `${secs}s`
}

export function PriceChart({ data, predictionHistory, latestPredictions }: Props) {
  const [visible, setVisible] = useState<Record<SeriesKey, boolean>>(DEFAULT_VISIBLE)
  const [windowSecs, setWindowSecs] = useState(DEFAULT_WINDOW_SECS)

  const toggleSeries = (key: SeriesKey) =>
    setVisible(v => ({ ...v, [key]: !v[key] }))

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500 text-sm">
        Waiting for price data...
      </div>
    )
  }

  const lastPoint  = data[data.length - 1]
  const cutoff     = lastPoint.time - windowSecs * 1000
  const windowData = data.filter(p => p.time >= cutoff)
  const hasPreds   = latestPredictions != null && latestPredictions['q50'] != null

  // Map each price point to its nearest prediction from history
  const formatted: ChartPoint[] = windowData.map(p => {
    const preds = findNearestPreds(predictionHistory, p.time)
    return {
      time:       formatTime(p.time),
      Mid:        +p.midPrice.toFixed(4),
      Bid:        +p.bestBid.toFixed(4),
      Ask:        +p.bestAsk.toFixed(4),
      'Q50 (5s)': preds?.['q50'] != null ? +preds['q50'].toFixed(4) : null,
      Q10:        preds?.['q10'] != null ? +preds['q10'].toFixed(4) : null,
      Q90:        preds?.['q90'] != null ? +preds['q90'].toFixed(4) : null,
    }
  })

  // Project the latest predictions forward to T+5s
  if (hasPreds) {
    const q10 = +latestPredictions!['q10'].toFixed(4)
    const q50 = +latestPredictions!['q50'].toFixed(4)
    const q90 = +latestPredictions!['q90'].toFixed(4)
    formatted.push({
      time:       formatTime(lastPoint.time + PRED_HORIZON_MS),
      Mid:        null,
      Bid:        null,
      Ask:        null,
      'Q50 (5s)': q50,
      Q10:        q10,
      Q90:        q90,
    })
  }

  // Domain: include Q values so the future points are always in view
  const allPrices = windowData.flatMap(p => [p.bestBid, p.bestAsk])
  if (hasPreds) {
    allPrices.push(latestPredictions!['q10'], latestPredictions!['q90'])
  }
  const min = Math.min(...allPrices)
  const max = Math.max(...allPrices)
  const pad = (max - min) * 0.15 || 0.02
  const domain: [number, number] = [+(min - pad).toFixed(4), +(max + pad).toFixed(4)]

  return (
    <div>
      {/* Toolbar: series toggles + window selector */}
      <div className="flex flex-wrap items-center justify-between gap-1.5 mb-3">
        <div className="flex flex-wrap gap-1.5">
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
        <div className="flex gap-1">
          {WINDOW_PRESETS.map(secs => (
            <button
              key={secs}
              onClick={() => setWindowSecs(secs)}
              className={`px-2 py-0.5 rounded text-xs transition-colors ${
                windowSecs === secs
                  ? 'bg-gray-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              {formatWindowLabel(secs)}
            </button>
          ))}
        </div>
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
