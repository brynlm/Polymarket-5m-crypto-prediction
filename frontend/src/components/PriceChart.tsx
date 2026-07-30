import { useState, useEffect, useRef } from 'react'
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
  latestPredictions: Record<string, Record<string, number>> | null
}

/** Binary-search predictionHistory for the entry nearest to `time`, within maxDeltaMs. */
function findNearestPreds(
  preds: PredictionPoint[],
  time: number,
  maxDeltaMs = 5000,
): Record<string, Record<string, number>> | null {
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
  ts:         number
  Mid:        number | null
  Bid:        number | null
  Ask:        number | null
  'Q50 (5s)': number | null
  Q10:        number | null
  Q90:        number | null
}

const TRANSITION_MS = 200

function lerp(a: number | null, b: number | null, t: number): number | null {
  if (a == null || b == null) return b
  return a + (b - a) * t
}

function smoothstep(t: number) {
  return t * t * (3 - 2 * t)
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
  const [now, setNow] = useState(() => Date.now())
  const rafRef = useRef<number>(0)
  const windowFilledRef = useRef(false)
  const transitionRef = useRef<{ from: ChartPoint; to: ChartPoint; startedAt: number } | null>(null)

  useEffect(() => {
    const tick = () => {
      setNow(Date.now())
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafRef.current)
  }, [])

  const toggleSeries = (key: SeriesKey) =>
    setVisible(v => ({ ...v, [key]: !v[key] }))

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500 text-sm">
        Waiting for price data...
      </div>
    )
  }

  const windowMs = windowSecs * 1000
  const cutoff   = now - windowMs

  // Lock to fixed-width domain once a full window of data has arrived.
  // Only reset on a genuine new-interval clear (data[0] jumps >10s ahead of cutoff),
  // not on the tiny per-point fluctuations of the rolling buffer hovering near cutoff.
  if (windowFilledRef.current && data[0].time > cutoff + 10_000) windowFilledRef.current = false
  if (!windowFilledRef.current && data[0].time <= cutoff) windowFilledRef.current = true
  const xLeft    = windowFilledRef.current ? cutoff : data[0].time

  const firstIdx   = data.findIndex(p => p.time >= xLeft)
  const windowData = data.slice(Math.max(0, firstIdx - 1))

  // Ticks anchored to absolute rounded time boundaries so they slide leftward with the
  // data rather than recalculating positions every frame and appearing to stutter.
  const TICK_COUNT    = 5
  const tickInterval  = Math.round((now - xLeft) / TICK_COUNT / 1000) * 1000 || 1000
  const lastTickBase  = Math.floor(now / tickInterval) * tickInterval
  const xTicks        = Array.from({ length: TICK_COUNT + 2 }, (_, i) => lastTickBase - i * tickInterval)
    .filter(t => t >= xLeft && t <= now)
  const hasPreds   = latestPredictions?.['UP']?.['q50'] != null

  // Map each price point to its nearest prediction from history (UP market for the chart)
  const formatted: ChartPoint[] = windowData.map(p => {
    const preds = findNearestPreds(predictionHistory, p.time)?.['UP']
    return {
      ts:         p.time,
      Mid:        +p.midPrice.toFixed(4),
      Bid:        +p.bestBid.toFixed(4),
      Ask:        +p.bestAsk.toFixed(4),
      'Q50 (5s)': preds?.['q50'] != null ? +preds['q50'].toFixed(4) : null,
      Q10:        preds?.['q10'] != null ? +preds['q10'].toFixed(4) : null,
      Q90:        preds?.['q90'] != null ? +preds['q90'].toFixed(4) : null,
    }
  })

  // Phantom "live" point pinned to now with interpolated Y values so the dot
  // smoothly transitions to each new data point rather than jumping.
  const realLast = formatted[formatted.length - 1]
  if (realLast && realLast.ts < now) {
    const prev = transitionRef.current
    if (!prev || prev.to.ts !== realLast.ts) {
      transitionRef.current = { from: prev?.to ?? realLast, to: realLast, startedAt: now }
    }
    const { from, to, startedAt } = transitionRef.current!
    const t = smoothstep(Math.min(1, (now - startedAt) / TRANSITION_MS))
    formatted.push({
      ts:         now,
      Mid:        lerp(from.Mid, to.Mid, t),
      Bid:        lerp(from.Bid, to.Bid, t),
      Ask:        lerp(from.Ask, to.Ask, t),
      'Q50 (5s)': lerp(from['Q50 (5s)'], to['Q50 (5s)'], t),
      Q10:        lerp(from.Q10, to.Q10, t),
      Q90:        lerp(from.Q90, to.Q90, t),
    })
  }

  // Domain: include Q values in range
  const allPrices = windowData.flatMap(p => [p.bestBid, p.bestAsk])
  formatted.forEach(p => {
    if (p.Q10 != null) allPrices.push(p.Q10)
    if (p.Q90 != null) allPrices.push(p.Q90)
  })
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
          <XAxis
            dataKey="ts"
            type="number"
            domain={[xLeft, now]}
            ticks={xTicks}
            tickFormatter={formatTime}
            tick={{ fill: '#6b7280', fontSize: 10 }}
          />
          <YAxis domain={domain} tick={{ fill: '#6b7280', fontSize: 10 }} width={55} />
          <Tooltip
            contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151', borderRadius: 6 }}
            labelStyle={{ color: '#9ca3af', marginBottom: 4 }}
            itemStyle={{ fontSize: 12 }}
            labelFormatter={(v: number) => formatTime(v)}
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
                dot={(props: any) => {
                  if (props.index !== formatted.length - 1 || props.cy == null) return <g key={props.key} />
                  return <circle key={props.key} cx={props.cx} cy={props.cy} r={3.5} fill={color} stroke="#111827" strokeWidth={1.5} />
                }}
                activeDot={false}
                connectNulls={false}
                isAnimationActive={false}
              />
            ) : null
          )}
        </LineChart>
      </ResponsiveContainer>

      {hasPreds && (
        <div className="mt-3 space-y-2">
          {(['UP', 'DOWN'] as const).map(mkt => {
            const mktPreds = latestPredictions![mkt]
            if (!mktPreds) return null
            const mktColors: Record<string, string> = {
              UP: 'text-emerald-400', DOWN: 'text-rose-400',
            }
            return (
              <div key={mkt}>
                <div className={`text-xs font-semibold mb-1 ${mktColors[mkt]}`}>{mkt}</div>
                <div className="grid grid-cols-3 gap-2">
                  {(['q10', 'q50', 'q90'] as const).map((key, i) => {
                    const labels = ['10th %ile', 'Median', '90th %ile']
                    const colors = ['text-violet-400', 'text-amber-400', 'text-orange-400']
                    const val = mktPreds[key]
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
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
