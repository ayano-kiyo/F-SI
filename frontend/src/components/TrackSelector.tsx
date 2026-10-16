import { useEffect, useRef, useState } from 'react'
import { Check, ChevronDown, MapPin } from 'lucide-react'
import type { TrackSummary } from '../types'

export function TrackSelector({ tracks, activeName, switching, onSelect }: {
  tracks: TrackSummary[]
  activeName: string | null
  switching: boolean
  onSelect: (name: string) => void
}) {
  const [open, setOpen] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)
  const active = tracks.find((t) => t.name === activeName) ?? null

  useEffect(() => {
    if (!open) return
    const onPointer = (event: PointerEvent) => {
      if (rootRef.current && !rootRef.current.contains(event.target as Node)) setOpen(false)
    }
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false)
    }
    document.addEventListener('pointerdown', onPointer)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('pointerdown', onPointer)
      document.removeEventListener('keydown', onKey)
    }
  }, [open ])

  return <div className="track-picker hud-surface" ref={rootRef}>
    <button
      className="track-select"
      onClick={() => setOpen((v) => !v)}
      aria-expanded={open}
      aria-busy={switching}
      aria-label={`Circuit ${active?.label ?? activeName ?? 'loading'}`}
      title="Switch circuit"
    >
      <MapPin size={13} />
      <span className="track-select-label">{switching ? 'LOADING…' : (active?.label ?? activeName ?? 'TRACK')}</span>
      <span className="track-select-length">{active ? `${(active.length_m / 1000).toFixed(2)} KM` : ''}</span>
      <ChevronDown size={13} />
    </button>
    {open && <div className="track-menu" role="listbox" aria-label="Select circuit">
      {tracks.length === 0 && <div className="track-menu-empty">Loading circuits…</div>}
      {tracks.map((t) => <button
        key={t.name}
        role="option"
        aria-selected={t.name === activeName}
        className={t.name === activeName ? 'selected' : ''}
        onClick={() => { setOpen(false); onSelect(t.name) }}
      >
        <span className="track-menu-name">{t.label}</span>
        <span className="track-menu-meta">{(t.length_m / 1000).toFixed(2)} km</span>
        {t.name === activeName && <Check size={12} />}
      </button>)}
    </div>}
  </div>
}
