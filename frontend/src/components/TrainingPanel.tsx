import { useEffect, useState } from 'react'
import { Check, ChevronDown, Clipboard, Flag, RefreshCw } from 'lucide-react'
import type { CheckpointSummary } from '../types'

const trainingCommand = 'cd backend && (test -x .venv/bin/python || python3 -m venv .venv) && .venv/bin/python -m pip install -r requirements-rl.txt && .venv/bin/python -m app.rl.train --num-races 8 --cars-per-race 4 --preview-every 1 --total-steps 2000000'

export function TrainingPanel({
  checkpoints,
  latestCheckpoint,
  activeCheckpoint,
  onRefresh,
  onLoad,
}: {
  checkpoints: CheckpointSummary[]
  latestCheckpoint: string | null
  activeCheckpoint: string | null
  onRefresh: () => Promise<void>
  onLoad: (name: string) => Promise<void>
}) {
  const [open, setOpen] = useState(false)
  const [selected, setSelected] = useState('')
  const [copied, setCopied] = useState(false)
  const [loading, setLoading] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const [message, setMessage] = useState<string | null>(null)

  useEffect(() => {
    if (selected && checkpoints.some((checkpoint) => checkpoint.name === selected)) return
    setSelected(activeCheckpoint ?? latestCheckpoint ?? checkpoints[0]?.name ?? '')
  }, [activeCheckpoint, checkpoints, latestCheckpoint, selected])

  const copyCommand = async () => {
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(trainingCommand)
      } else {
        const field = document.createElement('textarea')
        field.value = trainingCommand
        field.style.position = 'fixed'
        field.style.opacity = '0'
        document.body.append(field)
        field.select()
        const copiedText = document.execCommand('copy')
        field.remove()
        if (!copiedText) throw new Error('Clipboard access is unavailable')
      }
      setCopied(true)
      setMessage('Training command copied')
      window.setTimeout(() => setCopied(false), 1800)
    } catch {
      setMessage('Could not access the clipboard')
    }
  }

  const refresh = async () => {
    setRefreshing(true)
    setMessage(null)
    try {
      await onRefresh()
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Could not refresh checkpoints')
    } finally {
      setRefreshing(false)
    }
  }

  const load = async () => {
    if (!selected) return
    setLoading(true)
    setMessage(null)
    try {
      await onLoad(selected)
      setMessage('Checkpoint loaded; the race is ready')
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Could not load checkpoint')
    } finally {
      setLoading(false)
    }
  }

  return <div className="training-panel">
    <button
      className="training-trigger hud-surface"
      onClick={() => setOpen((value) => !value)}
      aria-expanded={open}
      aria-controls="training-panel-content"
    >
      <Flag size={14} />
      <span>TRAIN / RACE</span>
      <ChevronDown size={13} className={open ? 'rotated' : ''} />
    </button>
    {open && <section id="training-panel-content" className="training-card hud-surface" aria-label="Training and race controls">
      <div className="training-heading">
        <div>
          <span className="training-kicker">SHARED POLICY · SELF-PLAY</span>
          <h2>Train a race policy</h2>
        </div>
        <button className="training-icon-button" onClick={refresh} disabled={refreshing} aria-label="Refresh checkpoints" title="Refresh checkpoints">
          <RefreshCw size={14} className={refreshing ? 'spinning' : ''} />
        </button>
      </div>

      <p className="training-copy">Run this from the project root in a terminal to train eight four-car races on this computer. The hosted simulator races with the checkpoint deployed on its backend; local training results stay here until you deploy the updated checkpoint.</p>
      <code className="training-command">{trainingCommand}</code>
      <button className="training-copy-button" onClick={copyCommand}>
        {copied ? <Check size={14} /> : <Clipboard size={14} />}
        {copied ? 'Copied' : 'Copy training command'}
      </button>

      <div className="training-divider" />
      <label className="training-label" htmlFor="checkpoint-select">CHECKPOINT</label>
      <div className="checkpoint-row">
        <select id="checkpoint-select" value={selected} onChange={(event) => setSelected(event.target.value)} disabled={!checkpoints.length || loading}>
          {!checkpoints.length && <option value="">No checkpoints found</option>}
          {checkpoints.map((checkpoint) => <option key={checkpoint.name} value={checkpoint.name}>
            {checkpoint.name}{checkpoint.name === latestCheckpoint ? ' · latest' : ''}
          </option>)}
        </select>
        <button className="checkpoint-load-button" onClick={load} disabled={!selected || loading}>
          {loading ? 'Loading…' : 'Load race'}
        </button>
      </div>
      {activeCheckpoint && <div className="checkpoint-active">Loaded: {activeCheckpoint}</div>}
      {message && <div className="training-message" role="status">{message}</div>}
    </section>}
  </div>
}
