import { useReducedMotion } from "motion/react"
import { useCallback, useEffect, useRef, useState } from "react"

import { SectionLabel } from "@/components/SectionLabel"
import { Badge } from "@/components/ui/badge"
import { Card } from "@/components/ui/card"
import { dataUrl, loadJson, type Patient, type PatientSummary } from "@/lib/data"
import { FIRST_PATIENT } from "@/lib/labels"
import { scrollIntoSight, scrollToCentre, useArrival, useFinePointer, useSeenOnce } from "@/lib/navigation"
import { cn } from "@/lib/utils"
import { DiagnosisPanel } from "./DiagnosisPanel"
import { ModelPipeline } from "./ModelPipeline"
import { PatientPicker, type DragHandlers } from "./PatientPicker"
import { ScanViewer } from "./ScanViewer"
import { ScoreStrip } from "./ScoreStrip"
import { TopSlices } from "./TopSlices"

// waiting: no patient yet. reading: the scan is being replayed. done: the result is shown
export type Phase = "waiting" | "reading" | "done"

// How long the replay of a scan lasts
const READ_MS = 3000

// Load images before they are needed, so the replay doesn't stutter (a failed image doesn't block it)
function preload(paths: string[]): Promise<void> {
  return Promise.all(
    paths.map(
      (path) =>
        new Promise<void>((resolve) => {
          const img = new Image()
          img.onload = img.onerror = () => resolve()
          img.src = dataUrl(path)
        }),
    ),
  ).then(() => undefined)
}

// Round black number in front of each part of the demo
function StepHeading({ number, children, isSectionTitle = false }: { number: number; children: string; isSectionTitle?: boolean }) {
  return (
    <div className="flex items-center gap-2.5 md:gap-3">
      <span className="flex size-[26px] shrink-0 items-center justify-center rounded-full bg-ink text-[13px] font-semibold text-white md:size-[30px] md:text-sm">
        {number}
      </span>
      <h2
        className="m-0 text-[19px] font-semibold tracking-[-0.01em] focus:outline-none md:text-[22px]"
        {...(isSectionTitle ? { "data-section-title": true, tabIndex: -1 } : {})}
      >
        {children}
      </h2>
    </div>
  )
}

// The demo card: pick a patient (click, tap or drag), then watch the model read the scan and give its diagnosis
export function Demo({ patients }: { patients: PatientSummary[] | null }) {
  const reduceMotion = useReducedMotion() ?? false
  const canDrag = useFinePointer()
  const { arrivals, isLit } = useArrival("demo")
  const viewerRef = useRef<HTMLDivElement>(null)
  const readingRef = useRef<HTMLDivElement>(null)
  const [watchViewer, seen] = useSeenOnce<HTMLDivElement>(0.5)

  // The viewer is both measured (drops, scrolling) and watched (the first patient plays once it is half on screen)
  const setViewer = useCallback(
    (element: HTMLDivElement | null) => {
      viewerRef.current = element
      watchViewer(element)
    },
    [watchViewer],
  )

  const [selected, setSelected] = useState<string | null>(null)
  const [run, setRun] = useState(0)
  const [phase, setPhase] = useState<Phase>("waiting")
  const [step, setStep] = useState(-1)
  const [wasAnimated, setWasAnimated] = useState(false)
  const [cache, setCache] = useState<Record<string, Patient>>({})
  const [viewerSlice, setViewerSlice] = useState<number | null>(null)
  const [dragging, setDragging] = useState<string | null>(null)
  const [isOver, setIsOver] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Start reading a patient's scan. Picking the same patient again replays it
  const read = useCallback((id: string) => {
    setSelected(id)
    setViewerSlice(null)
    setRun((r) => r + 1)
  }, [])

  // The first time the viewer comes into view and nothing is picked yet, the first patient plays by itself
  useEffect(() => {
    if (!seen || selected !== null || !patients) return
    const timer = window.setTimeout(() => read(FIRST_PATIENT), 500)
    return () => window.clearTimeout(timer)
  }, [seen, selected, patients, read])

  // One run: load the patient, load its images, then replay the scan bar by bar (or show the result at once)
  useEffect(() => {
    if (!selected || !patients || run === 0) return
    const summary = patients.find((p) => p.id === selected)
    if (!summary) return
    let cancelled = false
    let frame = 0
    setPhase("reading")
    setStep(-1)
    setError(null)

    const cached = cache[selected]
    const load = cached ? Promise.resolve(cached) : loadJson<Patient>(summary.data)
    load
      .then(async (patient) => {
        if (cancelled) return
        setCache((c) => (c[patient.id] ? c : { ...c, [patient.id]: patient }))
        await preload([patient.frames.image, ...patient.top_slices.flatMap((s) => [s.image, s.heatmap])])
        if (cancelled) return

        const n = patient.bars.length
        if (reduceMotion) {
          setWasAnimated(false)
          setStep(n - 1)
          setPhase("done")
          return
        }
        setWasAnimated(true)
        const start = performance.now()
        const tick = (now: number) => {
          if (cancelled) return
          const elapsed = now - start
          if (elapsed >= READ_MS) {
            setStep(n - 1)
            setPhase("done")
            return
          }
          setStep(Math.min(n - 1, Math.floor((elapsed / READ_MS) * n)))
          frame = requestAnimationFrame(tick)
        }
        frame = requestAnimationFrame(tick)
      })
      .catch((e: Error) => {
        if (cancelled) return
        setError(e.message)
        setPhase("waiting")
      })

    return () => {
      cancelled = true
      cancelAnimationFrame(frame)
    }
    // The cache is read, not watched: filling it must not restart the run
  }, [selected, run, patients, reduceMotion])

  // Whether a point of the page (from a drag) is over the viewer
  function isOverViewer(point: { x: number; y: number }): boolean {
    const box = viewerRef.current?.getBoundingClientRect()
    if (!box) return false
    const x = point.x - window.scrollX
    const y = point.y - window.scrollY
    return x >= box.left && x <= box.right && y >= box.top && y <= box.bottom
  }

  const drag: DragHandlers = {
    onDragStart: (id) => setDragging(id),
    onDragMove: (point) => setIsOver(isOverViewer(point)),
    onDragEnd: (id, point) => {
      setDragging(null)
      setIsOver(false)
      if (isOverViewer(point)) {
        read(id)
        showReading()
      }
    },
  }

  // After a pick, bring the reading into sight: the whole viewer and strip on a laptop, the viewer on a phone
  function showReading() {
    window.setTimeout(() => (canDrag ? scrollIntoSight(readingRef.current) : scrollToCentre(viewerRef.current)), 80)
  }

  // A click, a tap or Enter on a card: read it
  function pick(id: string) {
    read(id)
    showReading()
  }

  const patient = selected ? (cache[selected] ?? null) : null
  const shownSlice = phase === "done" && patient ? (patient.top_slices.find((s) => s.slice === viewerSlice) ?? patient.top_slices[0]) : null

  return (
    <section id="demo" className="mx-auto max-w-[1200px] px-3 pb-14 md:px-8 md:pb-26">
      <SectionLabel lit={isLit} className="mb-3 text-center md:mb-4">
        01 · Try it
      </SectionLabel>
      <Card
        className={cn(
          "rounded-3xl p-3.5 shadow-[0_1px_2px_rgba(24,24,27,0.04),0_16px_40px_rgba(24,24,27,0.06)] transition-[box-shadow,border-color] duration-700 md:rounded-[28px] md:p-7",
          isLit && "border-ink shadow-[0_0_0_6px_rgba(24,24,27,0.10),0_16px_40px_rgba(24,24,27,0.08)]",
        )}
      >
        <div className="mb-3 flex flex-col gap-1 md:mb-4.5 md:flex-row md:items-center md:justify-between md:gap-3">
          <StepHeading number={1} isSectionTitle>
            Pick a patient
          </StepHeading>
          <p className="m-0 pl-9 text-[13px] text-muted md:pl-0 md:text-sm">
            <span className={cn("rounded-full px-2.5 py-1 transition-colors duration-500", isLit && "bg-soft text-ink")}>
              {canDrag ? "Drag a scan into the viewer, or click it" : "Tap a scan"}
            </span>
          </p>
        </div>
        {patients ? (
          <PatientPicker
            patients={patients}
            selected={selected}
            dragging={dragging}
            canDrag={canDrag}
            waveKey={reduceMotion ? 0 : arrivals}
            onPick={pick}
            drag={drag}
          />
        ) : (
          <div className="h-40 animate-pulse rounded-2xl bg-soft" />
        )}

        <div className="-mx-3.5 my-5 h-px bg-line md:-mx-7 md:my-7.5" />

        <div className="mb-3 flex flex-col gap-1.5 md:mb-4.5 md:flex-row md:items-center md:justify-between md:gap-3">
          <StepHeading number={2}>Watch the model read the scan</StepHeading>
          <div className="ml-9 flex items-center gap-2 md:ml-0">
            {patient && (
              <Badge>
                {patient.id} · Tehran · {patient.n_slices} slices
              </Badge>
            )}
          </div>
        </div>

        {error && <p className="text-covid-text">Could not load this patient: {error}</p>}

        <div
          ref={readingRef}
          className="flex flex-col gap-4 md:grid md:grid-cols-[minmax(0,460px)_minmax(0,1fr)] md:grid-rows-[auto_auto_auto_1fr] md:gap-x-7 md:gap-y-5"
        >
          <div className="order-1 min-w-0 md:col-start-1 md:row-span-4 md:row-start-1">
            <ScanViewer
              ref={setViewer}
              patient={patient}
              phase={phase}
              step={step}
              slice={shownSlice}
              reveal={wasAnimated && viewerSlice === null}
              isDragging={dragging !== null}
              isOver={isOver}
            />
          </div>
          <div className="order-2 min-w-0 md:col-start-2 md:row-start-2">
            <ModelPipeline patient={patient} phase={patient ? phase : "waiting"} step={step} slice={shownSlice} />
          </div>
          <div className="order-3 min-w-0 md:col-start-2 md:row-start-3">
            <ScoreStrip bars={patient?.bars ?? null} readUpTo={phase === "waiting" ? -1 : step} currentSlice={shownSlice?.slice ?? null} />
          </div>
          <div className="order-4 min-w-0 md:col-start-2 md:row-start-1">
            <DiagnosisPanel patient={patient} phase={patient ? phase : "waiting"} step={step} />
          </div>
          <div className="order-5 min-w-0 md:col-start-2 md:row-start-4">
            <TopSlices patient={patient} show={phase === "done"} animate={wasAnimated} current={shownSlice?.slice ?? null} onSelect={setViewerSlice} />
          </div>
        </div>

        <p className="mt-5 mb-0 text-center text-xs text-muted md:mt-6.5 md:text-[13px]">
          Judged by a model that never saw this patient in training. Results computed in advance.
        </p>
      </Card>
    </section>
  )
}
