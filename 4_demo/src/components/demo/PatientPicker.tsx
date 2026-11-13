import { motion, useAnimationControls, type PanInfo } from "motion/react"
import { useEffect, useRef } from "react"

import { Badge } from "@/components/ui/badge"
import { dataUrl, type PatientSummary } from "@/lib/data"
import { DIAGNOSIS_STYLE, PATIENT_TAGS } from "@/lib/labels"
import { cn } from "@/lib/utils"

export interface DragHandlers {
  onDragStart: (id: string) => void
  onDragMove: (point: { x: number; y: number }) => void
  onDragEnd: (id: string, point: { x: number; y: number }) => void
}

// One patient card. On a mouse or trackpad it can be dragged; let go over the viewer to read the scan,
// anywhere else and it springs back. A click (or Enter) picks it too
function PatientCard({
  patient,
  index,
  isSelected,
  isFaded,
  canDrag,
  waveKey,
  onPick,
  drag,
}: {
  patient: PatientSummary
  index: number
  isSelected: boolean
  isFaded: boolean
  canDrag: boolean
  waveKey: number
  onPick: (id: string) => void
  drag: DragHandlers
}) {
  const controls = useAnimationControls()
  const wasDragged = useRef(false)
  const style = DIAGNOSIS_STYLE[patient.diagnosis]
  const tag = PATIENT_TAGS[patient.id]

  // The wave: each card lifts a little, one after the other, when someone arrives at the demo
  useEffect(() => {
    if (waveKey === 0) return
    controls.start({ y: [0, -10, 0], transition: { duration: 0.5, delay: 0.15 + index * 0.06, ease: "easeOut" } })
  }, [waveKey, index, controls])

  // Page coordinates from the drag, as the demo expects them
  const point = (info: PanInfo) => ({ x: info.point.x, y: info.point.y })

  return (
    <motion.button
      type="button"
      aria-pressed={isSelected}
      animate={controls}
      drag={canDrag}
      dragSnapToOrigin
      dragMomentum={false}
      dragElastic={0.15}
      whileDrag={{ scale: 1.04, rotate: -3, zIndex: 30, boxShadow: "0 18px 40px rgba(24,24,27,0.22)" }}
      onDragStart={() => {
        wasDragged.current = true
        drag.onDragStart(patient.id)
      }}
      onDrag={(_, info) => drag.onDragMove(point(info))}
      onDragEnd={(_, info) => {
        drag.onDragEnd(patient.id, point(info))
        window.setTimeout(() => (wasDragged.current = false), 0)
      }}
      onClick={() => {
        if (!wasDragged.current) onPick(patient.id)
      }}
      className={cn(
        "relative flex flex-col rounded-xl p-1.5 text-left transition-[opacity,border-color,background-color] duration-200 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ink md:rounded-2xl md:p-2.5",
        canDrag ? "cursor-grab active:cursor-grabbing" : "cursor-pointer",
        isSelected ? "border-2 border-ink bg-card shadow-[0_6px_18px_rgba(24,24,27,0.10)]" : "border border-line bg-page hover:border-faded",
        isFaded && "opacity-40",
      )}
    >
      <span className="pointer-events-none relative block aspect-square overflow-hidden rounded-lg bg-black md:rounded-[10px]">
        <img
          src={dataUrl(patient.thumbnail)}
          alt=""
          draggable={false}
          className="block size-full object-cover select-none"
        />
        {tag && (
          <Badge variant="onImage" size="sm" className="absolute bottom-1.5 left-1.5 px-2 text-[10px] md:bottom-2 md:left-2 md:text-xs">
            {tag}
          </Badge>
        )}
      </span>
      <span className="mt-1.5 flex items-baseline justify-between md:mt-2.5">
        <span className="text-[13px] font-semibold md:text-[15px]">{patient.id}</span>
        <span className="hidden text-xs text-muted md:inline">{patient.n_slices} slices</span>
      </span>
      <span className="mt-0.5 flex items-center gap-1.5 text-xs text-text md:mt-1 md:text-[13px]">
        <span className={cn("size-2 shrink-0 rounded-full md:size-[9px]", style.dot)} />
        {style.name}
      </span>
    </motion.button>
  )
}

// Part 1 of the demo: the 6 patients as cards
export function PatientPicker({
  patients,
  selected,
  dragging,
  canDrag,
  waveKey,
  onPick,
  drag,
}: {
  patients: PatientSummary[]
  selected: string | null
  dragging: string | null
  canDrag: boolean
  waveKey: number
  onPick: (id: string) => void
  drag: DragHandlers
}) {
  return (
    <div className="grid grid-cols-3 gap-2 md:grid-cols-[repeat(auto-fit,minmax(150px,1fr))] md:gap-3.5">
      {patients.map((patient, index) => (
        <PatientCard
          key={patient.id}
          patient={patient}
          index={index}
          isSelected={patient.id === selected}
          isFaded={dragging !== null && dragging !== patient.id}
          canDrag={canDrag}
          waveKey={waveKey}
          onPick={onPick}
          drag={drag}
        />
      ))}
    </div>
  )
}
