import { AnimatePresence, motion } from "motion/react"
import { forwardRef } from "react"

import { dataUrl, HEATMAP_MIN_SCORE, pct, type Patient, type TopSlice } from "@/lib/data"
import { cn } from "@/lib/utils"
import type { Phase } from "./Demo"

// Small dark label on top of the image
function ImageLabel({ children, className }: { children: string; className?: string }) {
  return (
    <span className={cn("absolute rounded-md bg-black/55 px-2 py-0.5 text-[11px] text-zinc-100 md:text-xs", className)}>
      {children}
    </span>
  )
}

// One frame of the sheet of small images replayed while the scan is read (index = bar of the score strip)
export function Frame({ patient, index, className }: { patient: Patient; index: number; className?: string }) {
  const { image, columns, count } = patient.frames
  const rows = Math.ceil(count / columns)
  const column = index % columns
  const row = Math.floor(index / columns)
  return (
    <div
      className={cn("bg-black bg-no-repeat", className)}
      style={{
        backgroundImage: `url(${dataUrl(image)})`,
        backgroundSize: `${columns * 100}% ${rows * 100}%`,
        backgroundPosition: `${columns > 1 ? (column / (columns - 1)) * 100 : 0}% ${rows > 1 ? (row / (rows - 1)) * 100 : 0}%`,
      }}
    />
  )
}

// The finished result on one slice, with the heatmap fading in when the result has just been revealed
function SliceResult({ patientId, slice, nSlices, reveal }: { patientId: string; slice: TopSlice; nSlices: number; reveal: boolean }) {
  const hasHeatmap = slice.score >= HEATMAP_MIN_SCORE

  return (
    <>
      <img
        src={dataUrl(slice.image)}
        alt={`Axial CT slice ${slice.slice} of patient ${patientId}${hasHeatmap ? ", with the model's heatmap" : ""}`}
        className="absolute inset-0 block size-full object-contain"
      />
      {hasHeatmap && (
        <motion.img
          src={dataUrl(slice.heatmap)}
          alt=""
          className="absolute inset-0 block size-full object-contain"
          initial={{ opacity: reveal ? 0 : 1 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.15, ease: "easeOut" }}
        />
      )}
      <ImageLabel className="top-2.5 left-2.5 font-mono md:top-3 md:left-3.5">{`Slice ${slice.slice} / ${nSlices}`}</ImageLabel>
      <ImageLabel className="bottom-2.5 left-2.5 md:bottom-3 md:left-3.5">
        {hasHeatmap ? "Orange areas: where the model looked" : `Infection score ${pct(slice.score)}: too low for a heatmap`}
      </ImageLabel>
    </>
  )
}

// The viewer, which is also where a patient is dropped. Waiting: an empty drop zone. Reading: the scan plays from
// the top of the chest down. Done: the most suspicious slice (or the one picked below) with its heatmap
export const ScanViewer = forwardRef<
  HTMLDivElement,
  { patient: Patient | null; phase: Phase; step: number; slice: TopSlice | null; reveal: boolean; isDragging: boolean; isOver: boolean }
>(function ScanViewer({ patient, phase, step, slice, reveal, isDragging, isOver }, ref) {
  const bar = patient && step >= 0 ? patient.bars[step] : null

  return (
    <div
      ref={ref}
      className={cn(
        "relative aspect-square overflow-hidden rounded-[14px] transition-[box-shadow,background-color] duration-200 md:rounded-[18px]",
        phase === "waiting" ? "bg-page" : "bg-black",
        isDragging && "ring-2 ring-faded ring-offset-2",
        isOver && "ring-ink ring-offset-4",
      )}
    >
      {phase === "waiting" && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2.5 rounded-[inherit] border-2 border-dashed border-nothing p-6 text-center">
          <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" className="text-muted">
            <path d="M12 4v11M7 10l5 5 5-5M5 19h14" />
          </svg>
          <p className="m-0 font-semibold">
            <span className="hidden md:inline">Drag a patient here</span>
            <span className="md:hidden">Tap a patient above</span>
          </p>
          <p className="m-0 hidden text-[13px] text-muted md:block">or click one above</p>
        </div>
      )}

      {phase === "reading" && patient && bar && (
        <>
          <Frame patient={patient} index={step} className="absolute inset-0" />
          <ImageLabel className="top-2.5 left-2.5 font-mono md:top-3 md:left-3.5">{`Reading slice ${bar.last} / ${patient.n_slices}`}</ImageLabel>
        </>
      )}

      {phase === "done" && patient && slice && (
        <SliceResult key={`${patient.id}-${slice.slice}`} patientId={patient.id} slice={slice} nSlices={patient.n_slices} reveal={reveal} />
      )}

      <AnimatePresence>
        {isDragging && (
          <motion.div
            className={cn(
              "absolute inset-0 flex items-center justify-center rounded-[inherit] p-5",
              isOver ? "border-2 border-ink bg-soft/90" : phase === "waiting" ? "" : "bg-black/30",
            )}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
          >
            {isOver && <span className="rounded-full bg-card px-4 py-2 font-semibold shadow-sm">Drop to read the scan</span>}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
})
