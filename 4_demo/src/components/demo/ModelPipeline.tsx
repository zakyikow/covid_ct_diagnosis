import { ArrowRight } from "@/components/icons"
import { dataUrl, type Patient, type TopSlice } from "@/lib/data"
import { cn } from "@/lib/utils"
import type { Phase } from "./Demo"
import { Frame } from "./ScanViewer"

// A stack of feature maps: `count` squares of `size` px, each shifted a little, the deepest `lit` ones glowing as
// much as the infection score (`heat`)
function Stack({ size, count, heat, lit }: { size: number; count: number; heat: number; lit: number }) {
  const shift = 4
  return (
    <span className="relative block shrink-0" style={{ width: size + shift * (count - 1), height: size + shift * (count - 1) }}>
      {Array.from({ length: count }, (_, i) => {
        const isLit = i >= count - lit && heat > 0
        return (
          <span
            key={i}
            className={cn("absolute rounded-[3px] border transition-colors duration-150", isLit ? "border-covid-text/40" : "border-nothing")}
            style={{
              left: i * shift,
              top: i * shift,
              width: size,
              height: size,
              background: isLit ? `rgba(232,89,12,${(0.15 + heat * 0.75).toFixed(2)})` : i % 2 ? "#ffffff" : "#efede8",
            }}
          />
        )
      })}
    </span>
  )
}

// The 3 scores of a slice, from what is on screen: the bar being read, or the slice in the viewer once done
function currentScores(patient: Patient | null, phase: Phase, step: number, slice: TopSlice | null) {
  if (!patient || phase === "waiting") return { covid: 0, cap: 0, nothing: 0 }
  if (phase === "done" && slice) {
    const scores = patient.slices[slice.slice - 1].scores
    const at = (name: string) => scores[patient.slice_classes.indexOf(name)]
    return { covid: at("covid"), cap: at("cap"), nothing: at("no_lesion") }
  }
  const { covid, cap } = patient.bars[Math.max(0, step)].scores
  return { covid, cap, nothing: Math.max(0, 1 - Math.max(covid, cap)) }
}

// How the model reads each slice: its layers find simple, then more and more complex patterns, then 3 scores
// (COVID-19 / pneumonia / nothing visible). It runs along while the scan is read, then stays on the slice in the
// viewer, so it can be read at leisure
export function ModelPipeline({ patient, phase, step, slice }: { patient: Patient | null; phase: Phase; step: number; slice: TopSlice | null }) {
  const scores = currentScores(patient, phase, step, slice)
  const heat = Math.max(scores.covid, scores.cap)
  const bars = [
    { name: "COVID-19", value: scores.covid, colour: "bg-covid" },
    { name: "Pneumonia", value: scores.cap, colour: "bg-pneumonia" },
    { name: "Nothing visible", value: scores.nothing, colour: "bg-nothing" },
  ]

  return (
    <figure className="m-0 rounded-2xl border border-line bg-page px-3 pt-2.5 pb-2.5 md:px-4" aria-hidden="true">
      <div className="mb-2 flex items-baseline justify-between gap-2 text-xs text-muted md:text-[13px]">
        <span className="font-medium text-ink">How the model reads each slice</span>
        {phase === "done" && slice && <span className="font-mono">slice {slice.slice}</span>}
      </div>
      <div className="flex items-center justify-between gap-1.5">
        {phase === "reading" && patient ? (
          <Frame patient={patient} index={Math.max(0, step)} className="size-9 shrink-0 rounded-[4px] md:size-11" />
        ) : phase === "done" && slice ? (
          <img src={dataUrl(slice.image)} alt="" className="size-9 shrink-0 rounded-[4px] bg-black object-cover md:size-11" />
        ) : (
          <span className="size-9 shrink-0 rounded-[4px] bg-soft md:size-11" />
        )}
        <ArrowRight size={14} className="shrink-0 text-muted" />
        <Stack size={30} count={3} heat={heat} lit={0} />
        <Stack size={19} count={4} heat={heat} lit={1} />
        <Stack size={11} count={6} heat={heat} lit={2} />
        <ArrowRight size={14} className="shrink-0 text-muted" />
        <span className="flex h-9 items-end gap-[3px] md:h-11">
          {bars.map((bar) => (
            <span
              key={bar.name}
              className={cn("w-2 rounded-[2px] transition-[height] duration-150", phase === "waiting" ? "bg-track" : bar.colour)}
              style={{ height: `${Math.max(6, bar.value * 100)}%` }}
            />
          ))}
        </span>
      </div>
      <figcaption className="mt-1.5 grid grid-cols-[auto_1fr_auto] gap-3 text-[11px] text-muted">
        <span>slice</span>
        <span className="text-center">simple → complex patterns</span>
        <span>3 scores</span>
      </figcaption>
    </figure>
  )
}
