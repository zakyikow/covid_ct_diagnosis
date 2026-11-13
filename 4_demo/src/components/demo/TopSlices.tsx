import { motion } from "motion/react"

import { dataUrl, HEATMAP_MIN_SCORE, isMarkedInfected, type Patient } from "@/lib/data"
import { cn } from "@/lib/utils"

// What the radiologists' own slice labels say about these slices, in one sentence
function radiologistNote(patient: Patient): string {
  const labels = patient.top_slices.map((s) => s.radiologist_label)
  const marked = labels.filter(isMarkedInfected).length
  if (patient.diagnosis === "Normal") return "The radiologists found no infection anywhere in this scan."
  if (labels.every((label) => label === null)) return "This patient has no slice-by-slice labels from the radiologists."
  if (marked === labels.length) return `The radiologists marked all ${labels.length} as infected too.`
  return `The radiologists marked ${marked} of these ${labels.length} as infected.`
}

// The most suspicious slices with their heatmaps, sliding in one by one once the scan is read. Clicking one shows
// it in the viewer. Before that, empty places
export function TopSlices({
  patient,
  show,
  animate,
  current,
  onSelect,
}: {
  patient: Patient | null
  show: boolean
  animate: boolean
  current: number | null
  onSelect: (slice: number) => void
}) {
  const count = patient?.top_slices.length ?? 5
  return (
    <div>
      <p className="m-0 text-[13px] font-medium md:text-sm">The {count} most suspicious slices</p>
      <div className="mt-2 grid grid-cols-5 gap-1.5 md:mt-2.5 md:gap-2">
        {show && patient
          ? patient.top_slices.map((slice, i) => (
              <motion.button
                key={`${patient.id}-${slice.slice}`}
                type="button"
                aria-pressed={slice.slice === current}
                aria-label={`Show slice ${slice.slice} in the viewer`}
                onClick={() => onSelect(slice.slice)}
                initial={animate ? { opacity: 0, y: 10 } : false}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: animate ? 0.5 + i * 0.09 : 0, duration: 0.3, ease: "easeOut" }}
                className="group flex cursor-pointer flex-col items-center focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ink"
              >
                <span
                  className={cn(
                    "relative block aspect-square w-full overflow-hidden rounded-[7px] bg-black md:rounded-[10px]",
                    slice.slice === current ? "ring-2 ring-ink ring-offset-2" : "group-hover:ring-1 group-hover:ring-faded",
                  )}
                >
                  <img src={dataUrl(slice.image)} alt="" className="absolute inset-0 block size-full object-cover" />
                  {slice.score >= HEATMAP_MIN_SCORE && (
                    <img src={dataUrl(slice.heatmap)} alt="" className="absolute inset-0 block size-full object-cover" />
                  )}
                </span>
                <span className="mt-1 hidden font-mono text-[11px] text-muted md:block">#{slice.slice}</span>
              </motion.button>
            ))
          : Array.from({ length: count }, (_, i) => <span key={i} className="aspect-square rounded-[7px] bg-soft md:rounded-[10px]" />)}
      </div>
      <p className="mt-2 mb-0 min-h-[1.5em] text-xs text-muted md:text-[13px]">{show && patient ? radiologistNote(patient) : ""}</p>
    </div>
  )
}
