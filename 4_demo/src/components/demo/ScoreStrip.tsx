import type { Bar } from "@/lib/data"
import { LESION_BAR } from "@/lib/labels"
import { cn } from "@/lib/utils"

// Colour of a bar: the lesion class the model favours there, grey when neither reaches 50%
function barColour(bar: Bar): string {
  const { covid, cap } = bar.scores
  if (Math.max(covid, cap) < 0.5) return LESION_BAR.nothing
  return covid >= cap ? LESION_BAR.covid : LESION_BAR.cap
}

const LEGEND = [
  { name: "COVID-19", colour: LESION_BAR.covid },
  { name: "Pneumonia", colour: LESION_BAR.cap },
  { name: "Nothing visible", colour: LESION_BAR.nothing },
]

// The score along the scan, top of the chest first. Each bar covers a few slices and is as tall as the model's
// highest infection score among them. Bars after `readUpTo` are still waiting to be read. A short dark line
// marks the bar holding the slice in the viewer (an outline would overlap the bars next to it)
export function ScoreStrip({ bars, readUpTo, currentSlice }: { bars: Bar[] | null; readUpTo: number; currentSlice: number | null }) {
  return (
    <figure className="m-0">
      <div className="flex flex-wrap items-center justify-between gap-2 text-[13px] text-muted">
        <figcaption className="font-medium text-ink">Score along the scan</figcaption>
        <div className="flex flex-wrap gap-3 text-xs md:gap-3.5 md:text-[13px]">
          {LEGEND.map((item) => (
            <span key={item.name} className="flex items-center gap-1.5">
              <span className={cn("size-2.5 rounded-[3px]", item.colour)} />
              {item.name}
            </span>
          ))}
        </div>
      </div>

      <div
        className="mt-2.5 flex gap-[2px] md:gap-[3px]"
        role="img"
        aria-label="Infection score of each part of the scan, from the top of the chest to the bottom"
      >
        {(bars ?? Array.from({ length: 40 }, () => null)).map((bar, i) => {
          const isRead = bar !== null && i <= readUpTo
          const height = isRead ? Math.max(bar.scores.covid, bar.scores.cap) : 0
          const isCurrent = isRead && currentSlice !== null && bar.first <= currentSlice && currentSlice <= bar.last
          return (
            <span key={i} className="flex flex-1 flex-col gap-[3px]">
              <span className="flex h-[46px] items-end md:h-[60px]">
                <span
                  className={cn(
                    "w-full rounded-[2px] transition-[height,background-color] duration-150 ease-out md:rounded-[3px]",
                    isRead ? barColour(bar) : "bg-track",
                  )}
                  style={{ height: `${Math.max(6, height * 100)}%` }}
                />
              </span>
              <span className={cn("h-[3px] rounded-full", isCurrent ? "bg-ink" : "bg-transparent")} />
            </span>
          )
        })}
      </div>

      <div className="mt-1.5 flex justify-between text-[11px] text-muted md:text-xs">
        <span>Top of the chest</span>
        <span>Bottom</span>
      </div>
      <p className="mt-2 mb-0 text-xs text-muted md:text-[13px]">
        Each bar covers a few slices. The taller it is, the surer the model is that infection is visible there.
      </p>
    </figure>
  )
}
