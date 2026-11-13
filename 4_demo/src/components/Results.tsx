import NumberFlow from "@number-flow/react"
import { useReducedMotion } from "motion/react"
import { SectionLabel } from "@/components/SectionLabel"
import { Card } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow, TableRowHead } from "@/components/ui/table"
import { pct, type Summary } from "@/lib/data"
import { useArrival, useSeenOnce } from "@/lib/navigation"
import { DIAGNOSES, DIAGNOSIS_STYLE } from "@/lib/labels"
import { cn } from "@/lib/utils"

// Section 03: the honest model on all 305 patients, by true diagnosis, and its screening setting.
// The first time the table comes into view, its bars fill from zero and the numbers count up
export function Results({ summary }: { summary: Summary | null }) {
  const { isLit } = useArrival("results")
  const [tableRef, seen] = useSeenOnce<HTMLDivElement>(0.5)
  const reduceMotion = useReducedMotion() ?? false
  const filled = seen || reduceMotion

  if (!summary) {
    return <section id="results" className="mx-auto h-96 max-w-[1100px] px-4 md:px-8" />
  }

  const model = summary.honest_model
  const [low, high] = model.balanced_accuracy_ci95
  const covidRow = model.confusion_matrix[model.diagnoses.indexOf("COVID-19")]
  const asPneumonia = covidRow[model.diagnoses.indexOf("CAP")]
  const asHealthy = covidRow[model.diagnoses.indexOf("Normal")]
  const screening = model.screening

  return (
    <section id="results" className="mx-auto max-w-[1100px] px-4 pt-6 pb-12 md:px-8 md:pb-24">
      <div className="mb-5 text-center md:mb-10">
        <SectionLabel lit={isLit} className="mb-2 md:mb-3">03 · Honest results</SectionLabel>
        <h2 data-section-title tabIndex={-1} className="m-0 text-[30px] focus:outline-none leading-[1.1] font-semibold tracking-[-0.03em] md:text-[44px]">The honest model, patient by patient</h2>
        <p className="mx-auto mt-2 mb-0 max-w-[560px] text-text md:mt-3 md:text-lg">
          {model.n_patients} patients, each judged by a model that never saw them.
        </p>
      </div>

      <div className="flex flex-col gap-5 md:flex-row md:items-stretch md:gap-7">
        <Card
          ref={tableRef}
          className={cn(
            "min-w-0 px-4 pt-1.5 pb-4 transition-[box-shadow,border-color] duration-700 md:flex-[999_1_560px] md:px-6 md:pt-2 md:pb-5",
            isLit && "border-ink shadow-[0_0_0_6px_rgba(24,24,27,0.10)]",
          )}
        >
          <Table>
            <TableHeader>
              <TableRow className="border-line">
                <TableHead>True diagnosis</TableHead>
                <TableHead className="hidden text-right md:table-cell">Patients</TableHead>
                <TableHead className="text-right">Correct</TableHead>
                <TableHead className="w-[42%] md:w-[38%]">Rate</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {DIAGNOSES.map((diagnosis) => {
                const row = model.per_diagnosis[diagnosis]
                const rate = row.correct / row.patients
                const style = DIAGNOSIS_STYLE[diagnosis]
                return (
                  <TableRow key={diagnosis}>
                    <TableRowHead>
                      <span className="inline-flex items-center gap-2 font-medium">
                        <span className={cn("size-[9px] shrink-0 rounded-full", style.dot)} />
                        {style.name}
                      </span>
                    </TableRowHead>
                    <TableCell className="hidden text-right md:table-cell">{row.patients}</TableCell>
                    <TableCell className="text-right whitespace-nowrap">
                      {row.correct}
                      <span className="text-muted md:hidden"> / {row.patients}</span>
                    </TableCell>
                    <TableCell>
                      <span className="flex items-center gap-3">
                        <span className="h-2 flex-1 overflow-hidden rounded-full bg-track">
                          <span
                            className={cn("block h-full rounded-full transition-[width] duration-1000 ease-out", style.bar)}
                            style={{ width: `${filled ? rate * 100 : 0}%` }}
                          />
                        </span>
                        <span className={cn("w-11 text-right font-semibold transition-colors duration-700", !filled && "text-faded")}>
                          <NumberFlow value={filled ? Math.round(rate * 100) : 0} suffix="%" />
                        </span>
                      </span>
                    </TableCell>
                  </TableRow>
                )
              })}
              <TableRow>
                <TableRowHead colSpan={2} className="pt-4 pb-1 font-semibold md:hidden">
                  Balanced accuracy
                  <span className="block text-xs font-normal text-muted">average of the 3 rates</span>
                </TableRowHead>
                <TableRowHead colSpan={2} className="hidden pt-4 pb-1 font-semibold md:table-cell">
                  Balanced accuracy <span className="font-normal text-muted">(average of the 3 rates)</span>
                </TableRowHead>
                <TableCell className="hidden md:table-cell" />
                <TableCell className="pt-4 pb-1 text-right text-lg font-semibold">
                  <NumberFlow value={filled ? Math.round(model.balanced_accuracy * 100) : 0} suffix="%" />
                </TableCell>
              </TableRow>
            </TableBody>
          </Table>
          <div className="mt-4 px-2 text-sm text-text md:text-[15px]">
            <p className="m-0">Most mistakes are COVID-19 patients:</p>
            <ul className="mt-1 mb-0 list-disc space-y-0.5 pl-5">
              <li>
                {asPneumonia} read as pneumonia: the model found the infection but took it for the other kind, which
                can look alike on CT.
              </li>
              <li>{asHealthy} read as healthy: often mild cases, with little infection visible on the scan.</li>
            </ul>
          </div>
          <p className="mt-3 mb-0 px-2 text-xs text-muted md:text-[13px]">
            On other patients the score would vary: most likely between {pct(low)} and {pct(high)} (95%
            confidence interval).
          </p>
        </Card>

        <aside className="rounded-[20px] bg-soft p-5 md:flex-[1_1_300px] md:p-6.5">
          <p className="m-0 text-xs font-semibold tracking-[0.06em] text-muted uppercase md:text-[13px]">Screening setting</p>
          <p className="mt-1.5 mb-4 text-lg font-semibold tracking-[-0.015em] md:mt-2 md:mb-5 md:text-[22px]">
            If the goal is to miss almost no COVID-19
          </p>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-1 md:gap-4.5">
            <div>
              <p className="m-0 text-[30px] leading-tight font-semibold tracking-[-0.03em] md:text-[40px]">
                {screening.covid_caught}{" "}
                <span className="text-base font-medium text-muted md:text-[22px]">of {screening.covid_patients}</span>
              </p>
              <p className="m-0 text-[13px] text-text md:text-sm">COVID-19 patients caught</p>
            </div>
            <div>
              <p className="m-0 text-[30px] leading-tight font-semibold tracking-[-0.03em] md:text-[40px]">
                {pct(screening.false_alarms / screening.other_patients)}
              </p>
              <p className="m-0 text-[13px] text-text md:text-sm">
                false alarms ({screening.false_alarms} of the {screening.other_patients} patients without COVID-19)
              </p>
            </div>
          </div>
          <p className="mt-4 mb-0 text-xs text-muted md:mt-4.5 md:text-[13px]">
            The cut-off was chosen on separate validation patients, aiming to catch 95% of COVID-19{"\u00a0"}cases.
          </p>
        </aside>
      </div>
    </section>
  )
}
