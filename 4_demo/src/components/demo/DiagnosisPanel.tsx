import NumberFlow from "@number-flow/react"

import { CheckCircle, CrossCircle } from "@/components/icons"
import { isMarkedInfected, wholePercents, type Diagnosis, type Patient } from "@/lib/data"
import { DIAGNOSES, DIAGNOSIS_STYLE } from "@/lib/labels"
import { cn } from "@/lib/utils"
import type { Phase } from "./Demo"

// The three percentages, in page order, from the patient's values (stored in the order of `patient.diagnoses`)
function inPageOrder(patient: Patient, values: number[]): number[] {
  return DIAGNOSES.map((d) => values[patient.diagnoses.indexOf(d)])
}

// The verdict under the diagnosis: right or wrong, and for a wrong one whether the infection was still found
function Verdict({ patient }: { patient: Patient }) {
  const isCorrect = patient.prediction.diagnosis === patient.diagnosis
  const truth = patient.diagnosis === "Normal" ? "is healthy" : `has ${patient.diagnosis === "CAP" ? "pneumonia" : "COVID-19"}`
  const foundTheInfection = patient.top_slices.some((s) => isMarkedInfected(s.radiologist_label))
  return (
    <div className="mt-3.5 flex items-start gap-2 border-t border-line pt-3 text-[13px] text-text md:mt-4 md:pt-3.5 md:text-sm">
      {isCorrect ? <CheckCircle size={18} className="mt-px text-ink" /> : <CrossCircle size={18} className="mt-px text-covid-text" />}
      <p className="m-0">
        {isCorrect ? `Correct: this patient ${truth}.` : `Wrong: this patient ${truth}.`}
        {!isCorrect && foundTheInfection && " The model found the infected slices but got the type of infection wrong."}
      </p>
    </div>
  )
}

// The model's diagnosis for the whole patient. While the scan is read it shows, in grey, what the slices read so
// far point to; at the end it turns to the diagnosis colour and says whether it is right
export function DiagnosisPanel({ patient, phase, step }: { patient: Patient | null; phase: Phase; step: number }) {
  const isDone = phase === "done"
  const values = patient
    ? inPageOrder(patient, isDone ? patient.prediction.percents : patient.bars[Math.max(0, step)].reading)
    : [0, 0, 0]
  const shown = patient && phase !== "waiting" ? wholePercents(values) : [0, 0, 0]
  const leading: Diagnosis = DIAGNOSES[shown.indexOf(Math.max(...shown))]
  const style = DIAGNOSIS_STYLE[leading]

  return (
    <div className="rounded-2xl border border-line bg-page p-4 md:rounded-[18px] md:p-[22px]">
      <p className="m-0 text-xs font-medium text-muted md:text-[13px]">
        {phase === "reading" ? "The model is leaning towards" : "The model's diagnosis"}
      </p>
      <p className="mt-0.5 mb-0 flex items-baseline justify-between gap-3">
        <span
          className={cn(
            "text-[32px] font-semibold tracking-[-0.03em] transition-colors duration-300 md:text-[40px]",
            isDone ? style.text : "text-faded",
          )}
        >
          {phase === "waiting" ? "—" : style.name}
        </span>
        <span className={cn("font-mono text-[26px] font-medium transition-colors duration-300 md:text-[34px]", !isDone && "text-faded")}>
          <NumberFlow value={Math.max(...shown)} suffix="%" />
        </span>
      </p>

      <dl className="mt-3 mb-0 flex flex-col gap-2 md:mt-4 md:gap-2.5">
        {DIAGNOSES.map((diagnosis, i) => (
          <div key={diagnosis} className="grid grid-cols-[80px_1fr_40px] items-center gap-2.5 text-[13px] md:grid-cols-[90px_1fr_44px] md:gap-3 md:text-sm">
            <dt className="text-text">{DIAGNOSIS_STYLE[diagnosis].name}</dt>
            <dd className="m-0 h-2 overflow-hidden rounded-full bg-track">
              <span
                className={cn("block h-full rounded-full transition-[width] duration-150 ease-out", DIAGNOSIS_STYLE[diagnosis].bar, !isDone && "opacity-45")}
                style={{ width: `${shown[i]}%` }}
              />
            </dd>
            <dd className={cn("m-0 text-right font-mono", !isDone && "text-faded")}>
              <NumberFlow value={shown[i]} suffix="%" />
            </dd>
          </div>
        ))}
      </dl>

      {isDone && patient ? (
        <Verdict patient={patient} />
      ) : (
        <p className="mt-3.5 mb-0 border-t border-line pt-3 text-[13px] text-muted md:mt-4 md:pt-3.5 md:text-sm">
          {phase === "reading" ? "Reading the scan…" : "Pick a patient to see the model's diagnosis."}
        </p>
      )}

      <p className="sr-only" aria-live="polite">
        {isDone && patient ? `The model says ${DIAGNOSIS_STYLE[patient.prediction.diagnosis].name}, ${Math.max(...shown)}%.` : ""}
      </p>
    </div>
  )
}
