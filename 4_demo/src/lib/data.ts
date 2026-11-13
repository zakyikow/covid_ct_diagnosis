// The results the page shows. They are written by `bash scripts/run.sh step4` (export_demo.py) into
// public/data/: nothing is computed in the browser, the page only reads and replays them.

export type Diagnosis = "COVID-19" | "CAP" | "Normal"
export type LesionClass = "covid" | "cap"

export interface PatientSummary {
  id: string
  diagnosis: Diagnosis
  n_slices: number
  thumbnail: string
  data: string
}

export interface Bar {
  first: number
  last: number
  scores: Record<LesionClass, number>
  reading: number[]
}

export interface TopSlice {
  slice: number
  score: number
  radiologist_label: string | null
  image: string
  heatmap: string
}

export interface Patient {
  id: string
  diagnosis: Diagnosis
  n_slices: number
  slice_classes: string[]
  diagnoses: Diagnosis[]
  prediction: { diagnosis: Diagnosis; percents: number[] }
  slices: { slice: number; scores: number[] }[]
  bars: Bar[]
  frames: { image: string; columns: number; count: number }
  top_slices: TopSlice[]
}

export interface Summary {
  step1: { accuracy: number; n_slices: number }
  step2: { covid_slices_missed: number }
  step3: { random_split: number; patient_split: number }
  honest_model: {
    n_patients: number
    balanced_accuracy: number
    balanced_accuracy_ci95: [number, number]
    diagnoses: Diagnosis[]
    confusion_matrix: number[][]
    per_diagnosis: Record<Diagnosis, { patients: number; correct: number }>
    screening: { covid_caught: number; covid_patients: number; false_alarms: number; other_patients: number }
  }
}

// Full address of a file in public/data, relative to wherever the page is served from
export function dataUrl(path: string): string {
  return `${import.meta.env.BASE_URL}data/${path}`
}

// Read one JSON file from public/data
export async function loadJson<T>(path: string): Promise<T> {
  const response = await fetch(dataUrl(path))
  if (!response.ok) throw new Error(`Could not load ${path} (${response.status})`)
  return (await response.json()) as T
}

// Whole percentage as text: 0.941 -> "94%"
export function pct(value: number): string {
  return `${Math.round(value * 100)}%`
}

// Below this score a heatmap would only show where the model looked for an infection it didn't find, so
// none is shown
export const HEATMAP_MIN_SCORE = 0.5

// Whole percentages that add up to 100. The largest one is rounded normally, so it matches the reported
// result; the points left are shared between the others, largest remainders first
export function wholePercents(values: number[]): number[] {
  const raw = values.map((v) => v * 100)
  const top = raw.indexOf(Math.max(...raw))
  const rounded = raw.map((v, i) => (i === top ? Math.round(v) : Math.floor(v)))
  const leftover = 100 - rounded.reduce((a, b) => a + b, 0)
  const order = raw
    .map((v, i) => ({ i, remainder: v - Math.floor(v) }))
    .filter(({ i }) => i !== top)
    .sort((a, b) => b.remainder - a.remainder)
  for (let k = 0; k < leftover; k++) rounded[order[k % order.length].i] += 1
  return rounded
}

// A slice counts as "marked by the radiologists" when they labelled it with a lesion
export function isMarkedInfected(label: string | null): boolean {
  return label === "covid" || label === "cap"
}
