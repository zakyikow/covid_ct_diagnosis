import type { Diagnosis, LesionClass } from "./data"

// Names and colours of the three diagnoses, in the order the page lists them
export const DIAGNOSES: Diagnosis[] = ["COVID-19", "CAP", "Normal"]

export const DIAGNOSIS_STYLE: Record<Diagnosis, { name: string; dot: string; bar: string; text: string }> = {
  "COVID-19": { name: "COVID-19", dot: "bg-covid", bar: "bg-covid", text: "text-covid-text" },
  CAP: { name: "Pneumonia", dot: "bg-pneumonia", bar: "bg-pneumonia", text: "text-pneumonia-text" },
  Normal: { name: "Healthy", dot: "bg-healthy", bar: "bg-healthy", text: "text-healthy-text" },
}

// Colour of a bar of the score strip: the lesion class with the higher score, grey when neither reaches 50%
export const LESION_BAR: Record<LesionClass | "nothing", string> = {
  covid: "bg-covid",
  cap: "bg-pneumonia",
  nothing: "bg-nothing",
}

// Small labels shown on some patient cards
export const PATIENT_TAGS: Record<string, string> = {
  P021: "Hard case",
}

// The patient that plays by itself the first time the scan viewer comes into view
export const FIRST_PATIENT = "P002"

// Address of the code
export const REPO_URL = "https://github.com/zakyikow/covid_ct_diagnosis"
