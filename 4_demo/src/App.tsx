import { useEffect, useState } from "react"

import { Demo } from "@/components/demo/Demo"
import { Hero } from "@/components/Hero"
import { Results } from "@/components/Results"
import { SiteFooter } from "@/components/SiteFooter"
import { SiteHeader } from "@/components/SiteHeader"
import { Story } from "@/components/Story"
import { loadJson, type PatientSummary, type Summary } from "@/lib/data"

// The whole page: header, hero, demo (01), story (02), results (03), footer
export default function App() {
  const [patients, setPatients] = useState<PatientSummary[] | null>(null)
  const [summary, setSummary] = useState<Summary | null>(null)

  // The list of demo patients and the numbers of the story, written by export_demo.py
  useEffect(() => {
    loadJson<{ patients: PatientSummary[] }>("index.json").then((index) => setPatients(index.patients))
    loadJson<Summary>("summary.json").then(setSummary)
  }, [])

  // The page grows once the data is in, so a link to a section is followed again at that point
  useEffect(() => {
    if (!patients || !summary || !window.location.hash) return
    document.getElementById(window.location.hash.slice(1))?.scrollIntoView()
  }, [patients, summary])

  return (
    <div id="top" className="min-h-screen bg-page text-ink">
      <SiteHeader />
      <main>
        <Hero patients={summary?.honest_model.n_patients ?? null} />
        <Demo patients={patients} />
        <Story summary={summary} />
        <Results summary={summary} />
      </main>
      <SiteFooter />
    </div>
  )
}
