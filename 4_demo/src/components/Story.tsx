import type { ReactNode } from "react"

import { ArrowRight } from "@/components/icons"
import { SectionLabel } from "@/components/SectionLabel"
import { Badge } from "@/components/ui/badge"
import { pct, type Summary } from "@/lib/data"
import { useArrival } from "@/lib/navigation"
import { cn } from "@/lib/utils"

// One step of the story: the number on the left, what it means on the right
function StoryStep({ number, step, title, children }: { number: ReactNode; step: string; title: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-1.5 border-t border-line py-5.5 md:flex-row md:gap-10 md:py-7.5">
      <div className="md:w-[300px] md:shrink-0">{number}</div>
      <div className="md:flex-1">
        <p className="m-0 text-xs font-semibold tracking-[0.06em] text-muted uppercase md:text-[13px]">{step}</p>
        <h3 className="mt-1.5 mb-1 text-[19px] font-semibold tracking-[-0.015em] md:text-2xl">{title}</h3>
        <div className="text-[15px] text-text md:text-base">{children}</div>
      </div>
    </div>
  )
}

// A big number of the story
function BigNumber({ value, className }: { value: string; className?: string }) {
  return <p className={cn("m-0 text-5xl leading-none font-semibold tracking-[-0.04em] md:text-[64px]", className)}>{value}</p>
}

// Under a big number: what exactly it measures
function NumberNote({ children }: { children: string }) {
  return <p className="mt-1.5 mb-0 text-xs text-muted md:text-[13px]">{children}</p>
}

// Section 02: where the 99% comes from, in four checks
export function Story({ summary }: { summary: Summary | null }) {
  const s = summary
  const { isLit } = useArrival("story")
  const gap = s ? Math.round((s.step3.random_split - s.step3.patient_split) * 100) : 19
  return (
    <section id="story" className="border-t border-line">
      <div className="mx-auto max-w-[960px] px-4 pt-13 pb-8 md:px-8 md:pt-24 md:pb-18">
        <div className="mb-6 text-center md:mb-12">
          <SectionLabel lit={isLit} className="mb-2 md:mb-3">02 · The 99% problem</SectionLabel>
          <h2 data-section-title tabIndex={-1} className="m-0 text-[30px] focus:outline-none leading-[1.1] font-semibold tracking-[-0.03em] md:text-[44px]">Where the 99% comes from</h2>
          <p className="mx-auto mt-2 mb-0 max-w-[560px] text-text md:mt-3 md:text-lg">
            Four checks on the same kind of model, from the claim to the honest number.
          </p>
          <p className="mx-auto mt-2 mb-0 max-w-[620px] text-[13px] text-muted md:mt-2.5 md:text-sm">
            Steps 3a and 3b use balanced accuracy: the share found correctly for each diagnosis, averaged, so a rare
            diagnosis counts as much as a common one.
          </p>
        </div>

        <StoryStep number={<BigNumber value={s ? pct(s.step1.accuracy) : "99%"} />} step="Step 1 · The claim" title="Reproducing the popular model">
          Trained and tested on {s ? s.step1.n_slices.toLocaleString("en-US") : "19,685"} CT slices from Wuhan,
          shuffled at random, the model scores {s ? pct(s.step1.accuracy) : "99%"} accuracy. But shuffling puts slices
          of the same patients in both the training and test sets.
        </StoryStep>

        <StoryStep
          number={<BigNumber value={s ? pct(s.step2.covid_slices_missed) : "48%"} className="text-covid-text" />}
          step="Step 2 · Another hospital"
          title="It misses half the infection elsewhere"
        >
          On scans from an imaging centre in Tehran, the same model misses {s ? pct(s.step2.covid_slices_missed) : "48%"} of
          the slices that radiologists marked as showing COVID-19.
        </StoryStep>

        <StoryStep
          number={
            <div className="flex flex-col gap-3.5">
              <div className="flex items-start gap-3 md:gap-3.5">
                <div>
                  <BigNumber value={s ? pct(s.step3.random_split) : "96%"} className="text-faded md:text-[56px]" />
                  <NumberNote>slices shuffled</NumberNote>
                </div>
                <ArrowRight size={26} className="mt-2.5 md:mt-4" />
                <div>
                  <BigNumber value={s ? pct(s.step3.patient_split) : "77%"} className="md:text-[56px]" />
                  <NumberNote>split by patient</NumberNote>
                </div>
              </div>
              <Badge variant="gap" size="sm" className="self-start">−{gap} points</Badge>
            </div>
          }
          step="Step 3a · The leak"
          title="Same data, split by patient"
        >
          A new model is trained on the Tehran scans, which do have patient IDs, and judges one slice at a time.
          Shuffled, each test slice has near-identical neighbours in the training set, so the model can recognise
          them instead of learning the disease: {s ? pct(s.step3.random_split) : "96%"}. Split by patient, each
          patient only in training or only in test: {s ? pct(s.step3.patient_split) : "77%"}.
        </StoryStep>

        <div className="border-b border-line">
          <StoryStep
            number={
              <div>
                <BigNumber value={s ? pct(s.honest_model.balanced_accuracy) : "84%"} />
                <NumberNote>per patient</NumberNote>
              </div>
            }
            step="Step 3b · The honest model"
            title="One diagnosis per patient"
          >
            The honest model then combines all the slices of a scan into one diagnosis per patient: COVID-19,
            pneumonia or healthy. A whole scan holds more evidence than one slice, so the score rises to{" "}
            {s ? pct(s.honest_model.balanced_accuracy) : "84%"}, still on patients the model never saw.
          </StoryStep>
        </div>
      </div>
    </section>
  )
}
