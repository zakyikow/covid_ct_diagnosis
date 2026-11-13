import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ArrowDown } from "@/components/icons"
import { goToSection } from "@/lib/navigation"

// Title, the one-paragraph story, and the two ways in: try the demo or jump to the results
export function Hero({ patients }: { patients: number | null }) {
  return (
    <section className="mx-auto max-w-[1200px] px-4 pt-10 pb-8 text-center md:px-8 md:pt-20 md:pb-14">
      <Badge variant="outline" className="text-xs md:text-[13px]">
        <span className="hidden md:inline">A reproducibility study · </span>2 datasets ·{" "}
        {patients ?? 305} patients tested
      </Badge>
      <h1 className="mx-auto mt-5 max-w-[860px] text-[40px] leading-[1.04] font-semibold tracking-[-0.035em] md:mt-6 md:text-[66px] md:leading-[1.02]">
        Can a 99% <span className="whitespace-nowrap">COVID-19</span> detector hold up?
      </h1>
      <p className="mx-auto mt-4 max-w-[640px] text-base text-text md:mt-5 md:text-xl md:leading-relaxed">
        Chest CT classifiers trained on public data often report 99% accuracy. But the popular dataset doesn't
        say which slices come from which patient, so the same patients quietly end up in both the training and test
        sets. Here is the model rebuilt honestly, and what it really gets{"\u00a0"}right.
      </p>
      <div className="mt-6 flex flex-col justify-center gap-3 md:mt-8 md:flex-row">
        <Button asChild>
          <a href="#demo" onClick={(event) => goToSection(event, "demo")}>
            Try a patient
            <ArrowDown />
          </a>
        </Button>
        <Button asChild variant="outline" className="hidden md:inline-flex">
          <a href="#results" onClick={(event) => goToSection(event, "results")}>
            See the results
            <ArrowDown />
          </a>
        </Button>
      </div>
    </section>
  )
}
