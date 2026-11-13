import { Button } from "@/components/ui/button"
import { CodeIcon } from "@/components/icons"
import { REPO_URL } from "@/lib/labels"
import { goToSection, useActiveSection, type SectionId } from "@/lib/navigation"
import { cn } from "@/lib/utils"

const SECTIONS: { id: SectionId; number: string; name: string }[] = [
  { id: "demo", number: "01", name: "Try it" },
  { id: "story", number: "02", name: "The 99% problem" },
  { id: "results", number: "03", name: "Honest results" },
]

// Top bar, always visible: project name, the three sections in page order (the current one marked), the code
export function SiteHeader() {
  const active = useActiveSection()

  return (
    <header className="sticky top-0 z-20 border-b border-line bg-page/90 backdrop-blur-md">
      <div className="mx-auto flex max-w-[1200px] items-center justify-between gap-4 px-4 py-1.5 md:px-8 md:py-2.5">
        <a href="#top" className="font-mono text-sm font-medium text-ink no-underline md:text-[15px]">
          covid_ct_diagnosis
        </a>

        <nav aria-label="Sections" className="hidden items-center gap-1 text-sm md:flex">
          {SECTIONS.map((section) => (
            <a
              key={section.id}
              href={`#${section.id}`}
              onClick={(event) => goToSection(event, section.id)}
              aria-current={active === section.id ? "true" : undefined}
              className={cn(
                "inline-flex min-h-10 items-center gap-2 rounded-full border px-3.5 no-underline transition-colors",
                active === section.id
                  ? "border-line bg-card font-medium text-ink"
                  : "border-transparent text-text hover:border-line hover:bg-card",
              )}
            >
              <span className="font-mono text-xs text-muted">{section.number}</span>
              {section.name}
            </a>
          ))}
        </nav>

        <Button asChild size="sm" className="min-h-11 px-3.5 md:min-h-10 md:px-4">
          <a href={REPO_URL} target="_blank" rel="noopener noreferrer">
            <CodeIcon size={15} />
            <span className="md:hidden">Code</span>
            <span className="hidden md:inline">Code on GitHub</span>
            <span className="sr-only">(opens in a new tab)</span>
          </a>
        </Button>
      </div>
    </header>
  )
}
