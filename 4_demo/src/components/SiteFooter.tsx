import { Button } from "@/components/ui/button"
import { CodeIcon } from "@/components/icons"
import { REPO_URL } from "@/lib/labels"

// Data sources and the two notes that keep the page honest
export function SiteFooter() {
  return (
    <footer className="border-t border-line">
      <div className="mx-auto flex max-w-[1200px] flex-col gap-6 px-4 pt-6 pb-10 text-xs text-muted md:flex-row md:gap-12 md:px-8 md:pt-8 md:pb-12 md:text-[13px]">
        <Button asChild variant="outline" size="sm" className="min-h-11 md:hidden">
          <a href={REPO_URL} target="_blank" rel="noopener noreferrer">
            <CodeIcon size={15} />
            Code on GitHub
            <span className="sr-only">(opens in a new tab)</span>
          </a>
        </Button>
        <div className="md:flex-[1_1_420px]">
          <p className="m-0 mb-1.5 font-semibold text-ink">Data</p>
          <p className="m-0">Afshar P. et al. COVID-CT-MD. Scientific Data, 2021.</p>
          <p className="m-0">Ning W. et al. iCTCF. Nature Biomedical Engineering, 2020.</p>
        </div>
        <div className="md:flex-[1_1_300px]">
          <p className="m-0 mb-1.5 font-semibold text-ink">About this page</p>
          <p className="m-0">The model doesn't run here: its results were computed in advance.</p>
          <p className="m-0">A personal research project. Not a medical device.</p>
        </div>
      </div>
    </footer>
  )
}
