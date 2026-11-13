import { useEffect, useState, type MouseEvent } from "react"

// Sections a link can jump to, in page order
const SECTION_IDS = ["demo", "story", "results"] as const
export type SectionId = (typeof SECTION_IDS)[number]

const ARRIVAL_EVENT = "section-arrival"
const ARRIVAL_SHOWN_MS = 1500

// True when the visitor asked the system for less motion
function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches
}

// Scroll to a section, then tell it that someone arrived (it lights up) and move the keyboard focus to its title
export function goToSection(event: MouseEvent<HTMLAnchorElement>, id: SectionId) {
  const section = document.getElementById(id)
  if (!section) return
  event.preventDefault()
  history.replaceState(null, "", `#${id}`)
  section.scrollIntoView({ behavior: prefersReducedMotion() ? "auto" : "smooth", block: "start" })

  // Light the section up once the scroll has finished (or after 0.7 s on browsers without a scrollend event)
  let announced = false
  const announce = () => {
    if (announced) return
    announced = true
    window.removeEventListener("scrollend", announce)
    section.querySelector<HTMLElement>("[data-section-title]")?.focus({ preventScroll: true })
    window.dispatchEvent(new CustomEvent(ARRIVAL_EVENT, { detail: id }))
  }
  window.addEventListener("scrollend", announce, { once: true })
  window.setTimeout(announce, 700)
}

// Counts the arrivals at a section and says whether the last one is recent (about 1.5 s): the section uses it
// to show its ring and dark label, and to play its own cue each time the count changes
export function useArrival(id: SectionId): { arrivals: number; isLit: boolean } {
  const [arrivals, setArrivals] = useState(0)
  const [isLit, setIsLit] = useState(false)

  useEffect(() => {
    let timer = 0
    const onArrival = (event: Event) => {
      if ((event as CustomEvent<SectionId>).detail !== id) return
      setArrivals((n) => n + 1)
      setIsLit(true)
      window.clearTimeout(timer)
      timer = window.setTimeout(() => setIsLit(false), ARRIVAL_SHOWN_MS)
    }
    window.addEventListener(ARRIVAL_EVENT, onArrival)
    return () => {
      window.removeEventListener(ARRIVAL_EVENT, onArrival)
      window.clearTimeout(timer)
    }
  }, [id])

  return { arrivals, isLit }
}

// The section currently in the middle of the screen, for the nav bar (null above the first one)
export function useActiveSection(): SectionId | null {
  const [active, setActive] = useState<SectionId | null>(null)

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) setActive(entry.target.id as SectionId)
        }
      },
      { rootMargin: "-45% 0px -50% 0px" },
    )
    for (const id of SECTION_IDS) {
      const section = document.getElementById(id)
      if (section) observer.observe(section)
    }
    return () => observer.disconnect()
  }, [])

  return active
}

// Becomes true the first time the element is well into view, and stays true. Returns the ref to put on it
export function useSeenOnce<T extends Element>(threshold = 0.35): [(element: T | null) => void, boolean] {
  const [element, setElement] = useState<T | null>(null)
  const [seen, setSeen] = useState(false)

  useEffect(() => {
    if (!element || seen) return
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setSeen(true)
          observer.disconnect()
        }
      },
      { threshold },
    )
    observer.observe(element)
    return () => observer.disconnect()
  }, [element, seen, threshold])

  return [setElement, seen]
}

// Scroll the page so an element sits in the middle of the screen
export function scrollToCentre(element: Element | null) {
  if (!element) return
  const box = element.getBoundingClientRect()
  const top = window.scrollY + box.top - (window.innerHeight - box.height) / 2
  window.scrollTo({ top: Math.max(0, top), behavior: prefersReducedMotion() ? "auto" : "smooth" })
}

// Scroll just enough for an element to be fully in sight under the nav bar (its top first, if it's too tall)
export function scrollIntoSight(element: Element | null, navHeight = 80, margin = 24) {
  if (!element) return
  const box = element.getBoundingClientRect()
  if (box.top >= navHeight && box.bottom <= window.innerHeight - margin) return
  const fits = box.height <= window.innerHeight - navHeight - margin
  const top = window.scrollY + (fits ? box.bottom - window.innerHeight + margin : box.top - navHeight)
  window.scrollTo({ top, behavior: prefersReducedMotion() ? "auto" : "smooth" })
}

// True on a mouse or trackpad (drag and drop is offered), false on a touch screen (tap only)
export function useFinePointer(): boolean {
  const [fine, setFine] = useState(() => typeof window !== "undefined" && window.matchMedia("(pointer: fine)").matches)

  useEffect(() => {
    const query = window.matchMedia("(pointer: fine)")
    const onChange = () => setFine(query.matches)
    query.addEventListener("change", onChange)
    return () => query.removeEventListener("change", onChange)
  }, [])

  return fine
}
