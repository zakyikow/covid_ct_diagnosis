import { cn } from "@/lib/utils"

// The small numbered label above each section, matching the numbers of the nav bar. Right after a jump to the
// section it turns into a dark pill for a moment
export function SectionLabel({ children, lit = false, className }: { children: string; lit?: boolean; className?: string }) {
  return (
    <p className={cn("font-mono text-[13px] text-muted", className)}>
      <span className={cn("rounded-full px-2.5 py-1 transition-colors duration-500", lit && "bg-ink text-white")}>
        {children}
      </span>
    </p>
  )
}
