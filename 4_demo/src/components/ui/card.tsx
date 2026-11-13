import type { ComponentProps } from "react"

import { cn } from "@/lib/utils"

// shadcn/ui card, restyled: white, thin border, large rounded corners
export function Card({ className, ...props }: ComponentProps<"div">) {
  return <div className={cn("rounded-[20px] border border-line bg-card", className)} {...props} />
}
