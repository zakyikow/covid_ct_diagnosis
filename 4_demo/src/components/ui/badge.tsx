import { cva, type VariantProps } from "class-variance-authority"
import type { ComponentProps } from "react"

import { cn } from "@/lib/utils"

// shadcn/ui badge, restyled: small rounded labels
const badgeVariants = cva("inline-flex items-center gap-1.5 rounded-full font-medium whitespace-nowrap", {
  variants: {
    variant: {
      soft: "bg-soft text-text",
      outline: "border border-line bg-card text-text",
      onImage: "bg-white font-semibold text-ink",
      gap: "bg-gap-soft font-semibold text-gap",
    },
    size: {
      default: "px-3 py-1 text-[13px]",
      sm: "px-2.5 py-0.5 text-xs",
    },
  },
  defaultVariants: { variant: "soft", size: "default" },
})

export function Badge({ className, variant, size, ...props }: ComponentProps<"span"> & VariantProps<typeof badgeVariants>) {
  return <span className={cn(badgeVariants({ variant, size }), className)} {...props} />
}
