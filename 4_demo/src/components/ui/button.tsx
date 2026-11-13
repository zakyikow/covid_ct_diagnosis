import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"
import type { ComponentProps } from "react"

import { cn } from "@/lib/utils"

// shadcn/ui button, restyled: round pills in ink or white
const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 rounded-full font-semibold whitespace-nowrap transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ink disabled:pointer-events-none disabled:opacity-50 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        default: "bg-ink text-white hover:bg-ink/85",
        outline: "border border-line bg-card text-ink hover:bg-soft",
      },
      size: {
        default: "min-h-12 px-6 text-base",
        sm: "min-h-10 px-4 text-sm font-medium",
      },
    },
    defaultVariants: { variant: "default", size: "default" },
  },
)

// A button, or with `asChild` a link styled as one
export function Button({
  className,
  variant,
  size,
  asChild = false,
  ...props
}: ComponentProps<"button"> & VariantProps<typeof buttonVariants> & { asChild?: boolean }) {
  const Component = asChild ? Slot : "button"
  return <Component className={cn(buttonVariants({ variant, size }), className)} {...props} />
}
