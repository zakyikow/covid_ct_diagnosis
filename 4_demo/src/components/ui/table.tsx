import type { ComponentProps } from "react"

import { cn } from "@/lib/utils"

// shadcn/ui table, restyled: thin row lines, numbers in fixed-width digits
export function Table({ className, ...props }: ComponentProps<"table">) {
  return (
    <div className="w-full overflow-x-auto">
      <table className={cn("w-full border-collapse text-[15px] tabular-nums", className)} {...props} />
    </div>
  )
}

export function TableHeader(props: ComponentProps<"thead">) {
  return <thead {...props} />
}

export function TableBody(props: ComponentProps<"tbody">) {
  return <tbody {...props} />
}

export function TableRow({ className, ...props }: ComponentProps<"tr">) {
  return <tr className={cn("border-b border-line-soft last:border-0", className)} {...props} />
}

export function TableHead({ className, ...props }: ComponentProps<"th">) {
  return <th className={cn("px-2 py-3.5 text-left text-[13px] font-medium text-muted", className)} {...props} />
}

// The first cell of a row, which names the row (read by screen readers with each value of the row)
export function TableRowHead({ className, ...props }: ComponentProps<"th">) {
  return <th scope="row" className={cn("px-2 py-3.5 text-left font-normal", className)} {...props} />
}

export function TableCell({ className, ...props }: ComponentProps<"td">) {
  return <td className={cn("px-2 py-3.5", className)} {...props} />
}
