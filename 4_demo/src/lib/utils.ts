import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

// Join class names, letting later Tailwind classes override earlier ones (the shadcn/ui helper)
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
