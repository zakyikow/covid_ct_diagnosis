import type { ComponentProps } from "react"

// Small line icons, drawn in the current text colour. Decorative: hidden from screen readers
function Icon({ children, size = 16, ...props }: ComponentProps<"svg"> & { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      {children}
    </svg>
  )
}

export function ArrowDown(props: ComponentProps<"svg"> & { size?: number }) {
  return (
    <Icon {...props}>
      <path d="M12 5v14M6 13l6 6 6-6" />
    </Icon>
  )
}

export function ArrowRight(props: ComponentProps<"svg"> & { size?: number }) {
  return (
    <Icon {...props}>
      <path d="M4 12h16M14 6l6 6-6 6" />
    </Icon>
  )
}

export function CodeIcon(props: ComponentProps<"svg"> & { size?: number }) {
  return (
    <Icon {...props}>
      <path d="M8 7l-5 5 5 5M16 7l5 5-5 5" />
    </Icon>
  )
}

export function CheckCircle(props: ComponentProps<"svg"> & { size?: number }) {
  return (
    <Icon {...props}>
      <circle cx="12" cy="12" r="9" />
      <path d="M8 12.5l2.5 2.5L16 9.5" />
    </Icon>
  )
}

export function CrossCircle(props: ComponentProps<"svg"> & { size?: number }) {
  return (
    <Icon {...props}>
      <circle cx="12" cy="12" r="9" />
      <path d="M9 9l6 6M15 9l-6 6" />
    </Icon>
  )
}
