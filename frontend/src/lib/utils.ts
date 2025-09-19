import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatDate(date: string | Date): string {
  const d = new Date(date)
  return d.toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  })
}

export function calculateAge(birthDate: string | Date): number {
  const today = new Date()
  const birth = new Date(birthDate)
  let age = today.getFullYear() - birth.getFullYear()
  const monthDiff = today.getMonth() - birth.getMonth()

  if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birth.getDate())) {
    age--
  }

  return age
}

export function getRiskLevelColor(risk: string): string {
  switch (risk.toLowerCase()) {
    case "high":
      return "text-destructive bg-destructive/10"
    case "medium":
      return "text-orange-600 bg-orange-100"
    case "low":
      return "text-green-600 bg-green-100"
    default:
      return "text-muted-foreground bg-muted"
  }
}
