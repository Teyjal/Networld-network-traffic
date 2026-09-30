import type { ButtonHTMLAttributes, ReactNode } from "react";
import { cn } from "@/lib/utils";

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "outline" | "danger" | "ghost" | "accent";
  size?: "sm" | "md" | "lg" | string;
  children: ReactNode;
}

export function Button({
  className,
  variant = "primary",
  size = "md",
  children,
  ...props
}: ButtonProps) {
  const sizeClasses =
    size === "sm"
      ? "h-8 px-2.5 text-[11px]"
      : size === "lg"
      ? "h-11 px-5 text-sm"
      : "h-9 px-3.5 text-xs";

  return (
    <button
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-lg font-semibold uppercase tracking-wider transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50 cursor-pointer",
        sizeClasses,
        variant === "primary" &&
          "bg-primary text-primary-foreground shadow-sm hover:opacity-90 dark:bg-cyan-500 dark:text-[#061426] dark:hover:bg-cyan-400",
        variant === "outline" &&
          "border border-[#0b1f3a]/30 bg-card text-[#0b1f3a] hover:border-[#0b1f3a] hover:bg-slate-100 dark:border-border dark:bg-secondary dark:text-foreground dark:hover:border-primary/50 dark:hover:bg-secondary/80",
        variant === "accent" &&
          "bg-[#06b6d4] text-[#061426] font-bold shadow-[0_0_12px_rgba(6,182,212,0.25)] hover:bg-[#22d3ee]",
        variant === "danger" &&
          "bg-destructive text-destructive-foreground hover:bg-destructive/90",
        variant === "ghost" &&
          "bg-transparent text-foreground hover:bg-secondary",
        className
      )}
      {...props}
    >
      {children}
    </button>
  );
}
