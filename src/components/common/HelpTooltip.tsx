import { useState } from "react";
import { HelpCircle } from "lucide-react";

interface HelpTooltipProps {
  text: string;
  term?: string;
  className?: string;
}

export function HelpTooltip({ text, term, className = "" }: HelpTooltipProps) {
  const [open, setOpen] = useState(false);

  return (
    <span className={`relative inline-flex items-center align-middle ${className}`}>
      <button
        type="button"
        onClick={() => setOpen(!open)}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        className="ml-1 inline-flex text-muted-foreground/70 hover:text-primary transition-colors focus:outline-none"
        aria-label={term ? `Learn more about ${term}` : "Learn more"}
      >
        <HelpCircle size={13} />
      </button>

      {open && (
        <span
          className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 z-50 w-64 p-2.5 text-[11px] leading-relaxed font-sans font-normal text-foreground bg-popover/95 backdrop-blur-md rounded-md border border-border shadow-xl pointer-events-none transition-all"
          role="tooltip"
        >
          {term && <strong className="block text-primary font-semibold mb-0.5">{term}</strong>}
          {text}
        </span>
      )}
    </span>
  );
}
