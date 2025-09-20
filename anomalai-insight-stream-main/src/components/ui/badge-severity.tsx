import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";

interface SeverityBadgeProps {
  severity: 'low' | 'medium' | 'high';
  className?: string;
}

const severityConfig = {
  high: {
    className: "bg-severity-high text-severity-high-foreground hover:bg-severity-high/80",
    label: "High"
  },
  medium: {
    className: "bg-severity-medium text-severity-medium-foreground hover:bg-severity-medium/80",
    label: "Medium"
  },
  low: {
    className: "bg-severity-low text-severity-low-foreground hover:bg-severity-low/80",
    label: "Low"
  }
};

export function SeverityBadge({ severity, className }: SeverityBadgeProps) {
  const config = severityConfig[severity];
  
  return (
    <Badge 
      variant="secondary" 
      className={cn(config.className, className)}
    >
      {config.label}
    </Badge>
  );
}