import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { SeverityBadge } from "@/components/ui/badge-severity";
import { Copy } from "lucide-react";
import { Observation } from "@/types/report";
import { useToast } from "@/hooks/use-toast";

interface ObservationItemProps {
  observation: Observation;
  className?: string;
}

export function ObservationItem({ observation, className }: ObservationItemProps) {
  const { toast } = useToast();

  const copyToClipboard = (text: string, type: string) => {
    navigator.clipboard.writeText(text);
    toast({
      title: "Copied to clipboard",
      description: `${type} copied: ${text}`,
    });
  };

  // Get severity-based background color
  const getSeverityBg = (severity: string) => {
    switch (severity) {
      case 'high': return 'bg-severity-high/5 border-severity-high/20';
      case 'medium': return 'bg-severity-medium/5 border-severity-medium/20';
      case 'low': return 'bg-severity-low/5 border-severity-low/20';
      default: return 'bg-muted/20 border-border';
    }
  };

  return (
    <div className={`border rounded-lg p-5 space-y-4 ${getSeverityBg(observation.severity)} ${className}`}>
      {/* Timeframe and Frames */}
      <div className="flex items-center gap-4 text-sm">
        <div className="flex items-center gap-2">
          <span className="font-medium text-muted-foreground">Time:</span>
          <code className="bg-muted px-2 py-1 rounded text-xs">
            {observation.timeframe}
          </code>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => copyToClipboard(observation.timeframe, "Timeframe")}
            className="h-6 w-6 p-0"
          >
            <Copy className="w-3 h-3" />
          </Button>
        </div>
        
        <div className="flex items-center gap-2">
          <span className="font-medium text-muted-foreground">Frames:</span>
          <code className="bg-muted px-2 py-1 rounded text-xs">
            {observation.frames}
          </code>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => copyToClipboard(observation.frames, "Frames")}
            className="h-6 w-6 p-0"
          >
            <Copy className="w-3 h-3" />
          </Button>
        </div>
      </div>

      {/* Label and Severity */}
      <div className="flex items-start gap-3">
        <div className="flex-1">
          <h3 className="font-medium text-base mb-2">{observation.label.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</h3>
          <p className="text-sm text-muted-foreground leading-relaxed">
            Safety observation detected during the specified timeframe. This incident requires attention based on the severity level and associated risk factors.
          </p>
        </div>
        <SeverityBadge severity={observation.severity} />
      </div>

      {/* Tags */}
      {observation.tags.length > 0 && (
        <div className="bg-accent/5 p-3 rounded-lg space-y-2">
          <span className="text-sm font-semibold text-foreground">Related Tags</span>
          <div className="flex flex-wrap gap-2">
            {observation.tags.map((tag, index) => (
              <div 
                key={index} 
                className="text-xs bg-accent/10 border border-accent/30 px-3 py-2 rounded-sm font-medium text-accent-foreground"
              >
                {tag.replace(/_/g, ' ')}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Actors */}
      {observation.actors.length > 0 && (
        <div className="bg-primary/5 p-3 rounded-lg space-y-2">
          <span className="text-sm font-semibold text-foreground">Involved Parties</span>
          <div className="flex flex-wrap gap-2">
            {observation.actors.map((actor, index) => (
              <Badge key={index} variant="secondary" className="text-xs bg-primary/10 text-primary border-primary/20">
                {actor.replace('#', ' #')}
              </Badge>
            ))}
          </div>
          <p className="text-xs text-muted-foreground">
            Individuals and objects identified in the safety incident
          </p>
        </div>
      )}

      {/* Reasons */}
      {observation.reasons.length > 0 && (
        <div className="bg-status-warning/5 p-3 rounded-lg space-y-2">
          <span className="text-sm font-semibold text-foreground">Risk Factors</span>
          <div className="flex flex-wrap gap-2">
            {observation.reasons.map((reason, index) => (
              <Badge key={index} variant="outline" className="text-xs bg-status-warning/15 text-status-warning border-status-warning/30">
                {reason}
              </Badge>
            ))}
          </div>
          <p className="text-xs text-muted-foreground">
            Contributing factors that led to this safety concern
          </p>
        </div>
      )}

      {/* Actions */}
      {observation.actions.length > 0 && (
        <div className="bg-status-info/5 p-3 rounded-lg space-y-2">
          <span className="text-sm font-semibold text-foreground">Recommended Actions</span>
          <div className="flex flex-wrap gap-2">
            {observation.actions.map((action, index) => (
              <Badge key={index} variant="outline" className="text-xs bg-status-info/15 text-status-info border-status-info/30">
                {action}
              </Badge>
            ))}
          </div>
          <p className="text-xs text-muted-foreground">
            Suggested interventions to address this safety issue
          </p>
        </div>
      )}
    </div>
  );
}