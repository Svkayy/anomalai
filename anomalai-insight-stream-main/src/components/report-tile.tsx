import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { ChevronDown, ChevronRight, ExternalLink } from "lucide-react";
import { Report } from "@/types/report";
import { SeverityBadge } from "@/components/ui/badge-severity";
import { DeviceIcon } from "@/components/ui/device-icon";
import { useNavigate } from "react-router-dom";

interface ReportTileProps {
  report: Report;
}

function formatDuration(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.floor(seconds % 60);
  return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
}

function formatDateTime(dateString: string): string {
  return new Date(dateString).toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });
}

function generateReadableName(reportId: string): string {
  // Extract location and device info from reportId
  const parts = reportId.split('_');
  if (parts.length >= 3) {
    const location = parts[0].replace(/[0-9]/g, '').toUpperCase();
    const device = parts[1].replace(/[0-9]/g, '').toUpperCase();
    const deviceNum = parts[1].match(/[0-9]+/)?.[0] || '';
    const date = new Date(parts[2] + 'T' + (parts[3] || '0000').substring(0, 2) + ':' + (parts[3] || '0000').substring(2, 4));
    
    const locationMap: { [key: string]: string } = {
      'WH': 'Warehouse',
      'DC': 'Distribution Center',
      'FAC': 'Factory',
      'OFF': 'Office'
    };
    
    const deviceMap: { [key: string]: string } = {
      'CCTV': 'Camera',
      'PHONE': 'Mobile',
      'GLASSES': 'Smart Glasses'
    };
    
    const locationName = locationMap[location] || location;
    const deviceName = deviceMap[device] || device;
    
    return `${locationName} ${deviceName} ${deviceNum}`;
  }
  
  return reportId;
}

export function ReportTile({ report }: ReportTileProps) {
  const [isOpen, setIsOpen] = useState(false);
  const navigate = useNavigate();

  const handleViewDetails = () => {
    navigate(`/reports/${report.reportId}`);
  };

  // Get top 3 observations for preview
  const previewObservations = report.observations
    .sort((a, b) => {
      const severityOrder = { high: 3, medium: 2, low: 1 };
      return severityOrder[b.severity] - severityOrder[a.severity];
    })
    .slice(0, 3);

  return (
    <Card className="hover:bg-card/80 transition-colors">
      <Collapsible open={isOpen} onOpenChange={setIsOpen}>
        <CollapsibleTrigger asChild>
          <CardHeader className="cursor-pointer">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                {isOpen ? (
                  <ChevronDown className="w-4 h-4 text-muted-foreground" />
                ) : (
                  <ChevronRight className="w-4 h-4 text-muted-foreground" />
                )}
                <div className="flex flex-col">
                  <CardTitle className="text-lg font-semibold">
                    {generateReadableName(report.reportId)}
                  </CardTitle>
                  <p className="text-xs text-muted-foreground font-mono">{report.reportId}</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <DeviceIcon deviceType={report.video.deviceType} className="w-4 h-4" />
                <Badge variant="outline" className="text-xs">
                  {report.video.deviceType.toUpperCase()}
                </Badge>
              </div>
            </div>
            
            <div className="flex items-center justify-between text-sm text-muted-foreground">
              <div className="flex items-center gap-4">
                <span>{formatDateTime(report.video.capturedAt)}</span>
                <span>{formatDuration(report.video.duration)}</span>
              </div>
              
              <div className="flex items-center gap-4">
                <div className="flex items-center justify-center gap-1.5">
                  <div className="w-2 h-2 rounded-full bg-severity-high"></div>
                  <span className="text-xs text-muted-foreground">High:</span>
                  <span className="font-semibold text-severity-high">{report.summaryCounts.high}</span>
                </div>
                <div className="flex items-center justify-center gap-1.5">
                  <div className="w-2 h-2 rounded-full bg-severity-medium"></div>
                  <span className="text-xs text-muted-foreground">Medium:</span>
                  <span className="font-semibold text-severity-medium">{report.summaryCounts.medium}</span>
                </div>
                <div className="flex items-center justify-center gap-1.5">
                  <div className="w-2 h-2 rounded-full bg-severity-low"></div>
                  <span className="text-xs text-muted-foreground">Low:</span>
                  <span className="font-semibold text-severity-low">{report.summaryCounts.low}</span>
                </div>
                <div className="flex items-center justify-center gap-1.5 ml-2 pl-2 border-l border-border">
                  <span className="text-xs text-muted-foreground">Total:</span>
                  <span className="font-semibold">{report.summaryCounts.totalObservations}</span>
                </div>
              </div>
            </div>
          </CardHeader>
        </CollapsibleTrigger>
        
        <CollapsibleContent>
          <CardContent className="pt-0">
            <div className="space-y-4">
              <div className="text-sm font-medium text-muted-foreground">
                Top Observations Preview
              </div>
              
              <div className="space-y-3">
                {previewObservations.map((observation, index) => (
                  <div key={index} className="flex items-center justify-between p-4 bg-accent/20 border border-accent/30 rounded-lg shadow-sm hover:bg-accent/30 hover:border-accent/40 transition-all duration-200 cursor-pointer">
                    <div className="flex items-center gap-4">
                      <div className="bg-accent/10 px-3 py-1.5 rounded-md border">
                        <code className="text-xs font-mono font-medium">
                          {observation.timeframe}
                        </code>
                      </div>
                      <span className="font-medium text-foreground">{observation.label.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</span>
                      <SeverityBadge severity={observation.severity} />
                    </div>
                    <div className="flex gap-2">
                      {observation.tags.slice(0, 2).map((tag, tagIndex) => (
                        <div 
                          key={tagIndex} 
                          className="text-xs bg-secondary border border-border px-3 py-1.5 rounded-md font-medium shadow-sm"
                        >
                          {tag}
                        </div>
                      ))}
                      {observation.tags.length > 2 && (
                        <div className="text-xs bg-accent border border-border px-3 py-1.5 rounded-md font-medium shadow-sm">
                          +{observation.tags.length - 2}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
              
              <Button 
                onClick={handleViewDetails} 
                className="w-full bg-primary hover:bg-primary/90 text-primary-foreground"
                variant="default"
              >
                <ExternalLink className="w-4 h-4 mr-2" />
                View Full Details
              </Button>
            </div>
          </CardContent>
        </CollapsibleContent>
      </Collapsible>
    </Card>
  );
}