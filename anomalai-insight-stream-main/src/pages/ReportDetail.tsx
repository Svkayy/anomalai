import { useState, useEffect } from "react";
import { useParams, useNavigate, useSearchParams } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { ArrowLeft, ChevronDown, ChevronUp, Copy, Code } from "lucide-react";
import { VideoPlayer } from "@/components/video-player";
import { ObservationItem } from "@/components/observation-item";
import { DeviceIcon } from "@/components/ui/device-icon";
import { Skeleton } from "@/components/ui/skeleton";
import { getReport, getAssetUrl } from "@/lib/api-client";
import { Report } from "@/types/report";
import { useToast } from "@/hooks/use-toast";

export default function ReportDetail() {
  const { reportId } = useParams<{ reportId: string }>();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [report, setReport] = useState<Report | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [videoLoading, setVideoLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showRawJson, setShowRawJson] = useState(false);
  const { toast } = useToast();

  useEffect(() => {
    if (!reportId) return;

    const fetchReport = async () => {
      try {
        setLoading(true);
        setError(null);
        
        const reportData = await getReport(reportId);
        setReport(reportData);
        
        // Fetch video URL
        setVideoLoading(true);
        try {
          const assetResponse = await getAssetUrl(reportId);
          setVideoUrl(assetResponse.url);
        } catch (assetError) {
          console.warn('Failed to fetch video URL:', assetError);
          // Don't show error for video, just continue without it
        } finally {
          setVideoLoading(false);
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to fetch report';
        setError(message);
        toast({
          title: "Error",
          description: message,
          variant: "destructive",
        });
      } finally {
        setLoading(false);
      }
    };

    fetchReport();
  }, [reportId, toast]);

  const formatDuration = (seconds: number): string => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  const formatDateTime = (dateString: string): string => {
    return new Date(dateString).toLocaleString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      timeZoneName: 'short'
    });
  };

  const generateReadableName = (reportId: string): string => {
    // Extract location and device info from reportId
    const parts = reportId.split('_');
    if (parts.length >= 3) {
      const location = parts[0].replace(/[0-9]/g, '').toUpperCase();
      const device = parts[1].replace(/[0-9]/g, '').toUpperCase();
      const deviceNum = parts[1].match(/[0-9]+/)?.[0] || '';
      
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
  };

  const copyReportJson = () => {
    if (report) {
      navigator.clipboard.writeText(JSON.stringify(report, null, 2));
      toast({
        title: "Copied to clipboard",
        description: "Report JSON copied to clipboard",
      });
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-background">
        <div className="container mx-auto px-4 py-6">
          <Skeleton className="h-8 w-48 mb-6" />
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Skeleton className="aspect-video" />
            <div className="space-y-4">
              <Skeleton className="h-32" />
              <Skeleton className="h-64" />
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (error || !report) {
    return (
      <div className="min-h-screen bg-background">
        <div className="container mx-auto px-4 py-6">
          <Button
            variant="ghost"
            onClick={() => navigate('/')}
            className="mb-6"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Dashboard
          </Button>
          
          <div className="flex flex-col items-center justify-center py-12 space-y-4">
            <h3 className="text-lg font-semibold">Report Not Found</h3>
            <p className="text-muted-foreground">
              {error || `Report ${reportId} could not be found.`}
            </p>
            <Button onClick={() => navigate('/')} variant="outline">
              Return to Dashboard
            </Button>
          </div>
        </div>
      </div>
    );
  }

  // Get active filters for breadcrumb
  const activeFilters = Array.from(searchParams.entries()).filter(([key]) => 
    ['severity', 'deviceType', 'from', 'to', 'q'].includes(key)
  );

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              onClick={() => navigate('/')}
            >
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Dashboard
            </Button>
            
            {/* Active filters breadcrumb */}
            {activeFilters.length > 0 && (
              <div className="flex items-center gap-2">
                <span className="text-sm text-muted-foreground">Filters:</span>
                {activeFilters.map(([key, value]) => (
                  <Badge key={`${key}-${value}`} variant="outline" className="text-xs">
                    {key}: {value}
                  </Badge>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Report Header */}
        <Card className="mb-6">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div className="flex flex-col">
                <CardTitle className="text-2xl font-bold">{generateReadableName(report.reportId)}</CardTitle>
                <p className="text-sm text-muted-foreground font-mono mt-1">{report.reportId}</p>
              </div>
              <div className="flex items-center gap-3">
                <div className="bg-accent/10 p-2 rounded-lg border">
                  <DeviceIcon deviceType={report.video.deviceType} className="w-5 h-5" />
                </div>
                <Badge variant="outline" className="bg-secondary/50">
                  {report.video.deviceType.toUpperCase()}
                </Badge>
              </div>
            </div>
            
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              <div className="bg-accent/5 p-4 rounded-lg border">
                <span className="text-muted-foreground text-sm">Captured:</span>
                <div className="font-semibold text-foreground mt-1">{formatDateTime(report.video.capturedAt)}</div>
              </div>
              <div className="bg-accent/5 p-4 rounded-lg border">
                <span className="text-muted-foreground text-sm">Duration:</span>
                <div className="font-semibold text-foreground mt-1">{formatDuration(report.video.duration)}</div>
              </div>
              <div className="bg-accent/5 p-4 rounded-lg border">
                <span className="text-muted-foreground text-sm">Total Observations:</span>
                <div className="font-semibold text-foreground mt-1">{report.summaryCounts.totalObservations}</div>
              </div>
              <div className="bg-accent/5 p-4 rounded-lg border">
                <span className="text-muted-foreground text-sm">Severity Breakdown:</span>
                <div className="flex flex-col gap-2 mt-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-1.5">
                      <div className="w-2.5 h-2.5 rounded-full bg-severity-high"></div>
                      <span className="text-xs">High</span>
                    </div>
                    <span className="font-semibold text-severity-high">{report.summaryCounts.high}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-1.5">
                      <div className="w-2.5 h-2.5 rounded-full bg-severity-medium"></div>
                      <span className="text-xs">Medium</span>
                    </div>
                    <span className="font-semibold text-severity-medium">{report.summaryCounts.medium}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-1.5">
                      <div className="w-2.5 h-2.5 rounded-full bg-severity-low"></div>
                      <span className="text-xs">Low</span>
                    </div>
                    <span className="font-semibold text-severity-low">{report.summaryCounts.low}</span>
                  </div>
                </div>
              </div>
            </div>
          </CardHeader>
        </Card>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Video Player */}
          <Card>
            <CardHeader>
              <CardTitle>Video</CardTitle>
            </CardHeader>
            <CardContent>
              {videoLoading ? (
                <Skeleton className="aspect-video w-full" />
              ) : videoUrl ? (
                <VideoPlayer src={videoUrl} observations={report.observations} />
              ) : (
                <div className="aspect-video bg-muted rounded-lg flex items-center justify-center">
                  <div className="text-center">
                    <p className="text-muted-foreground">Video not available</p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Observation Timeline */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0">
              <CardTitle>Observation Timeline</CardTitle>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={copyReportJson}
                >
                  <Copy className="w-4 h-4 mr-2" />
                  Copy JSON
                </Button>
                <Collapsible open={showRawJson} onOpenChange={setShowRawJson}>
                  <CollapsibleTrigger asChild>
                    <Button variant="outline" size="sm">
                      <Code className="w-4 h-4 mr-2" />
                      Raw JSON
                      {showRawJson ? (
                        <ChevronUp className="w-4 h-4 ml-2" />
                      ) : (
                        <ChevronDown className="w-4 h-4 ml-2" />
                      )}
                    </Button>
                  </CollapsibleTrigger>
                  <CollapsibleContent className="mt-4">
                    <pre className="bg-muted p-4 rounded-lg text-xs overflow-auto max-h-64">
                      {JSON.stringify(report, null, 2)}
                    </pre>
                  </CollapsibleContent>
                </Collapsible>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              {report.observations.length === 0 ? (
                <div className="text-center py-8">
                  <p className="text-muted-foreground">No observations recorded</p>
                </div>
              ) : (
                report.observations.map((observation, index) => (
                  <ObservationItem 
                    key={index} 
                    observation={observation}
                  />
                ))
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}