import { useState, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import { FilterPanel } from "@/components/filter-panel";
import { StatsCards } from "@/components/stats-cards";
import { ReportTile } from "@/components/report-tile";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { AlertCircle, Radio } from "lucide-react";
import { getReports } from "@/lib/api-client";
import { Report, ReportFilters, SummaryCounts } from "@/types/report";
import { useToast } from "@/hooks/use-toast";

export default function Dashboard() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [reports, setReports] = useState<Report[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filterPanelCollapsed, setFilterPanelCollapsed] = useState(false);
  const { toast } = useToast();

  // Parse filters from URL
  const getFiltersFromUrl = (): ReportFilters => {
    const filters: ReportFilters = {};
    
    const from = searchParams.get('from');
    const to = searchParams.get('to');
    const q = searchParams.get('q');
    
    if (from) filters.from = from;
    if (to) filters.to = to;
    if (q) filters.q = q;
    
    const severity = searchParams.getAll('severity');
    if (severity.length) filters.severity = severity;
    
    const deviceType = searchParams.getAll('deviceType');
    if (deviceType.length) filters.deviceType = deviceType;
    
    const label = searchParams.getAll('label');
    if (label.length) filters.label = label;
    
    const tag = searchParams.getAll('tag');
    if (tag.length) filters.tag = tag;
    
    return filters;
  };

  const [filters, setFilters] = useState<ReportFilters>(getFiltersFromUrl());

  // Update URL when filters change
  const updateFilters = (newFilters: ReportFilters) => {
    setFilters(newFilters);
    
    const params = new URLSearchParams();
    
    if (newFilters.from) params.set('from', newFilters.from);
    if (newFilters.to) params.set('to', newFilters.to);
    if (newFilters.q) params.set('q', newFilters.q);
    
    newFilters.severity?.forEach(s => params.append('severity', s));
    newFilters.deviceType?.forEach(d => params.append('deviceType', d));
    newFilters.label?.forEach(l => params.append('label', l));
    newFilters.tag?.forEach(t => params.append('tag', t));
    
    setSearchParams(params);
  };

  // Calculate summary counts from filtered reports
  const calculateSummaryCounts = (reports: Report[]): SummaryCounts => {
    let totalObservations = 0;
    let high = 0;
    let medium = 0;
    let low = 0;

    reports.forEach(report => {
      totalObservations += report.summaryCounts.totalObservations;
      high += report.summaryCounts.high;
      medium += report.summaryCounts.medium;
      low += report.summaryCounts.low;
    });

    return { totalObservations, high, medium, low };
  };

  // Fetch reports
  const fetchReports = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await getReports(filters);
      setReports(response.reports);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch reports';
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

  useEffect(() => {
    fetchReports();
  }, [filters]);

  const summaryCounts = calculateSummaryCounts(reports);

  const renderContent = () => {
    if (loading) {
      return (
        <div className="space-y-4">
          {Array.from({ length: 3 }).map((_, i) => (
            <Skeleton key={i} className="h-32 w-full" />
          ))}
        </div>
      );
    }

    if (error) {
      return (
        <div className="flex flex-col items-center justify-center py-12 space-y-4">
          <AlertCircle className="w-12 h-12 text-destructive" />
          <div className="text-center">
            <h3 className="text-lg font-semibold">Error Loading Reports</h3>
            <p className="text-muted-foreground">{error}</p>
          </div>
          <Button onClick={fetchReports} variant="outline">
            Try Again
          </Button>
        </div>
      );
    }

    if (reports.length === 0) {
      return (
        <div className="flex flex-col items-center justify-center py-12 space-y-4">
          <div className="text-center">
            <h3 className="text-lg font-semibold">No Reports Found</h3>
            <p className="text-muted-foreground">
              No reports match the current filters.
            </p>
          </div>
          <Button onClick={() => updateFilters({})} variant="outline">
            Clear Filters
          </Button>
        </div>
      );
    }

    return (
      <div className="space-y-4">
        {reports.map((report) => (
          <ReportTile key={report.reportId} report={report} />
        ))}
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 sticky top-0 z-10">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <h1 className="text-2xl font-bold">AnomalAI</h1>
              <div className="flex items-center gap-2 px-2 py-1 bg-status-success/10 text-status-success rounded-full text-sm">
                <Radio className="w-3 h-3 animate-pulse" />
                Live
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-6">
        <div className="flex gap-6">
          {/* Filter Panel */}
          <FilterPanel
            filters={filters}
            onFiltersChange={updateFilters}
            isCollapsed={filterPanelCollapsed}
            onToggleCollapse={() => setFilterPanelCollapsed(!filterPanelCollapsed)}
          />

          {/* Main Content */}
          <div className="flex-1 space-y-6">
            <StatsCards counts={summaryCounts} />
            {renderContent()}
          </div>
        </div>
      </div>
    </div>
  );
}