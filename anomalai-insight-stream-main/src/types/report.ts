export interface Video {
  duration: number;
  capturedAt: string;
  deviceType: 'cctv' | 'phone' | 'glasses';
}

export interface Observation {
  timeframe: string;
  frames: string;
  label: string;
  severity: 'low' | 'medium' | 'high';
  actors: string[];
  reasons: string[];
  actions: string[];
  tags: string[];
}

export interface SummaryCounts {
  totalObservations: number;
  low: number;
  medium: number;
  high: number;
}

export interface Report {
  reportId: string;
  video: Video;
  createdAt: string;
  summaryCounts: SummaryCounts;
  observations: Observation[];
}

export interface ReportsResponse {
  reports: Report[];
  total: number;
  page: number;
  limit: number;
}

export interface AssetUrlResponse {
  url: string;
}

export interface ReportFilters {
  from?: string;
  to?: string;
  severity?: string[];
  label?: string[];
  tag?: string[];
  deviceType?: string[];
  q?: string;
  page?: number;
  limit?: number;
}