import { Report, ReportsResponse, AssetUrlResponse, ReportFilters } from '@/types/report';

const BASE_URL = '/api';
const USE_FIXTURES = true; // Feature flag for development

// Mock data for development
const mockReports: Report[] = [
  {
    reportId: "wh3_cctv07_2025-09-19_2112",
    video: { 
      duration: 86.4, 
      capturedAt: "2025-09-19T21:12:00Z", 
      deviceType: "cctv" 
    },
    createdAt: "2025-09-19T21:12:12Z",
    summaryCounts: { totalObservations: 2, low: 0, medium: 1, high: 1 },
    observations: [
      {
        timeframe: "00:05–00:08",
        frames: "153–240",
        label: "forklift_proximity",
        severity: "high",
        actors: ["person#47", "forklift#12"],
        reasons: ["distance < 2m", "blind corner"],
        actions: ["alert operator", "activate slow zone"],
        tags: ["forklift", "near_miss", "pedestrian"]
      },
      {
        timeframe: "00:21–00:23",
        frames: "651–690",
        label: "ppe_check",
        severity: "medium",
        actors: ["person#47"],
        reasons: ["no helmet"],
        actions: ["notify supervisor"],
        tags: ["ppe"]
      }
    ]
  },
  {
    reportId: "wh3_phone02_2025-09-19_1545",
    video: { 
      duration: 124.8, 
      capturedAt: "2025-09-19T15:45:00Z", 
      deviceType: "phone" 
    },
    createdAt: "2025-09-19T15:45:15Z",
    summaryCounts: { totalObservations: 3, low: 1, medium: 2, high: 0 },
    observations: [
      {
        timeframe: "00:12–00:15",
        frames: "360–450",
        label: "equipment_misuse",
        severity: "medium",
        actors: ["person#23"],
        reasons: ["improper lifting technique"],
        actions: ["safety training reminder"],
        tags: ["lifting", "ergonomics"]
      },
      {
        timeframe: "00:45–00:48",
        frames: "1350–1440",
        label: "slip_hazard",
        severity: "low",
        actors: ["person#23"],
        reasons: ["wet floor"],
        actions: ["place warning signs"],
        tags: ["spill", "cleanup"]
      },
      {
        timeframe: "01:50–01:55",
        frames: "3300–3450",
        label: "vehicle_speed",
        severity: "medium",
        actors: ["forklift#08"],
        reasons: ["exceeding speed limit"],
        actions: ["operator counseling"],
        tags: ["speed", "warehouse"]
      }
    ]
  }
];

const mockAssetUrl = {
  url: "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
};

function buildQueryString(filters: ReportFilters): string {
  const params = new URLSearchParams();
  
  if (filters.from) params.append('from', filters.from);
  if (filters.to) params.append('to', filters.to);
  if (filters.severity?.length) {
    filters.severity.forEach(s => params.append('severity[]', s));
  }
  if (filters.label?.length) {
    filters.label.forEach(l => params.append('label[]', l));
  }
  if (filters.tag?.length) {
    filters.tag.forEach(t => params.append('tag[]', t));
  }
  if (filters.deviceType?.length) {
    filters.deviceType.forEach(d => params.append('deviceType[]', d));
  }
  if (filters.q) params.append('q', filters.q);
  if (filters.page) params.append('page', filters.page.toString());
  if (filters.limit) params.append('limit', filters.limit.toString());
  
  return params.toString();
}

export async function getReports(filters: ReportFilters = {}): Promise<ReportsResponse> {
  if (USE_FIXTURES) {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 300));
    
    let filteredReports = [...mockReports];
    
    // Apply filters
    if (filters.severity?.length) {
      filteredReports = filteredReports.filter(report => 
        report.observations.some(obs => filters.severity!.includes(obs.severity))
      );
    }
    
    if (filters.deviceType?.length) {
      filteredReports = filteredReports.filter(report =>
        filters.deviceType!.includes(report.video.deviceType)
      );
    }
    
    if (filters.q) {
      const query = filters.q.toLowerCase();
      filteredReports = filteredReports.filter(report =>
        report.reportId.toLowerCase().includes(query) ||
        report.observations.some(obs =>
          obs.label.toLowerCase().includes(query) ||
          obs.tags.some(tag => tag.toLowerCase().includes(query)) ||
          obs.actors.some(actor => actor.toLowerCase().includes(query))
        )
      );
    }
    
    return {
      reports: filteredReports,
      total: filteredReports.length,
      page: filters.page || 1,
      limit: filters.limit || 10
    };
  }
  
  const queryString = buildQueryString(filters);
  const response = await fetch(`${BASE_URL}/reports?${queryString}`);
  
  if (!response.ok) {
    throw new Error(`Failed to fetch reports: ${response.statusText}`);
  }
  
  return response.json();
}

export async function getReport(reportId: string): Promise<Report> {
  if (USE_FIXTURES) {
    await new Promise(resolve => setTimeout(resolve, 200));
    const report = mockReports.find(r => r.reportId === reportId);
    if (!report) {
      throw new Error(`Report not found: ${reportId}`);
    }
    return report;
  }
  
  const response = await fetch(`${BASE_URL}/reports/${reportId}`);
  
  if (!response.ok) {
    throw new Error(`Failed to fetch report: ${response.statusText}`);
  }
  
  return response.json();
}

export async function getAssetUrl(assetId: string): Promise<AssetUrlResponse> {
  if (USE_FIXTURES) {
    await new Promise(resolve => setTimeout(resolve, 100));
    return mockAssetUrl;
  }
  
  const response = await fetch(`${BASE_URL}/assets/${assetId}/url`);
  
  if (!response.ok) {
    throw new Error(`Failed to fetch asset URL: ${response.statusText}`);
  }
  
  return response.json();
}