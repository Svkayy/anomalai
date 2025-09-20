import { useState } from "react";
import { format } from "date-fns";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import { Calendar } from "@/components/ui/calendar";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { CalendarIcon, X, Filter, Search, AlertTriangle, AlertCircle, Info, Camera, Smartphone, Glasses } from "lucide-react";
import { cn } from "@/lib/utils";
import { ReportFilters } from "@/types/report";

interface FilterPanelProps {
  filters: ReportFilters;
  onFiltersChange: (filters: ReportFilters) => void;
  isCollapsed?: boolean;
  onToggleCollapse?: () => void;
}

const severityOptions = [
  { value: 'high', label: 'High', icon: AlertTriangle, color: 'text-severity-high', bgColor: 'bg-severity-high/10', borderColor: 'border-severity-high/20' },
  { value: 'medium', label: 'Medium', icon: AlertCircle, color: 'text-severity-medium', bgColor: 'bg-severity-medium/10', borderColor: 'border-severity-medium/20' },
  { value: 'low', label: 'Low', icon: Info, color: 'text-severity-low', bgColor: 'bg-severity-low/10', borderColor: 'border-severity-low/20' }
];

const deviceTypeOptions = [
  { value: 'cctv', label: 'CCTV Camera', icon: Camera, color: 'text-primary', bgColor: 'bg-primary/10', borderColor: 'border-primary/20' },
  { value: 'phone', label: 'Mobile Device', icon: Smartphone, color: 'text-accent', bgColor: 'bg-accent/10', borderColor: 'border-accent/20' },
  { value: 'glasses', label: 'Smart Glasses', icon: Glasses, color: 'text-secondary', bgColor: 'bg-secondary/10', borderColor: 'border-secondary/20' }
];

export function FilterPanel({ filters, onFiltersChange, isCollapsed, onToggleCollapse }: FilterPanelProps) {
  const [localFilters, setLocalFilters] = useState<ReportFilters>(filters);

  const handleFilterChange = (key: keyof ReportFilters, value: any) => {
    const newFilters = { ...localFilters, [key]: value };
    setLocalFilters(newFilters);
    onFiltersChange(newFilters);
  };

  const handleArrayFilterChange = (key: keyof ReportFilters, value: string, checked: boolean) => {
    const currentArray = (localFilters[key] as string[]) || [];
    const newArray = checked 
      ? [...currentArray, value]
      : currentArray.filter(item => item !== value);
    
    handleFilterChange(key, newArray.length > 0 ? newArray : undefined);
  };

  const clearAllFilters = () => {
    const clearedFilters = { page: 1, limit: 10 };
    setLocalFilters(clearedFilters);
    onFiltersChange(clearedFilters);
  };

  const hasActiveFilters = Object.keys(localFilters).some(key => 
    key !== 'page' && key !== 'limit' && localFilters[key as keyof ReportFilters]
  );

  if (isCollapsed) {
    return (
      <div className="mb-4">
        <Button
          variant="outline"
          size="lg"
          onClick={onToggleCollapse}
          className="flex items-center gap-2 bg-accent/5 hover:bg-accent/10 border-accent/20"
        >
          <Filter className="w-4 h-4" />
          <span>Show Filters</span>
          {hasActiveFilters && (
            <Badge variant="secondary" className="ml-2 bg-primary text-primary-foreground">
              {Object.keys(localFilters).filter(key => 
                key !== 'page' && key !== 'limit' && localFilters[key as keyof ReportFilters]
              ).length}
            </Badge>
          )}
        </Button>
      </div>
    );
  }

  return (
    <Card className="w-80 flex-shrink-0 bg-gradient-to-br from-background to-accent/5 border-accent/20 shadow-lg">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4 bg-gradient-to-r from-accent/10 to-primary/5 border-b border-accent/20">
        <div className="flex items-center gap-2">
          <div className="p-2 bg-primary/10 rounded-lg">
            <Filter className="w-4 h-4 text-primary" />
          </div>
          <CardTitle className="text-lg font-semibold">Filter Reports</CardTitle>
        </div>
        <Button variant="ghost" size="sm" onClick={onToggleCollapse} className="hover:bg-accent/10">
          <X className="w-4 h-4" />
        </Button>
      </CardHeader>
      <CardContent className="space-y-6 p-6">
        {/* Search */}
        <div className="space-y-3">
          <Label className="text-sm font-semibold flex items-center gap-2">
            <Search className="w-4 h-4 text-primary" />
            Search Reports
          </Label>
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input
              id="search"
              placeholder="Search reports, labels, tags, actors..."
              value={localFilters.q || ''}
              onChange={(e) => handleFilterChange('q', e.target.value || undefined)}
              className="pl-10 bg-accent/5 border-accent/20 focus:border-primary/50"
            />
          </div>
        </div>

        {/* Date Range */}
        <div className="space-y-3">
          <Label className="text-sm font-semibold flex items-center gap-2">
            <CalendarIcon className="w-4 h-4 text-primary" />
            Date Range
          </Label>
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-2">
              <Label className="text-xs text-muted-foreground font-medium">From Date</Label>
              <Popover>
                <PopoverTrigger asChild>
                  <Button
                    variant="outline"
                    className={cn(
                      "w-full justify-start text-left font-normal bg-accent/5 border-accent/20 hover:bg-accent/10 focus:border-primary/50 h-10 px-3 overflow-hidden",
                      !localFilters.from && "text-muted-foreground"
                    )}
                  >
                    <CalendarIcon className="mr-2 h-4 w-4 flex-shrink-0" />
                    <span className="truncate">
                      {localFilters.from ? format(new Date(localFilters.from), "MMM d, yyyy") : "Pick a date"}
                    </span>
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-auto p-0" align="start">
                  <Calendar
                    mode="single"
                    selected={localFilters.from ? new Date(localFilters.from) : undefined}
                    onSelect={(date) => handleFilterChange('from', date ? date.toISOString().split('T')[0] : undefined)}
                    initialFocus
                    className="pointer-events-auto"
                  />
                </PopoverContent>
              </Popover>
            </div>
            <div className="space-y-2">
              <Label className="text-xs text-muted-foreground font-medium">To Date</Label>
              <Popover>
                <PopoverTrigger asChild>
                  <Button
                    variant="outline"
                    className={cn(
                      "w-full justify-start text-left font-normal bg-accent/5 border-accent/20 hover:bg-accent/10 focus:border-primary/50 h-10 px-3 overflow-hidden",
                      !localFilters.to && "text-muted-foreground"
                    )}
                  >
                    <CalendarIcon className="mr-2 h-4 w-4 flex-shrink-0" />
                    <span className="truncate">
                      {localFilters.to ? format(new Date(localFilters.to), "MMM d, yyyy") : "Pick a date"}
                    </span>
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-auto p-0" align="start">
                  <Calendar
                    mode="single"
                    selected={localFilters.to ? new Date(localFilters.to) : undefined}
                    onSelect={(date) => handleFilterChange('to', date ? date.toISOString().split('T')[0] : undefined)}
                    initialFocus
                    className="pointer-events-auto"
                  />
                </PopoverContent>
              </Popover>
            </div>
          </div>
        </div>

        {/* Severity */}
        <div className="space-y-3">
          <Label className="text-sm font-semibold">Risk Severity Levels</Label>
          <div className="space-y-3">
            {severityOptions.map((option) => {
              const Icon = option.icon;
              const isChecked = (localFilters.severity || []).includes(option.value);
              return (
                <div 
                  key={option.value} 
                  className={`flex items-center space-x-3 p-3 rounded-lg border transition-all cursor-pointer hover:scale-[1.02] ${
                    isChecked 
                      ? `${option.bgColor} ${option.borderColor} shadow-sm` 
                      : 'bg-accent/5 border-border hover:bg-accent/10'
                  }`}
                  onClick={() => handleArrayFilterChange('severity', option.value, !isChecked)}
                >
                  <Checkbox
                    id={`severity-${option.value}`}
                    checked={isChecked}
                    onCheckedChange={(checked) => 
                      handleArrayFilterChange('severity', option.value, !!checked)
                    }
                    className="border-2"
                  />
                  <Icon className={`w-4 h-4 ${isChecked ? option.color : 'text-muted-foreground'}`} />
                  <Label htmlFor={`severity-${option.value}`} className={`text-sm font-medium cursor-pointer ${isChecked ? option.color : 'text-foreground'}`}>
                    {option.label} Risk
                  </Label>
                </div>
              );
            })}
          </div>
        </div>

        {/* Device Type */}
        <div className="space-y-3">
          <Label className="text-sm font-semibold">Device Sources</Label>
          <div className="space-y-3">
            {deviceTypeOptions.map((option) => {
              const Icon = option.icon;
              const isChecked = (localFilters.deviceType || []).includes(option.value);
              return (
                <div 
                  key={option.value} 
                  className={`flex items-center space-x-3 p-3 rounded-lg border transition-all cursor-pointer hover:scale-[1.02] ${
                    isChecked 
                      ? `${option.bgColor} ${option.borderColor} shadow-sm` 
                      : 'bg-accent/5 border-border hover:bg-accent/10'
                  }`}
                  onClick={() => handleArrayFilterChange('deviceType', option.value, !isChecked)}
                >
                  <Checkbox
                    id={`device-${option.value}`}
                    checked={isChecked}
                    onCheckedChange={(checked) => 
                      handleArrayFilterChange('deviceType', option.value, !!checked)
                    }
                    className="border-2"
                  />
                  <Icon className={`w-4 h-4 ${isChecked ? option.color : 'text-muted-foreground'}`} />
                  <Label htmlFor={`device-${option.value}`} className={`text-sm font-medium cursor-pointer ${isChecked ? option.color : 'text-foreground'}`}>
                    {option.label}
                  </Label>
                </div>
              );
            })}
          </div>
        </div>

        {/* Clear Filters */}
        {hasActiveFilters && (
          <div className="pt-2 border-t border-accent/20">
            <Button 
              variant="outline" 
              onClick={clearAllFilters}
              className="w-full bg-blue-500/10 border-blue-500/30 text-blue-600 hover:bg-blue-500/20 hover:text-blue-700 hover:border-blue-500/40 transition-all"
            >
              <X className="w-4 h-4 mr-2" />
              Clear All Filters
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}