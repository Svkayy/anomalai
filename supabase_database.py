"""
Supabase database module for storing video analysis reports
"""

import os
from datetime import datetime
from typing import Dict, List, Optional
from supabase import create_client, Client
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SupabaseDatabaseManager:
    def __init__(self):
        """Initialize Supabase client"""
        self.client = None
        self.connect()
    
    def connect(self):
        """Establish Supabase connection"""
        try:
            # Get Supabase configuration from environment variables (matching the pattern)
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_KEY")  # Changed from SUPABASE_ANON_KEY to match pattern
            
            if not url or not key:
                print("Supabase environment variables not found:")
                print("  SUPABASE_URL=your_supabase_project_url")
                print("  SUPABASE_KEY=your_supabase_key")
                self.client = None
                return
            
            self.client = create_client(url, key)
            print("Supabase client initialized successfully")
            
        except Exception as e:
            print(f"Error initializing Supabase client: {e}")
            self.client = None
    
    def is_available(self) -> bool:
        """Check if Supabase client is available"""
        return self.client is not None
    
    def has_observations_column(self) -> bool:
        """Check if observations column exists in the reports table"""
        if not self.client:
            return False
        
        try:
            # Try to select the observations column
            response = self.client.table('reports').select('observations').limit(1).execute()
            return True
        except Exception as e:
            if 'observations' in str(e):
                return False
            # Other errors might be temporary, so return True to try
            return True
    
    def create_report(self, video_id: str, video_duration: float, 
                     video_captured_at: datetime, video_device_type: str = "smart glasses", 
                     observations: Optional[Dict] = None) -> str:
        """
        Create a new report entry in the database
        
        Args:
            video_id: Unique identifier for the video
            video_duration: Duration of the video in seconds
            video_captured_at: Timestamp when video was captured
            video_device_type: Type of device used to capture video
            observations: Optional structured observations data (JSONB)
            
        Returns:
            report_id: Unique identifier for the report
        """
        if not self.client:
            raise Exception("Supabase client not available")
        
        try:
            # Generate unique report ID
            report_id = f"report_{video_id}_{int(datetime.now().timestamp())}"
            
            # Prepare data for insertion
            report_data = {
                "report_id": report_id,
                "video_duration": video_duration,
                "video_captured_at": video_captured_at.isoformat(),
                "video_device_type": video_device_type,
                "total_observations": 0,
                "low": 0,
                "medium": 0,
                "high": 0
            }
            
            # Add observations only if provided and column exists
            if observations is not None and self.has_observations_column():
                report_data["observations"] = observations
            elif observations is not None:
                # Store observations in description column as JSON as a workaround
                print("Warning: Observations column not available, storing in description column")
                report_data["description"] = json.dumps(observations, indent=2)
            
            # Insert into Supabase
            response = self.client.table('reports').insert(report_data).execute()
            
            if response.data:
                print(f"Report created successfully: {report_id}")
                return report_id
            else:
                raise Exception("Failed to create report - no data returned")
                
        except Exception as e:
            print(f"Error creating report: {e}")
            raise
    
    def update_report_observations(self, report_id: str, observations_data: Dict, 
                                 structured_observations: Optional[Dict] = None) -> bool:
        """
        Update report with observation counts from AnomalAI analysis
        
        Args:
            report_id: Unique identifier for the report
            observations_data: Dictionary containing observation counts
            structured_observations: Optional structured observations data (JSONB)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.client:
            raise Exception("Supabase client not available")
        
        try:
            # Prepare update data
            update_data = {
                "total_observations": observations_data.get('total_observations', 0),
                "low": observations_data.get('low', 0),
                "medium": observations_data.get('medium', 0),
                "high": observations_data.get('high', 0)
            }
            
            # Add structured observations if provided and column exists
            if structured_observations is not None and self.has_observations_column():
                update_data["observations"] = structured_observations
            elif structured_observations is not None:
                # Store observations in description column as JSON as a workaround
                print("Warning: Observations column not available, storing in description column")
                update_data["description"] = json.dumps(structured_observations, indent=2)
            
            # Update the report
            response = self.client.table('reports').update(update_data).eq('report_id', report_id).execute()
            
            if response.data:
                print(f"Report observations updated successfully: {report_id}")
                return True
            else:
                print(f"No report found with ID: {report_id}")
                return False
                
        except Exception as e:
            print(f"Error updating report observations: {e}")
            return False
    
    def get_report(self, report_id: str) -> Optional[Dict]:
        """
        Retrieve a report from the database
        
        Args:
            report_id: Unique identifier for the report
            
        Returns:
            Dictionary containing report data or None if not found
        """
        if not self.client:
            raise Exception("Supabase client not available")
        
        try:
            response = self.client.table('reports').select('*').eq('report_id', report_id).execute()
            
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
            
        except Exception as e:
            print(f"Error retrieving report: {e}")
            return None
    
    def create_formal_report(self, report_id: str, video_duration: float, 
                           video_captured_at: datetime, video_device_type: str, 
                           total_observations: int, low: int, medium: int, high: int,
                           formal_report_content: str) -> str:
        """
        Create a formal report entry in the formal_reports table
        
        Args:
            report_id: Original report ID (will be prefixed with 'formal_')
            video_duration: Duration of the video in seconds
            video_captured_at: Timestamp when video was captured
            video_device_type: Type of device used to capture video
            total_observations: Total number of observations
            low: Number of low severity observations
            medium: Number of medium severity observations
            high: Number of high severity observations
            formal_report_content: Generated formal safety report content
            
        Returns:
            formal_report_id: Unique identifier for the formal report
        """
        if not self.client:
            raise Exception("Supabase client not available")
        
        try:
            # Generate unique formal report ID
            formal_report_id = f"formal_{report_id}"
            
            # Prepare data for insertion
            formal_report_data = {
                "report_id": formal_report_id,
                "video_duration": video_duration,
                "video_captured_at": video_captured_at.isoformat(),
                "video_device_type": video_device_type,
                "total_observations": total_observations,
                "low": low,
                "medium": medium,
                "high": high,
                "description": formal_report_content  # Store RAG content in description column
            }
            
            # Insert into Supabase formal_reports table
            response = self.client.table('formal_reports').insert(formal_report_data).execute()
            
            if response.data:
                print(f"Formal report created successfully: {formal_report_id}")
                return formal_report_id
            else:
                raise Exception("Failed to create formal report - no data returned")
                
        except Exception as e:
            print(f"Error creating formal report: {e}")
            raise

    def get_formal_report(self, formal_report_id: str) -> Optional[Dict]:
        """
        Retrieve a formal report from the database
        
        Args:
            formal_report_id: Unique identifier for the formal report
            
        Returns:
            Dictionary containing formal report data or None if not found
        """
        if not self.client:
            raise Exception("Supabase client not available")
        
        try:
            response = self.client.table('formal_reports').select('*').eq('report_id', formal_report_id).execute()
            
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
            
        except Exception as e:
            print(f"Error retrieving formal report: {e}")
            return None

    def list_reports(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """
        List reports from the database
        
        Args:
            limit: Maximum number of reports to return
            offset: Number of reports to skip
            
        Returns:
            List of report dictionaries
        """
        if not self.client:
            raise Exception("Supabase client not available")
        
        try:
            response = (self.client.table('reports')
                       .select('*')
                       .order('video_captured_at', desc=True)
                       .range(offset, offset + limit - 1)
                       .execute())
            
            return response.data if response.data else []
            
        except Exception as e:
            print(f"Error listing reports: {e}")
            return []

def parse_anomalai_structured_data(structured_analysis: str) -> Dict:
    """
    Parse AnomalAI structured analysis to extract full structured data
    
    Args:
        structured_analysis: The structured analysis text from AnomalAI
        
    Returns:
        Dictionary with full structured observations data
    """
    try:
        if not structured_analysis:
            return {
                'description': '',
                'summary': {'low': 0, 'medium': 0, 'high': 0},
                'observations': []
            }
        
        # Try to find JSON in the structured analysis
        lines = structured_analysis.split('\n')
        json_start = -1
        json_end = -1
        
        for i, line in enumerate(lines):
            if line.strip().startswith('{'):
                json_start = i
                break
        
        if json_start >= 0:
            # Find the end of JSON
            brace_count = 0
            for i in range(json_start, len(lines)):
                line = lines[i]
                for char in line:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i
                            break
                if json_end >= 0:
                    break
            
            if json_end >= 0:
                # Extract and parse JSON
                json_text = '\n'.join(lines[json_start:json_end + 1])
                data = json.loads(json_text)
                
                # Return the full structured data
                return {
                    'description': data.get('description', ''),
                    'summary': data.get('summary', {'low': 0, 'medium': 0, 'high': 0}),
                    'observations': data.get('observations', [])
                }
        
        return {
            'description': '',
            'summary': {'low': 0, 'medium': 0, 'high': 0},
            'observations': []
        }
        
    except Exception as e:
        print(f"Error parsing AnomalAI structured data: {e}")
        return {
            'description': '',
            'summary': {'low': 0, 'medium': 0, 'high': 0},
            'observations': []
        }

def parse_anomalai_observations(structured_analysis: str) -> Dict:
    """
    Parse AnomalAI structured analysis to extract observation counts
    
    Args:
        structured_analysis: The structured analysis text from AnomalAI
        
    Returns:
        Dictionary with observation counts by severity
    """
    try:
        # Initialize counts
        observations = {
            'total_observations': 0,
            'low': 0,
            'medium': 0,
            'high': 0
        }
        
        if not structured_analysis:
            return observations
        
        # Try to find JSON in the structured analysis
        lines = structured_analysis.split('\n')
        json_start = -1
        json_end = -1
        
        for i, line in enumerate(lines):
            if line.strip().startswith('{'):
                json_start = i
                break
        
        if json_start >= 0:
            # Find the end of JSON
            brace_count = 0
            for i in range(json_start, len(lines)):
                line = lines[i]
                for char in line:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i
                            break
                if json_end >= 0:
                    break
            
            if json_end >= 0:
                # Extract and parse JSON
                json_text = '\n'.join(lines[json_start:json_end + 1])
                data = json.loads(json_text)
                
                # Count observations by severity
                if 'observations' in data:
                    observations['total_observations'] = len(data['observations'])
                    for obs in data['observations']:
                        severity = obs.get('severity', 'low').lower()
                        if severity in observations:
                            observations[severity] += 1
                
                # Also check summary if available
                if 'summary' in data:
                    summary = data['summary']
                    observations['low'] = summary.get('low', observations['low'])
                    observations['medium'] = summary.get('medium', observations['medium'])
                    observations['high'] = summary.get('high', observations['high'])
                    observations['total_observations'] = observations['low'] + observations['medium'] + observations['high']
        
        return observations
        
    except Exception as e:
        print(f"Error parsing AnomalAI observations: {e}")
        return {
            'total_observations': 0,
            'low': 0,
            'medium': 0,
            'high': 0
        }

# Global Supabase database manager instance
supabase_db_manager = SupabaseDatabaseManager()
