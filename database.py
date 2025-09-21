"""
Database module for storing video analysis reports
"""

import psycopg2
import psycopg2.extras
import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

class DatabaseManager:
    def __init__(self):
        """Initialize database connection"""
        self.connection = None
        self.connect()
    
    def connect(self):
        """Establish database connection"""
        try:
            # Get database connection parameters from environment variables
            db_config = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': os.getenv('DB_PORT', '5432'),
                'database': os.getenv('DB_NAME', 'safety_analysis'),
                'user': os.getenv('DB_USER', 'postgres'),
                'password': os.getenv('DB_PASSWORD', 'password')
            }
            
            self.connection = psycopg2.connect(**db_config)
            print("Database connection established successfully")
        except psycopg2.OperationalError as e:
            print(f"Database connection failed: {e}")
            print("Please ensure PostgreSQL is running and check your environment variables:")
            print("  DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD")
            self.connection = None
        except Exception as e:
            print(f"Unexpected error connecting to database: {e}")
            self.connection = None
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            print("Database connection closed")
    
    def is_available(self) -> bool:
        """Check if database connection is available"""
        return self.connection is not None
    
    def reconnect(self):
        """Attempt to reconnect to the database"""
        self.close()
        self.connect()
    
    def create_report(self, video_id: str, video_duration: float, 
                     video_captured_at: datetime, video_device_type: str = "smart glasses") -> str:
        """
        Create a new report entry in the database
        
        Args:
            video_id: Unique identifier for the video
            video_duration: Duration of the video in seconds
            video_captured_at: Timestamp when video was captured
            video_device_type: Type of device used to capture video
            
        Returns:
            report_id: Unique identifier for the report
        """
        if not self.connection:
            raise Exception("Database connection not available")
        
        try:
            # Generate unique report ID
            report_id = f"report_{video_id}_{int(datetime.now().timestamp())}"
            
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO reports (report_id, video_duration, video_captured_at, video_device_type)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (report_id) DO UPDATE SET
                        video_duration = EXCLUDED.video_duration,
                        video_captured_at = EXCLUDED.video_captured_at,
                        video_device_type = EXCLUDED.video_device_type
                    RETURNING report_id
                """, (report_id, video_duration, video_captured_at, video_device_type))
                
                result = cursor.fetchone()
                self.connection.commit()
                
                if result:
                    print(f"Report created successfully: {result[0]}")
                    return result[0]
                else:
                    raise Exception("Failed to create report")
                    
        except Exception as e:
            print(f"Error creating report: {e}")
            self.connection.rollback()
            raise
    
    def update_report_observations(self, report_id: str, observations_data: Dict) -> bool:
        """
        Update report with observation counts from AnomalAI analysis
        
        Args:
            report_id: Unique identifier for the report
            observations_data: Dictionary containing observation counts
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.connection:
            raise Exception("Database connection not available")
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    UPDATE reports 
                    SET total_observations = %s,
                        low = %s,
                        medium = %s,
                        high = %s
                    WHERE report_id = %s
                """, (
                    observations_data.get('total_observations', 0),
                    observations_data.get('low', 0),
                    observations_data.get('medium', 0),
                    observations_data.get('high', 0),
                    report_id
                ))
                
                if cursor.rowcount > 0:
                    self.connection.commit()
                    print(f"Report observations updated successfully: {report_id}")
                    return True
                else:
                    print(f"No report found with ID: {report_id}")
                    return False
                    
        except Exception as e:
            print(f"Error updating report observations: {e}")
            self.connection.rollback()
            return False
    
    def get_report(self, report_id: str) -> Optional[Dict]:
        """
        Retrieve a report from the database
        
        Args:
            report_id: Unique identifier for the report
            
        Returns:
            Dictionary containing report data or None if not found
        """
        if not self.connection:
            raise Exception("Database connection not available")
        
        try:
            with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM reports WHERE report_id = %s
                """, (report_id,))
                
                result = cursor.fetchone()
                if result:
                    return dict(result)
                return None
                
        except Exception as e:
            print(f"Error retrieving report: {e}")
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
        if not self.connection:
            raise Exception("Database connection not available")
        
        try:
            with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM reports 
                    ORDER BY video_captured_at DESC 
                    LIMIT %s OFFSET %s
                """, (limit, offset))
                
                results = cursor.fetchall()
                return [dict(row) for row in results]
                
        except Exception as e:
            print(f"Error listing reports: {e}")
            return []

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

# Global database manager instance
db_manager = DatabaseManager()
