-- Database setup script for safety analysis reports
-- Run this script in your PostgreSQL database

-- Create the reports table
CREATE TABLE IF NOT EXISTS public.reports (
    id bigserial NOT NULL,
    report_id text NOT NULL,
    video_duration double precision NOT NULL,
    video_captured_at timestamp with time zone NOT NULL,
    video_device_type text NOT NULL,
    created_at timestamp with time zone NOT NULL DEFAULT now(),
    total_observations integer NOT NULL DEFAULT 0,
    low integer NOT NULL DEFAULT 0,
    medium integer NOT NULL DEFAULT 0,
    high integer NOT NULL DEFAULT 0,
    observations JSONB,
    CONSTRAINT reports_pkey PRIMARY KEY (id),
    CONSTRAINT reports_report_id_key UNIQUE (report_id)
) TABLESPACE pg_default;

-- Create index for efficient querying by capture time
CREATE INDEX IF NOT EXISTS reports_captured_at_idx 
ON public.reports USING btree (video_captured_at DESC) 
TABLESPACE pg_default;

-- Create index for efficient querying by device type
CREATE INDEX IF NOT EXISTS reports_device_type_idx 
ON public.reports USING btree (video_device_type) 
TABLESPACE pg_default;

-- Create index for efficient querying by total observations
CREATE INDEX IF NOT EXISTS reports_observations_idx 
ON public.reports USING btree (total_observations DESC) 
TABLESPACE pg_default;

-- Create the formal_reports table (same structure as reports)
CREATE TABLE IF NOT EXISTS public.formal_reports (
    id bigserial NOT NULL,
    report_id text NOT NULL,
    video_duration double precision NOT NULL,
    video_captured_at timestamp with time zone NOT NULL,
    video_device_type text NOT NULL,
    created_at timestamp with time zone NOT NULL DEFAULT now(),
    total_observations integer NOT NULL DEFAULT 0,
    low integer NOT NULL DEFAULT 0,
    medium integer NOT NULL DEFAULT 0,
    high integer NOT NULL DEFAULT 0,
    observations JSONB,
    CONSTRAINT formal_reports_pkey PRIMARY KEY (id),
    CONSTRAINT formal_reports_report_id_key UNIQUE (report_id)
) TABLESPACE pg_default;

-- Create indexes for formal_reports table
CREATE INDEX IF NOT EXISTS formal_reports_captured_at_idx 
ON public.formal_reports USING btree (video_captured_at DESC) 
TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS formal_reports_device_type_idx 
ON public.formal_reports USING btree (video_device_type) 
TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS formal_reports_observations_idx 
ON public.formal_reports USING btree (total_observations DESC) 
TABLESPACE pg_default;

-- Grant permissions (adjust as needed for your setup)
-- GRANT ALL PRIVILEGES ON TABLE public.reports TO your_app_user;
-- GRANT USAGE, SELECT ON SEQUENCE public.reports_id_seq TO your_app_user;
-- GRANT ALL PRIVILEGES ON TABLE public.formal_reports TO your_app_user;
-- GRANT USAGE, SELECT ON SEQUENCE public.formal_reports_id_seq TO your_app_user;
