import os
import re
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import AzureOpenAI
import logging
from logging.handlers import RotatingFileHandler
from sqlalchemy import create_engine, text
from openai import AzureOpenAI


load_dotenv()
# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
def setup_logging():
    """Configure logging for the application"""
    logger = logging.getLogger('improved_manufacturing_chatbot')
    logger.setLevel(logging.INFO)
   
    # Clear existing handlers to prevent duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
   
    # Console handler
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        logger.addHandler(console_handler)
   
    # File handler with rotation
    try:
        file_handler = RotatingFileHandler(
            'improved_chatbot.log',
            maxBytes=10485760, # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(console_formatter)
        if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
            logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Could not create file handler: {e}")
   
    logger.propagate = False
    return logger
# Initialize logger
logger = setup_logging()
# =============================================================================
# CONFIGURATION & INITIALIZATION
# =============================================================================
class Config:
    """Configuration management for the chatbot"""
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
   
    # SQL safety settings - UPDATED with new schema
    MAX_RESULTS = 1000
    ALLOWED_TABLES = [
        "sites", "departments", "production_lines", "erp_orders",
        "iso55001_metrics", "kpi_metrics", "maintenance_records",
        "process_variables", "quality_inspections", "s88_batch_control",
        "dashboard_status"
        # We explicitly exclude the _history tables
    ]
    DANGEROUS_KEYWORDS = ["DROP", "DELETE", "TRUNCATE", "ALTER", "CREATE", "INSERT", "UPDATE"]
# Initialize clients
@st.cache_resource
def init_clients():
    """Initialize Supabase and Azure OpenAI clients"""
    if not all([Config.SUPABASE_URL, Config.SUPABASE_KEY]):
        st.error("Missing Supabase credentials. Check your .env file.")
        st.stop()
   
    supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
   
    azure_client = None
    if all([Config.AZURE_OPENAI_API_KEY, Config.AZURE_OPENAI_ENDPOINT]):
        azure_client = AzureOpenAI(
            api_key=Config.AZURE_OPENAI_API_KEY,
            api_version=Config.AZURE_OPENAI_API_VERSION,
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT
        )
   
    return supabase, azure_client
# =============================================================================
# SQL SAFETY & VALIDATION
# =============================================================================
class SQLValidator:
    """Validates and sanitizes SQL queries for safety"""
   
    @staticmethod
    def is_safe_query(sql: str) -> Tuple[bool, str]:
        """Check if SQL query is safe to execute"""
        sql_upper = sql.upper()
       
        # Check for dangerous keywords
        for keyword in Config.DANGEROUS_KEYWORDS:
            if keyword in sql_upper:
                return False, f"Dangerous operation '{keyword}' not allowed"
       
        # Must be a SELECT query
        if not sql_upper.strip().startswith("SELECT"):
            return False, "Only SELECT queries are allowed"
       
        # Check for allowed tables
        # This regex finds tables after FROM or JOIN clauses
        table_pattern = r'\b(?:FROM|JOIN)\s+([a-zA-Z0-9_]+)'
        tables = re.findall(table_pattern, sql_upper)
       
        for table in tables:
            if table.lower() not in Config.ALLOWED_TABLES:
                # Allow tables aliased in WITH clauses
                with_clause_pattern = r'\bWITH\s+(\w+)\s+AS\b'
                with_tables = re.findall(with_clause_pattern, sql_upper)
                if table.lower() not in [t.lower() for t in with_tables]:
                    return False, f"Access to table '{table}' not allowed"
       
        # Check for SQL injection patterns (but allow legitimate SQL comments)
        injection_patterns = [
            r";\s*DROP",
            r";\s*DELETE",
            r"/\*.*\*/",
            r"UNION.*SELECT",
            r"EXEC(\s|\()",
            r"EXECUTE(\s|\()"
        ]
        for pattern in injection_patterns:
            if re.search(pattern, sql_upper):
                return False, "Potential SQL injection detected"
       
        # Check for dangerous comment patterns (but allow legitimate ones)
        if re.search(r'--;.*(?:DROP|DELETE|INSERT|UPDATE|ALTER)', sql_upper):
            return False, "Dangerous comment pattern detected"
       
        return True, ""
# =============================================================================
# SIMPLE SQL GENERATOR (Updated for new schema)
# =============================================================================
class SimpleSQLGenerator:
    """Generates clean, simple SQL queries without unnecessary complexity"""
   
    @staticmethod
    def generate_factory_overview_query() -> Dict[str, Any]:
        """Generate factory overview query that works with NEW database structure"""
       
        # Use a simple query that will work even if some tables are empty
        # Start with sites and build from there
        # SCHEMA: sites -> departments -> production_lines -> data tables
        sql = """
        SELECT
            COALESCE(s.site_name, 'Sample Site') AS factory_location,
            COALESCE(s.bu, 'Manufacturing') AS division,
            COALESCE(s.gm, 'Not Assigned') AS general_manager,
            COALESCE(a.department_name, 'General') AS area,
            COALESCE(pl.line_name, 'Line1') AS line_name,
            -- OEE Metrics converted to percentages
            ROUND((COALESCE(mk.availability, 0) * 100)::numeric, 0) AS availability,
            ROUND((COALESCE(mk.quality, 0) * 100)::numeric, 0) AS quality,
            ROUND((COALESCE(mk.performance, 0) * 100)::numeric, 0) AS performance,
            ROUND((COALESCE(mk.oee, 0) * 100)::numeric, 0) AS oee,
            -- Order Information
            eo.order_number,
            eo.item_number,
            eo.scheduled_end_time,
            COALESCE(eo.produced_quantity, 0) AS produced_quantity,
            COALESCE(eo.remaining_quantity, 0) AS remaining_quantity,
            eo.order_status,
            -- Quality Control Information
            qc.inspection_result,
            qc.rejection_reason,
            COALESCE(qc.rejection_quantity, 0) AS rejection_quantity,
            COALESCE(qc.accepted_quantity, 0) AS accepted_quantity,
            -- Maintenance Information
            mr.maintenance_status,
            mr.last_maintenance_date,
            mr.next_maintenance_date,
            -- Timestamps
            mk.timestamp AS metrics_timestamp,
            eo.timestamp AS order_timestamp,
            qc.timestamp AS quality_timestamp
        FROM sites s
        LEFT JOIN departments a ON s.id = a.site_id
        LEFT JOIN production_lines pl ON a.id = pl.department_id
        LEFT JOIN (
            SELECT DISTINCT ON (line_id) line_id, availability, quality, performance, oee, timestamp
            FROM kpi_metrics
            ORDER BY line_id, timestamp DESC
        ) mk ON mk.line_id = pl.id
        LEFT JOIN (
            SELECT DISTINCT ON (line_id) line_id, order_number, item_number, scheduled_end_time,
                   produced_quantity, remaining_quantity, order_status, timestamp
            FROM erp_orders
            ORDER BY line_id, scheduled_end_time DESC
        ) eo ON eo.line_id = pl.id
        LEFT JOIN (
            SELECT DISTINCT ON (line_id) line_id, inspection_result, rejection_reason,
                   rejection_quantity, accepted_quantity, timestamp
            FROM quality_inspections
            ORDER BY line_id, timestamp DESC
        ) qc ON qc.line_id = pl.id
        LEFT JOIN (
            SELECT DISTINCT ON (line_id) line_id, maintenance_status, last_maintenance_date,
                   next_maintenance_date
            FROM maintenance_records
            ORDER BY line_id, timestamp DESC
        ) mr ON mr.line_id = pl.id
        WHERE s.site_name IN ('Biyagama', 'Katunayake')
        ORDER BY
            s.site_name ASC,
            CASE
                WHEN COALESCE(a.department_name, 'General') ILIKE '%press%' THEN 1
                WHEN COALESCE(a.department_name, 'General') ILIKE '%heat%' THEN 2
                WHEN COALESCE(a.department_name, 'General') ILIKE '%assembly%' THEN 3
                ELSE 4
            END,
            COALESCE(pl.line_name, 'Line1') ASC
        LIMIT 50;
        """
       
        return {
            'sql': sql.strip(),
            'params': [],
            'explanation': 'Factory overview with comprehensive operational data from database',
            'is_factory_overview': True
        }
   
    @staticmethod
    def generate_simple_factory_overview_query() -> Dict[str, Any]:
        """Generate real database factory overview query - no hardcoded values"""
       
        # Use the real comprehensive query instead of hardcoded data
        return SimpleSQLGenerator.generate_factory_overview_query()
   
    @staticmethod
    def clean_value(value: str) -> str:
        """Clean and normalize string values"""
        if not value:
            return ""
        # Remove possessives and normalize
        value = value.replace("'s", "").replace("'", "")
        # Keep alphanumeric characters, spaces, and hyphens
        value = re.sub(r'[^a-zA-Z0-9\s-]', '', value)
        return value.strip()
   
    @staticmethod
    def generate_oee_query(site: str = None, area: str = None, line: str = None,
                          time_filter: str = None, limit: int = 1) -> Dict[str, Any]:
        """Generate clean OEE query using correct NEW schema with proper JOINs"""
       
        where_conditions = []
       
        # Build the proper JOIN chain according to the NEW schema hierarchy
        # kpi_metrics -> production_lines -> departments -> sites
        joins = """FROM kpi_metrics k
        JOIN production_lines pl ON k.line_id = pl.id
        JOIN departments a ON pl.department_id = a.id
        JOIN sites s ON a.site_id = s.id"""
       
        # Apply filters using ILIKE for flexible string matching
        if site:
            clean_site = SimpleSQLGenerator.clean_value(site)
            where_conditions.append(f"LOWER(s.site_name) ILIKE '%{clean_site.lower()}%'")
           
        if area:
            clean_area = SimpleSQLGenerator.clean_value(area)
            where_conditions.append(f"LOWER(a.department_name) ILIKE '%{clean_area.lower()}%'")
           
        if line:
            clean_line = SimpleSQLGenerator.clean_value(line)
            # Handle both "Line1" and "Line 1" formats with flexible matching
            line_num = re.search(r'(\d+)', clean_line)
            if line_num:
                number = line_num.group(1)
                # Use ILIKE to match both "Line1" and "Line 1" formats
                where_conditions.append(f"(pl.line_name ILIKE 'Line{number}' OR pl.line_name ILIKE 'Line {number}' OR pl.line_name ILIKE '%Line%{number}%')")
            else:
                where_conditions.append(f"pl.line_name ILIKE '%{clean_line}%'")
       
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
       
        time_condition = ""
        if time_filter:
            time_condition = f" AND k.timestamp >= NOW() - INTERVAL '{time_filter}'"
       
        sql = f"""
        SELECT
            s.site_name,
            a.department_name AS area_name,
            pl.line_name,
            k.oee,
            k.availability,
            k.performance,
            k.quality,
            k.timestamp
        {joins}
        WHERE {where_clause}{time_condition}
        ORDER BY k.timestamp DESC
        LIMIT {limit}
        """
       
        return {
            'sql': sql.strip(),
            'params': [],
            'explanation': f'OEE metrics for {line or "all lines"} in {area or "all areas"} at {site or "all sites"}'
        }
   
    @staticmethod
    def generate_maintenance_query(site: str = None, machine: str = None,
                                overdue_only: bool = False) -> Dict[str, Any]:
        """Generate maintenance query using correct NEW schema hierarchy"""
   
        where_conditions = []
       
        # For "which machines need maintenance" - look for incomplete maintenance
        # Default behavior: show machines that need maintenance
        where_conditions.append("(mr.maintenance_status IS NULL OR (LOWER(mr.maintenance_status) NOT ILIKE '%completed%' AND LOWER(mr.maintenance_status) NOT ILIKE '%done%'))")
       
        # If specifically asking for overdue, add date condition
        if overdue_only:
            where_conditions.append("mr.next_maintenance_date < NOW()")
   
        # Add site filter using ILIKE for flexible matching
        if site:
            clean_site = SimpleSQLGenerator.clean_value(site)
            where_conditions.append(f"LOWER(s.site_name) ILIKE '%{clean_site.lower()}%'")
       
        # Add machine filter - use flexible ILIKE matching for machine_id
        if machine:
            clean_machine = SimpleSQLGenerator.clean_value(machine)
            # Use LOWER() and ILIKE for case-insensitive flexible matching
            where_conditions.append(f"LOWER(mr.machine_id) ILIKE '%{clean_machine.lower()}%'")
   
        # Build WHERE clause
        where_clause = " AND ".join(where_conditions)
   
        # NEW JOIN Hierarchy
        sql = f"""
        SELECT
            s.site_name,
            a.department_name AS area_name,
            pl.line_name,
            mr.machine_id,
            mr.maintenance_status,
            mr.last_maintenance_date,
            mr.next_maintenance_date,
            mr.timestamp
        FROM maintenance_records mr
        JOIN production_lines pl ON mr.line_id = pl.id
        JOIN departments a ON pl.department_id = a.id
        JOIN sites s ON a.site_id = s.id
        WHERE {where_clause}
        ORDER BY mr.timestamp DESC
        LIMIT 100
        """
   
        return {
            'sql': sql.strip(),
            'params': [],
            'explanation': f"Machines needing maintenance for {machine or 'all machines'} in {site or 'all sites'}"
        }
   
    @staticmethod
    def generate_mtbf_query(machine: str = None, site: str = None) -> Dict[str, Any]:
        """Generate MTBF query with proper joins to kpi_metrics table"""
       
        where_conditions = []
       
        # Add machine filter with flexible ILIKE matching
        if machine:
            clean_machine = SimpleSQLGenerator.clean_value(machine)
            # Use LOWER() and ILIKE for case-insensitive flexible matching
            where_conditions.append(f"LOWER(mr.machine_id) ILIKE '%{clean_machine.lower()}%'")
       
        # Add site filter using ILIKE for flexible matching
        if site:
            clean_site = SimpleSQLGenerator.clean_value(site)
            where_conditions.append(f"LOWER(s.site_name) ILIKE '%{clean_site.lower()}%'")
       
        # Build WHERE clause
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
       
        # NEW JOIN Hierarchy
        sql = f"""
        SELECT
            s.site_name,
            a.department_name AS area_name,
            pl.line_name,
            mr.machine_id,
            k.mtbf,
            k.timestamp
        FROM maintenance_records mr
        JOIN production_lines pl ON mr.line_id = pl.id
        JOIN departments a ON pl.department_id = a.id
        JOIN sites s ON a.site_id = s.id
        JOIN kpi_metrics k ON pl.id = k.line_id
        WHERE {where_clause}
        ORDER BY k.timestamp DESC
        LIMIT 1
        """
       
        return {
            'sql': sql.strip(),
            'params': [],
            'explanation': f"MTBF (Mean Time Between Failures) data for {machine or 'all machines'} in {site or 'all sites'}"
        }
   
    @staticmethod
    def generate_mttr_query(machine: str = None, site: str = None) -> Dict[str, Any]:
        """Generate MTTR query with proper joins to kpi_metrics table"""
       
        where_conditions = []
       
        # Add machine filter with flexible ILIKE matching
        if machine:
            clean_machine = SimpleSQLGenerator.clean_value(machine)
            # Use LOWER() and ILIKE for case-insensitive flexible matching
            where_conditions.append(f"LOWER(mr.machine_id) ILIKE '%{clean_machine.lower()}%'")
       
        # Add site filter using ILIKE for flexible matching
        if site:
            clean_site = SimpleSQLGenerator.clean_value(site)
            where_conditions.append(f"LOWER(s.site_name) ILIKE '%{clean_site.lower()}%'")
       
        # Build WHERE clause
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
       
        # NEW JOIN Hierarchy
        sql = f"""
        SELECT
            s.site_name,
            a.department_name AS area_name,
            pl.line_name,
            mr.machine_id,
            k.mttr,
            k.timestamp
        FROM maintenance_records mr
        JOIN production_lines pl ON mr.line_id = pl.id
        JOIN departments a ON pl.department_id = a.id
        JOIN sites s ON a.site_id = s.id
        JOIN kpi_metrics k ON pl.id = k.line_id
        WHERE {where_clause}
        ORDER BY k.timestamp DESC
        LIMIT 1
        """
       
        return {
            'sql': sql.strip(),
            'params': [],
            'explanation': f"MTTR (Mean Time To Repair) data for {machine or 'all machines'} in {site or 'all sites'}"
        }
   
    @staticmethod
    def generate_overdue_maintenance_query(site: str = None, machine: str = None) -> Dict[str, Any]:
        """Generate overdue maintenance query using ILIKE pattern matching"""
       
        where_conditions = []
       
        # Filter for overdue maintenance status using ILIKE
        where_conditions.append("LOWER(mr.maintenance_status) ILIKE '%overdue%'")
       
        # Add site filter using proper schema hierarchy
        if site:
            clean_site = SimpleSQLGenerator.clean_value(site)
            where_conditions.append(f"LOWER(s.site_name) ILIKE '%{clean_site.lower()}%'")
       
        # Add machine filter with flexible ILIKE matching
        if machine:
            clean_machine = SimpleSQLGenerator.clean_value(machine)
            where_conditions.append(f"LOWER(mr.machine_id) ILIKE '%{clean_machine.lower()}%'")
       
        # Build WHERE clause
        where_clause = " AND ".join(where_conditions)
       
        # NEW JOIN Hierarchy
        sql = f"""
        SELECT
            s.site_name,
            a.department_name AS area_name,
            pl.line_name,
            mr.machine_id,
            mr.maintenance_status,
            mr.last_maintenance_date,
            mr.next_maintenance_date,
            mr.timestamp
        FROM maintenance_records mr
        JOIN production_lines pl ON mr.line_id = pl.id
        JOIN departments a ON pl.department_id = a.id
        JOIN sites s ON a.site_id = s.id
        WHERE {where_clause}
        ORDER BY mr.timestamp DESC
        LIMIT 100
        """
       
        return {
            'sql': sql.strip(),
            'params': [],
            'explanation': f"Machines with overdue maintenance status for {machine or 'all machines'} in {site or 'all sites'}"
        }
       
   
    @staticmethod
    def generate_batch_control_query(recipe_filter: str = None,
                                       time_filter: str = "7 days") -> Dict[str, Any]:
        """Generate S88 batch control query for soda recipes and production parameters"""
       
        where_conditions = []
        where_conditions.append(f"bc.timestamp >= NOW() - INTERVAL '{time_filter}'")
       
        if recipe_filter:
            clean_recipe = SimpleSQLGenerator.clean_value(recipe_filter)
            where_conditions.append(f"bc.soda_recipe ILIKE '%{clean_recipe}%'")
       
        where_clause = " AND ".join(where_conditions)
       
        # NEW JOIN Hierarchy
        sql = f"""
        SELECT
            s.site_name,
            a.department_name AS area_name,
            pl.line_name,
            bc.soda_recipe,
            bc.production_parameters,
            bc.batch_mixing_tank_status,
            bc.bottler_status,
            bc.capper_status,
            bc.temperature_controller,
            bc.volume_control,
            bc.operator_interface_status,
            bc.quality_data,
            bc.safety_status,
            bc.timestamp
        FROM s88_batch_control bc
        JOIN production_lines pl ON bc.line_id = pl.id
        JOIN departments a ON pl.department_id = a.id
        JOIN sites s ON a.site_id = s.id
        WHERE {where_clause}
        ORDER BY bc.timestamp DESC
        LIMIT 20
        """
       
        return {
            'sql': sql.strip(),
            'params': [],
            'explanation': f'Batch control data for {recipe_filter or "all recipes"} in last {time_filter}'
        }
   
    @staticmethod
    def generate_simple_batch_control_query(recipe_filter: str = None, limit: int = 20) -> Dict[str, Any]:
        """Generate simple batch control query for soda recipes and production parameters"""
       
        where_conditions = []
       
        if recipe_filter:
            clean_recipe = SimpleSQLGenerator.clean_value(recipe_filter)
            where_conditions.append(f"LOWER(soda_recipe) ILIKE '%{clean_recipe.lower()}%'")
       
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
       
        sql = f"""
        SELECT
            line_id,
            soda_recipe,
            production_parameters,
            timestamp
        FROM s88_batch_control
        WHERE {where_clause}
        ORDER BY timestamp DESC
        LIMIT {limit}
        """
       
        return {
            'sql': sql.strip(),
            'params': [],
            'explanation': f'Simple batch control data showing soda recipes and production parameters for {recipe_filter or "all recipes"}'
        }
   
    @staticmethod
    def generate_item_rejection_query(time_filter: str = "30 days") -> Dict[str, Any]:
        """Generate query for item numbers with highest rejection quantities"""
       
        # NEW JOIN Hierarchy
        sql = f"""
        SELECT
            qc.item_number,
            s.site_name,
            a.department_name AS area_name,
            pl.line_name,
            qc.rejection_reason,
            SUM(qc.rejection_quantity) as total_rejection_quantity,
            SUM(qc.accepted_quantity) as total_accepted_quantity,
            COUNT(*) as rejection_incidents
        FROM quality_inspections qc
        JOIN production_lines pl ON qc.line_id = pl.id
        JOIN departments a ON pl.department_id = a.id
        JOIN sites s ON a.site_id = s.id
        WHERE qc.timestamp >= NOW() - INTERVAL '{time_filter}'
        GROUP BY qc.item_number, s.site_name, a.department_name, pl.line_name, qc.rejection_reason
        ORDER BY total_rejection_quantity DESC
        LIMIT 20
        """
       
        return {
            'sql': sql.strip(),
            'params': [],
            'explanation': f'Items with highest rejection quantities in last {time_filter}'
        }
   
    @staticmethod
    def generate_quality_query(site: str = None, area: str = None, time_filter: str = "24 hours") -> Dict[str, Any]:
        """Generate quality control query using correct NEW schema hierarchy"""
       
        where_conditions = []
        where_conditions.append(f"qc.timestamp >= NOW() - INTERVAL '{time_filter}'")
       
        # Add site filter using ILIKE for flexible matching
        if site:
            clean_site = SimpleSQLGenerator.clean_value(site)
            where_conditions.append(f"LOWER(s.site_name) ILIKE '%{clean_site.lower()}%'")
       
        # Add area filter using ILIKE for flexible matching
        if area:
            clean_area = SimpleSQLGenerator.clean_value(area)
            where_conditions.append(f"LOWER(a.department_name) ILIKE '%{clean_area.lower()}%'")
       
        where_clause = " AND ".join(where_conditions)
       
        # NEW JOIN Hierarchy
        sql = f"""
        SELECT
            s.site_name,
            a.department_name AS area_name,
            pl.line_name,
            qc.item_number,
            qc.order_number,
            qc.inspection_result,
            qc.rejection_reason,
            qc.rejection_quantity,
            qc.accepted_quantity,
            qc.timestamp
        FROM quality_inspections qc
        JOIN production_lines pl ON qc.line_id = pl.id
        JOIN departments a ON pl.department_id = a.id
        JOIN sites s ON a.site_id = s.id
        WHERE {where_clause}
        ORDER BY qc.timestamp DESC
        LIMIT 100
        """
       
        return {
            'sql': sql.strip(),
            'params': [],
            'explanation': f'Quality control data for {area or "all areas"} in {site or "all sites"} for last {time_filter}'
        }
# =============================================================================
# IMPROVED NATURAL LANGUAGE TO SQL TRANSLATOR (Updated for new schema)
# =============================================================================
class ImprovedNLToSQLTranslator:
    """Improved translator that generates clean, simple SQL queries"""
   
    def __init__(self, azure_client: Optional[AzureOpenAI], executor):
        self.azure_client = azure_client
        self.executor = executor
        self.system_prompt = self._build_system_prompt()
   
    def _get_schema_info(self) -> Dict[str, List[str]]:
        """Get schema information from executor"""
        return self.executor.get_schema_info()
   
    def _build_system_prompt(self) -> str:
        """Build system prompt that encourages clean SQL generation (UPDATED)"""
       
        # Get actual schema from database
        schema_info = self._get_schema_info()
        schema_text = "\nDATABASE SCHEMA (Key Tables):\n"
       
        # Highlight the most important tables for manufacturing queries
        # UPDATED key tables
        key_tables = [
            'sites', 'departments', 'production_lines', 'erp_orders',
            'kpi_metrics', 'quality_inspections', 'maintenance_records',
            'dashboard_status'
        ]
       
        for table in key_tables:
            if table in schema_info:
                columns = ', '.join(schema_info[table])
                schema_text += f"- {table}: {columns}\n"
       
        # Build the prompt without using f-string for the template part
        # UPDATED with new hierarchy, tables, and columns
        prompt = """You are a SQL expert for a manufacturing database. Generate SCHEMA-ACCURATE, CLEAN PostgreSQL queries.
CRITICAL: Always match user intent to the CORRECT TABLE based on schema. Wrong table = wrong results!
{}
MANDATORY DATABASE SCHEMA HIERARCHY:
- sites: id (PK), site_name, gm, bu
- departments: id (PK), site_id (FK to sites.id), department_name
- production_lines: id (PK), department_id (FK to departments.id), line_name
- kpi_metrics: id (PK), line_id (FK to production_lines.id), availability, quality, performance, oee, teep, mttr, mtbf, timestamp
- quality_inspections: id (PK), line_id (FK to production_lines.id), order_number, item_number, inspection_result, rejection_reason, rejection_quantity, accepted_quantity, timestamp
- maintenance_records: id (PK), line_id (FK to production_lines.id), machine_id, maintenance_status, last_maintenance_date, next_maintenance_date, maintenance_history, timestamp
- erp_orders: id (PK), line_id (FK to production_lines.id), order_number, order_status, scheduled_start_time, scheduled_end_time, produced_quantity, remaining_quantity, item_number, timestamp
TABLE SELECTION GUIDE:
- "soda recipe", "batch control", "production parameters" → s88_batch_control table
- "OEE", "availability", "quality", "performance" → kpi_metrics table
- "maintenance", "MTBF", "MTTR" → maintenance_records (+ kpi_metrics for metrics)
- "quality control", "rejection", "inspection" → quality_inspections table
- "orders", "production quantity", "scheduled" → erp_orders table
- Simple data requests: SELECT directly from target table
- Context needed: JOIN with hierarchy tables
MANDATORY JOIN RULES:
1. ALWAYS use proper JOINs - never assume columns exist in tables without checking schema
2. To filter by site name: JOIN departments->sites and use sites.site_name
3. To filter by area/department name: JOIN departments and use departments.department_name
4. All data tables (kpi_metrics, quality_inspections, etc.) connect via line_id to production_lines.id
5. Use ORDER BY timestamp DESC LIMIT 1 for "current" data
6. For line names: Handle both "Line1" (no space) and "Line 1" (with space) formats using ILIKE with multiple patterns
CRITICAL RULES FOR CLEAN SQL:
1. Use exact column names from schema - do not invent columns
2. Always JOIN through the hierarchy: sites->departments->production_lines->data_tables
3. For VARCHAR/CHAR fields, use LOWER() + ILIKE '%value%' for flexible matching
4. For exact numeric/date matches, use = operator
5. NO parameterized queries - embed values directly in SQL
6. Always ORDER BY timestamp DESC for time-series data
7. Use reasonable LIMIT values (1 for "current", 10-100 for lists)
FACTORY OVERVIEW QUERIES:
For queries like "Tell me about our factories", "Factory overview", "About our factories":
- Generate the comprehensive factory overview SQL that joins all relevant tables
- MUST include "is_factory_overview": true in the response
- Use proper hierarchy: sites->departments->production_lines with LEFT JOIN to all operational tables
- Include OEE metrics, order data, quality control, and maintenance information
CORRECT EXAMPLE PATTERNS (Updated for new schema):
- "Current OEE for Biyagama Press Line1" →
SELECT
    s.site_name,
    a.department_name AS area_name,
    pl.line_name,
    k.oee,
    k.availability,
    k.performance,
    k.quality,
    k.timestamp
FROM kpi_metrics k
JOIN production_lines pl ON k.line_id = pl.id
JOIN departments a ON pl.department_id = a.id
JOIN sites s ON a.site_id = s.id
WHERE LOWER(s.site_name) ILIKE '%biyagama%' AND LOWER(a.department_name) ILIKE '%press%'
  AND (pl.line_name ILIKE 'Line1' OR pl.line_name ILIKE 'Line 1' OR pl.line_name ILIKE '%Line%1%')
ORDER BY k.timestamp DESC
LIMIT 1;
- "Which machines need maintenance?" →
SELECT
    s.site_name,
    a.department_name AS area_name,
    pl.line_name,
    mr.machine_id,
    mr.maintenance_status,
    mr.last_maintenance_date,
    mr.next_maintenance_date,
    mr.timestamp
FROM maintenance_records mr
JOIN production_lines pl ON mr.line_id = pl.id
JOIN departments a ON pl.department_id = a.id
JOIN sites s ON a.site_id = s.id
WHERE (mr.maintenance_status IS NULL OR LOWER(mr.maintenance_status) NOT IN ('completed', 'done'))
ORDER BY mr.timestamp DESC
LIMIT 100;
- "What is the MTBF for Machine-91?" →
SELECT
    s.site_name,
    a.department_name AS area_name,
    pl.line_name,
    mr.machine_id,
    k.mtbf,
    k.timestamp
FROM maintenance_records mr
JOIN production_lines pl ON mr.line_id = pl.id
JOIN departments a ON pl.department_id = a.id
JOIN sites s ON a.site_id = s.id
JOIN kpi_metrics k ON pl.id = k.line_id
WHERE LOWER(mr.machine_id) ILIKE '%machine-91%'
ORDER BY k.timestamp DESC
LIMIT 1;
- "List all machines with overdue maintenance" →
SELECT
    s.site_name,
    a.department_name AS area_name,
    pl.line_name,
    mr.machine_id,
    mr.maintenance_status,
    mr.last_maintenance_date,
    mr.next_maintenance_date,
    mr.timestamp
FROM maintenance_records mr
JOIN production_lines pl ON mr.line_id = pl.id
JOIN departments a ON pl.department_id = a.id
JOIN sites s ON a.site_id = s.id
WHERE LOWER(mr.maintenance_status) ILIKE '%overdue%'
ORDER BY mr.timestamp DESC
LIMIT 100;
- "Show soda recipe and production parameters for recent batch controls" →
SELECT
    line_id,
    soda_recipe,
    production_parameters,
    timestamp
FROM s88_batch_control
ORDER BY timestamp DESC
LIMIT 20;
SCHEMA-ACCURATE QUERY RULES:
1. Always match user intent to the correct table based on schema
2. Batch control data (soda recipes, production parameters) comes from s88_batch_control table
3. Manufacturing KPIs (OEE, availability, quality) come from kpi_metrics table
4. Maintenance data comes from maintenance_records table
5. Quality control data comes from quality_inspections table
6. Production orders come from erp_orders table
7. Use simple SELECT when user asks for basic data, complex JOINs only when context is needed
RESPONSE FORMAT (JSON):
{}
For conceptual questions (definitions, explanations):
{}
Generate clean, executable SQL following the mandatory schema hierarchy. Always use proper JOINs."""
       
        # Format with the schema and JSON examples
        json_example = '''{
  "sql": "Clean SQL query here",
  "params": [],
  "explanation": "Brief explanation",
  "is_conceptual": false
}
For factory overview queries:
{
  "sql": "Comprehensive factory overview SQL with all operational data",
  "params": [],
  "explanation": "Comprehensive factory overview with OEE, orders, and quality data",
  "is_factory_overview": true
}'''
       
        conceptual_example = '''{
  "is_conceptual": true,
  "answer": "Your explanation here",
  "explanation": "This is a conceptual question"
}'''
       
        return prompt.format(schema_text, json_example, conceptual_example)
    def translate(self, question: str, context: Dict = None) -> Dict[str, Any]:
        """Translate natural language to SQL"""
       
        logger.info(f"=== TRANSLATING QUERY ===")
        logger.info(f"User Question: {question}")
        logger.info(f"Context: {context}")
       
        # Check if it's a conceptual question (exclude factory overview queries)
        conceptual_keywords = ['what is', 'define', 'explain', 'meaning of', 'definition',
                              'how does', 'why is']
       
        # Factory overview keywords that should NOT be treated as conceptual
        factory_overview_keywords = ['tell me about our factories', 'tell me about our factory',
                                   'about our factories', 'about our factory', 'factory overview',
                                   'manufacturing overview', 'plant overview']
       
        is_factory_overview_query = any(kw in question.lower() for kw in factory_overview_keywords)
        is_likely_conceptual = any(kw in question.lower() for kw in conceptual_keywords) and not is_factory_overview_query
       
        # Check for factory overview queries first (before LLM)
        factory_overview_keywords = ['tell me about our factories', 'tell me about our factory',
                                   'about our factories', 'about our factory', 'factory overview',
                                   'manufacturing overview', 'plant overview', 'production overview']
       
        if any(kw in question.lower() for kw in factory_overview_keywords):
            logger.info("Factory overview query detected - using dedicated SQL generator")
            return SimpleSQLGenerator.generate_factory_overview_query()
       
        # Try LLM for other queries
        if self.azure_client:
            try:
                result = self._llm_translate(question, context)
               
                if result.get('is_conceptual'):
                    return result
               
                if 'sql' in result:
                    is_safe, error = SQLValidator.is_safe_query(result['sql'])
                    if not is_safe:
                        return {'error': f"Generated SQL is unsafe: {error}"}
               
                return result
               
            except Exception as e:
                logger.warning(f"LLM translation failed: {str(e)}. Using fallback.")
       
        # Handle conceptual questions
        if is_likely_conceptual:
            return self._handle_conceptual_question(question)
       
        # Fallback to simple rule-based parsing
        return self._fallback_translate(question, context)
   
    def _llm_translate(self, question: str, context: Dict) -> Dict[str, Any]:
        """Use LLM to translate"""
        user_message = f"Question: {question}"
       
        if context:
            user_message += f"\n\nContext filters: {json.dumps(context)}"
       
        response = self.azure_client.chat.completions.create(
            model=Config.AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0,
            max_tokens=30000
        )
       
        result_text = response.choices[0].message.content.strip()
       
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())
                logger.info(f"LLM Generated: {result}")
                return result
            except json.JSONDecodeError as json_err:
                logger.error(f"Failed to parse LLM JSON response: {json_err}")
                logger.error(f"Raw text: {result_text}")
                return {'error': f'Failed to parse LLM response: {json_err}'}
        else:
            logger.error(f"No JSON found in LLM response: {result_text}")
            return {'error': 'Failed to parse LLM response'}
   
    def _handle_conceptual_question(self, question: str) -> Dict[str, Any]:
        """Handle conceptual questions"""
        question_lower = question.lower()
       
        concepts = {
            'oee': 'OEE (Overall Equipment Effectiveness) measures manufacturing productivity as the percentage of time that is truly productive. It equals Availability × Performance × Quality. A perfect OEE of 100% means producing only good parts, as fast as possible, with no downtime.',
            'availability': 'Availability measures the percentage of scheduled time that equipment is available to operate. It accounts for downtime events like equipment failures, material shortages, and changeovers.',
            'performance': 'Performance measures how fast equipment runs compared to its designed capacity. It accounts for factors like slow cycles and minor stops that reduce the manufacturing speed.',
            'quality': 'Quality measures the percentage of good units produced versus total units started. It accounts for defective units and units that need rework.',
            'mtbf': 'MTBF (Mean Time Between Failures) is the average time between system breakdowns. Higher MTBF indicates more reliable equipment.',
            'mttr': 'MTTR (Mean Time To Repair) is the average time needed to repair failed equipment. Lower MTTR indicates faster maintenance response.',
        }
       
        for key, definition in concepts.items():
            if key in question_lower:
                return {
                    'is_conceptual': True,
                    'answer': definition,
                    'explanation': f'Conceptual explanation of {key.upper()}'
                }
       
        return {
            'error': 'I can explain manufacturing concepts like OEE, Availability, Performance, Quality, MTBF, and MTTR. Please ask about available data or rephrase your question.'
        }
   
    def _fallback_translate(self, question: str, context: Dict) -> Dict[str, Any]:
        """Simple fallback parser"""
        question_lower = question.lower()
       
        overview_keywords = ['tell me about our factories', 'tell me about our factory', 'tell me about the factories',
                        'tell me about the factory', 'about our factories', 'about our factory',
                        'overview', 'factory summary', 'factory status', 'our factory', 'our factories',
                        'all lines', 'entire factory', 'complete picture', 'factory performance',
                        'all operations', 'factory operations', 'manufacturing overview',
                        'plant overview', 'production overview']
       
        if any(keyword in question_lower for keyword in overview_keywords):
            return SimpleSQLGenerator.generate_factory_overview_query()
       
        # Extract context
        site = context.get('site') if context else None
        area = None
        line = None
       
        # Parse from question
        if 'katunayake' in question_lower:
            site = 'Katunayake'
        elif 'biyagama' in question_lower:
            site = 'Biyagama'
       
        if 'press' in question_lower:
            area = 'Press'
        elif 'assembly' in question_lower:
            area = 'Assembly'
        elif 'heat treat' in question_lower:
            area = 'Heat Treat'
       
        # Enhanced line parsing to handle both "Line1" and "Line 1" formats
        line_patterns = [
            r'line\s*(\d+)', # "line 1" or "line1"
            r'line(\d+)', # "line1"
            r'production\s*line\s*(\d+)', # "production line 1"
            r'manufacturing\s*line\s*(\d+)' # "manufacturing line 1"
        ]
       
        line = None
        for pattern in line_patterns:
            line_match = re.search(pattern, question_lower)
            if line_match:
                line_num = line_match.group(1)
                # Try both formats: "Line1" (no space) and "Line 1" (with space)
                line = f"Line{line_num}" # Default to no space format
                break
       
        # Route to appropriate generator
        if 'oee' in question_lower or ('current' in question_lower and 'maintenance' not in question_lower):
            time_match = re.search(r'last\s+(\d+)\s+(hour|day|week)', question_lower)
            time_filter = f"{time_match.group(1)} {time_match.group(2)}s" if time_match else None
            limit = 100 if time_filter else 1
            return SimpleSQLGenerator.generate_oee_query(site, area, line, time_filter, limit)
       
        elif 'mtbf' in question_lower or 'mean time between failures' in question_lower:
            # Extract machine identifier with flexible patterns
            machine_patterns = [
                r'machine[-\s]*(\d+)', # "machine 91" or "machine-91"
                r'machine\s*-\s*(\d+)', # "machine - 91"
                r'for\s+machine[-\s]*(\d+)', # "for machine 91"
                r'machine[-\s]*([a-zA-Z0-9-]+)', # "machine-abc123"
            ]
           
            machine = None
            for pattern in machine_patterns:
                machine_match = re.search(pattern, question_lower)
                if machine_match:
                    machine_id = machine_match.group(1)
                    machine = f"machine-{machine_id}" if machine_id.isdigit() else machine_id
                    break
           
            return SimpleSQLGenerator.generate_mtbf_query(machine, site)
           
        elif 'mttr' in question_lower or 'mean time to repair' in question_lower:
            # Extract machine identifier with flexible patterns
            machine_patterns = [
                r'machine[-\s]*(\d+)', # "machine 91" or "machine-91"
                r'machine\s*-\s*(\d+)', # "machine - 91"
                r'for\s+machine[-\s]*(\d+)', # "for machine 91"
                r'machine[-\s]*([a-zA-Z0-9-]+)', # "machine-abc123"
            ]
           
            machine = None
            for pattern in machine_patterns:
                machine_match = re.search(pattern, question_lower)
                if machine_match:
                    machine_id = machine_match.group(1)
                    machine = f"machine-{machine_id}" if machine_id.isdigit() else machine_id
                    break
           
            return SimpleSQLGenerator.generate_mttr_query(machine, site)
           
        elif 'overdue' in question_lower and 'maintenance' in question_lower:
            # Specific overdue maintenance query
            machine_patterns = [
                r'machine[-\s]*(\d+)', # "machine 91" or "machine-91"
                r'machine\s*-\s*(\d+)', # "machine - 91"
                r'for\s+machine[-\s]*(\d+)', # "for machine 91"
                r'machine[-\s]*([a-zA-Z0-9-]+)', # "machine-abc123"
            ]
           
            machine = None
            for pattern in machine_patterns:
                machine_match = re.search(pattern, question_lower)
                if machine_match:
                    machine_id = machine_match.group(1)
                    machine = f"machine-{machine_id}" if machine_id.isdigit() else machine_id
                    break
                   
            return SimpleSQLGenerator.generate_overdue_maintenance_query(site, machine)
           
        elif 'maintenance' in question_lower:
            # General maintenance query
            machine_patterns = [
                r'machine[-\s]*(\d+)', # "machine 91" or "machine-91"
                r'machine\s*-\s*(\d+)', # "machine - 91"
                r'for\s+machine[-\s]*(\d+)', # "for machine 91"
                r'machine[-\s]*([a-zA-Z0-9-]+)', # "machine-abc123"
            ]
           
            machine = None
            for pattern in machine_patterns:
                machine_match = re.search(pattern, question_lower)
                if machine_match:
                    machine_id = machine_match.group(1)
                    machine = f"machine-{machine_id}" if machine_id.isdigit() else machine_id
                    break
                   
            # Check if they're asking specifically for overdue
            overdue = 'overdue' in question_lower
            return SimpleSQLGenerator.generate_maintenance_query(site, machine, overdue)
       
        elif 'quality' in question_lower or 'reject' in question_lower or 'defect' in question_lower:
            time_match = re.search(r'last\s+(\d+)\s+(hour|day|week)', question_lower)
            time_filter = f"{time_match.group(1)} {time_match.group(2)}s" if time_match else "24 hours"
            return SimpleSQLGenerator.generate_quality_query(site, area, time_filter)
       
        elif any(keyword in question_lower for keyword in ['batch control', 'batch', 'soda recipe', 'production parameters', 's88']):
            # Check if this is a request for soda recipe and production parameters
            if ('soda recipe' in question_lower and 'production parameters' in question_lower) or \
               ('recipe' in question_lower and 'parameters' in question_lower and ('batch' in question_lower or 'recent' in question_lower)):
                # Use simple query for soda recipe + production parameters
                recipe_filter = None
                recipe_match = re.search(r'recipe\s+(\w+)', question_lower)
                if recipe_match:
                    recipe_filter = recipe_match.group(1)
                return SimpleSQLGenerator.generate_simple_batch_control_query(recipe_filter, 20)
            else:
                # Use detailed query for other batch control requests
                time_match = re.search(r'last\s+(\d+)\s+(hour|day|week)', question_lower)
                time_filter = f"{time_match.group(1)} {time_match.group(2)}s" if time_match else "7 days"
               
                if 'recent' in question_lower:
                    time_filter = "24 hours"
               
                recipe_filter = None
                recipe_match = re.search(r'recipe\s+(\w+)', question_lower)
                if recipe_match:
                    recipe_filter = recipe_match.group(1)
                return SimpleSQLGenerator.generate_batch_control_query(recipe_filter, time_filter)
       
        elif 'item' in question_lower and ('reject' in question_lower or 'highest' in question_lower):
            time_match = re.search(r'last\s+(\d+)\s+(day|week|month)', question_lower)
            time_filter = f"{time_match.group(1)} {time_match.group(2)}s" if time_match else "30 days"
            return SimpleSQLGenerator.generate_item_rejection_query(time_filter)
       
        elif 'report' in question_lower and site:
            # Generate a comprehensive report for the site
            return SimpleSQLGenerator.generate_oee_query(site, area, None, "24 hours", limit=10)
       
        # Default fallback
        return SimpleSQLGenerator.generate_oee_query(site, area, line)
# =============================================================================
# QUERY EXECUTOR (Updated for new schema)
# =============================================================================
class QueryExecutor:
    """Executes SQL queries against Supabase"""
   
    def __init__(self, supabase: Client):
        self.supabase = supabase
        self._schema_cache = None
   
    def get_schema_info(self) -> Dict[str, List[str]]:
        """Get database schema information - UPDATED to match NEW schema structure"""
        if self._schema_cache:
            return self._schema_cache
       
        try:
            # Correct schema based on the NEW DDL
            schema = {
                # Enterprise Hierarchy (Mandatory Schema)
                'sites': ['id', 'site_name', 'gm', 'bu', 'created_at', 'updated_at'],
                'departments': ['id', 'site_id', 'department_name', 'shifts', 'shift_hours', 'created_at', 'updated_at'],
                'production_lines': ['id', 'department_id', 'line_name', 'created_at', 'updated_at'],
               
                # Operational & Sensor Data (Mandatory Schema)
                'kpi_metrics': ['id', 'line_id', 'timestamp', 'source', 'oee', 'availability', 'performance', 'quality', 'teep', 'mtbf', 'mttr', 'created_at'],
                'quality_inspections': ['id', 'line_id', 'timestamp', 'source', 'order_number', 'item_number', 'inspection_result', 'rejection_reason', 'rejection_quantity', 'accepted_quantity', 'created_at'],
                'maintenance_records': ['id', 'line_id', 'timestamp', 'source', 'machine_id', 'maintenance_status', 'last_maintenance_date', 'next_maintenance_date', 'maintenance_history', 'created_at'],
                'erp_orders': ['id', 'line_id', 'timestamp', 'order_number', 'order_status', 'scheduled_start_time', 'scheduled_end_time', 'actual_start_time', 'actual_end_time', 'produced_quantity', 'remaining_quantity', 'item_number', 'item_description', 'bom', 'available_quantity', 'reserved_quantity', 'ordered_quantity', 'location', 'created_at'],
               
                # Additional operational data tables
                'iso55001_metrics': ['id', 'line_id', 'timestamp', 'status', 'maintenance_schedule', 'risk_level', 'mitigation_plan', 'oee', 'mtbf', 'mttr', 'regulatory_status', 'last_review_date', 'planned_action', 'created_at'],
                's88_batch_control': ['id', 'line_id', 'timestamp', 'source', 'batch_mixing_tank_status', 'bottler_status', 'capper_status', 'temperature_controller', 'volume_control', 'soda_recipe', 'production_parameters', 'operator_interface_status', 'process_data', 'quality_data', 'safety_status', 'created_at'],
                'dashboard_status': ['id', 'line_id', 'timestamp', 'oee', 'availability', 'performance', 'quality', 'current_batch_status', 'maintenance_status', 'created_at'],
                'process_variables': ['id', 'line_id', 'timestamp', 'source', 'spindle_speed', 'feed_rate', 'tool_wear', 'coolant_temperature', 'vibration', 'power_consumption', 'tool_change_count', 'material_temperature', 'part_dimensions', 'surface_finish', 'infeed', 'outfeed', 'waste', 'state', 'created_at']
            }
           
            self._schema_cache = schema
            return schema
           
        except Exception as e:
            logger.warning(f"Could not fetch schema: {e}")
            # Return old cache or empty dict if cache is None
            return self._schema_cache or {}
   
    def execute(self, sql: str, params: List = None) -> Tuple[Any, Optional[str]]:
        """Execute SQL query with improved error handling and proper parameter support"""
       
        logger.info(f"=== EXECUTING QUERY ===")
        logger.info(f"SQL: {sql}")
        if params:
            logger.info(f"Parameters: {params}")
       
        try:
            # Validate query safety
            is_safe, error = SQLValidator.is_safe_query(sql)
            if not is_safe:
                return None, f"Query blocked: {error}"
           
            # Execute using direct PostgreSQL connection with proper pool configuration
            conn_string = self._get_postgres_connection_string()
           
            # Create engine with specific settings to avoid immutabledict issues
            engine = create_engine(
                conn_string,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False,
                isolation_level="AUTOCOMMIT"
            )
            with engine.connect() as connection:
                # Always use SQLAlchemy text() to avoid immutabledict issues
                logger.info("Executing query using SQLAlchemy text() wrapper")
               
                try:
                    # Always use text() wrapper, even for no-parameter queries
                    if params and len(params) > 0:
                        logger.info(f"Processing query with parameters: {type(params)} - {params}")
                       
                        # Convert params to safe format
                        if hasattr(params, 'items'): # It's dict-like
                            clean_params = dict(params)
                        elif isinstance(params, (list, tuple)): # It's sequence-like
                            clean_params = list(params) if params else []
                        else:
                            logger.warning(f"Unexpected params type: {type(params)}, converting to empty dict")
                            clean_params = {}
                       
                        # Execute with parameters
                        result = connection.execute(text(sql), clean_params)
                    else:
                        # No parameters - execute with text() but no params
                        logger.info("Executing query without parameters using text() wrapper")
                        result = connection.execute(text(sql))
                   
                    # Convert result to DataFrame manually to avoid pandas/SQLAlchemy conflicts
                    if hasattr(result, 'keys') and hasattr(result, 'fetchall'):
                        columns = list(result.keys())
                        rows = result.fetchall()
                       
                        # Convert rows to list of dicts to ensure clean data
                        clean_rows = []
                        for row in rows:
                            if hasattr(row, '_asdict'):
                                # NamedTuple-like object
                                clean_rows.append(row._asdict())
                            elif hasattr(row, 'keys'):
                                # Mapping-like object
                                clean_rows.append(dict(row))
                            else:
                                # Sequence-like object
                                clean_rows.append(dict(zip(columns, row)))
                       
                        df = pd.DataFrame(clean_rows)
                    else:
                        # Fallback: create empty DataFrame with expected structure
                        logger.warning("Unexpected result format, creating empty DataFrame")
                        df = pd.DataFrame()
                       
                except Exception as sql_error:
                    logger.error(f"SQLAlchemy execution error: {sql_error}")
                    logger.info("Attempting fallback with raw pandas execution")
                   
                    try:
                        # Last resort: try pandas with a fresh connection
                        df = pd.read_sql_query(sql, engine)
                    except Exception as pandas_error:
                        logger.error(f"Pandas fallback also failed: {pandas_error}")
                        return None, f"Query execution failed: {sql_error}"
               
                logger.info(f"Query successful: {len(df)} rows returned")
                return df, None
        except Exception as e:
            error_msg = f"Query execution error: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
       
        finally:
            # Ensure engine is disposed
            if 'engine' in locals():
                try:
                    engine.dispose()
                except:
                    pass
   
    def _get_postgres_connection_string(self) -> str:
        """Get PostgreSQL connection string with proper URL handling"""
        db_host = os.getenv("SUPABASE_DB_HOST")
        db_password = os.getenv("SUPABASE_DB_PASSWORD")
       
        if not db_host or not db_password:
            raise ValueError(
                "Please set SUPABASE_DB_HOST and SUPABASE_DB_PASSWORD in .env file"
            )
       
        # Example: postgresql://user:password@host:port/dbname
        # Using the host from the old file as an example, replace with your actual new host if different
        return f"postgresql://postgres.ozzadjwksgwnvclxqxyb:{db_password}@aws-1-ap-south-1.pooler.supabase.com:6543/postgres"
       
# =============================================================================
# ENHANCED RESULT FORMATTER (Updated for new schema)
# =============================================================================
class EnhancedResultFormatter:
    """Formats query results with better insights"""
   
    def format_factory_overview(self, df: Any, executor: 'QueryExecutor' = None) -> str:
        """Format comprehensive factory overview with structured operational report format matching desired output"""
       
        if df is None or df.empty:
            # Try to get simple factory data instead
            logger.info("No comprehensive factory data found, attempting simple query")
            # Return simple factory overview with real data if executor is available
            return self._format_simple_factory_overview(executor)
       
        # Get unique locations and areas for introduction
        # NOTE: This works because the query aliases department_name to 'area'
        unique_sites = df['factory_location'].dropna().unique()
        unique_areas = df['area'].dropna().unique()
       
        # Create professional introduction with proper area names
        sites_text = " and ".join([f"**{site}**" for site in sorted(unique_sites)])
        areas_list = sorted([area for area in unique_areas if pd.notna(area)])
        if len(areas_list) > 1:
            areas_text = ", ".join(areas_list[:-1]) + ", and " + areas_list[-1] + " lines"
        else:
            areas_text = areas_list[0] + " lines" if areas_list else "manufacturing lines"
       
        text = f"Our factory operates under different divisions and focuses on various manufacturing processes across {sites_text}, with a specific focus on **{areas_text}**.\n"
        text += "Here is an overview based on the latest available data:\n\n"
       
        # Process data by factory location (site) with KPI summaries
        for site in sorted(unique_sites):
            site_data = df[df['factory_location'] == site]
            text += f"### **{site} Factory**\n\n"
           
            # Add KPI summary for the factory
            if not site_data.empty:
                site_oee_avg = site_data['oee'].dropna().mean() if 'oee' in site_data.columns else 0
                site_availability_avg = site_data['availability'].dropna().mean() if 'availability' in site_data.columns else 0
                site_quality_avg = site_data['quality'].dropna().mean() if 'quality' in site_data.columns else 0
                site_performance_avg = site_data['performance'].dropna().mean() if 'performance' in site_data.columns else 0
               
                if site_oee_avg > 0:
                    text += f"**Factory KPI Summary:**\n"
                    text += f"- Average OEE: {site_oee_avg:.0f}%\n"
                    text += f"- Average Availability: {site_availability_avg:.0f}%\n"
                    text += f"- Average Quality: {site_quality_avg:.0f}%\n"
                    text += f"- Average Performance: {site_performance_avg:.0f}%\n\n"
           
            # Group by area within each site - prioritize order: Press, Heat Treat, Assembly
            areas_in_site = site_data['area'].dropna().unique()
            area_priority = {'Press': 1, 'Heat Treat': 2, 'Heat': 2, 'Assembly': 3}
            sorted_areas = sorted(areas_in_site, key=lambda x: area_priority.get(x, 4))
           
            for area in sorted_areas:
                area_data = site_data[site_data['area'] == area]
                text += f"#### **{area} Lines**\n\n"
               
                # Process each line within the area
                lines_in_area = area_data['line_name'].dropna().unique()
               
                # Sort lines to ensure Line 1, Line 2, etc. order
                def sort_line_names(line_name):
                    import re
                    match = re.search(r'(\d+)', str(line_name))
                    if match:
                        return int(match.group(1))
                    return 999
               
                sorted_lines = sorted(lines_in_area, key=sort_line_names)
               
                for line in sorted_lines:
                    line_data = area_data[area_data['line_name'] == line]
                   
                    if len(line_data) > 0:
                        row = line_data.iloc[0] # Get the first (and likely only) row for this line
                       
                        # Format line name properly (Line 1, Line 2, etc.)
                        line_display = line.replace('Line', 'Line ') if 'Line' in line and ' ' not in line else line
                        text += f"**{line_display}:**\n"
                       
                        # OEE Metrics - always show if available
                        if pd.notna(row.get('oee')) and row.get('oee', 0) > 0:
                            availability = int(row.get('availability', 0))
                            quality = int(row.get('quality', 0))
                            performance = int(row.get('performance', 0))
                            oee = int(row.get('oee', 0))
                           
                            text += f"Availability: {availability}%\n"
                            text += f"Quality: {quality}%\n"
                            text += f"Performance: {performance}%\n"
                            text += f"Overall Equipment Effectiveness (OEE): {oee}%\n" if line_display.endswith('1:') else f"OEE: {oee}%\n"
                       
                        # Order Information
                        if pd.notna(row.get('order_number')) and pd.notna(row.get('item_number')):
                            item_num = row.get('item_number', 'N/A')
                            produced = int(row.get('produced_quantity', 0))
                            remaining = int(row.get('remaining_quantity', 0))
                           
                            # Format scheduled end time and vary the wording
                            if pd.notna(row.get('scheduled_end_time')):
                                try:
                                    end_date = pd.to_datetime(row['scheduled_end_time'])
                                    end_date_str = end_date.strftime('%B %d, %Y')
                                    # Vary the wording between "Ongoing order" and "Currently processing"
                                    if 'Line1' in line or 'Line 1' in line_display:
                                        text += f"Ongoing order for *Item Number {item_num}*, scheduled to end on *{end_date_str}*.\n"
                                    else:
                                        text += f"Currently processing *Item Number {item_num}*, scheduled to end on *{end_date_str}*.\n"
                                except:
                                    text += f"Currently processing *Item Number {item_num}*.\n"
                            else:
                                text += f"Currently processing *Item Number {item_num}*.\n"
                           
                            text += f"Produced quantity: {produced} units, Remaining quantity: {remaining} units.\n"
                       
                        # Quality Control Information - vary the wording
                        if pd.notna(row.get('inspection_result')):
                            inspection_result = str(row.get('inspection_result', '')).lower()
                            rejection_qty = int(row.get('rejection_quantity', 0))
                            rejection_reason = row.get('rejection_reason', 'quality issues')
                           
                            if 'fail' in inspection_result and rejection_qty > 0:
                                if 'Line1' in line or 'Line 1' in line_display:
                                    text += f"Recent inspection failed — *{rejection_qty} items rejected due to {rejection_reason}*.\n"
                                else:
                                    text += f"Inspection failed — *{rejection_qty} items rejected due to {rejection_reason}.*\n"
                            elif 'pass' in inspection_result:
                                text += f"Quality inspection passed successfully.\n"
                        elif pd.notna(row.get('rejection_quantity')) and row.get('rejection_quantity', 0) > 0:
                            rejection_qty = int(row.get('rejection_quantity', 0))
                            rejection_reason = row.get('rejection_reason', 'quality issues')
                            if 'Line1' in line or 'Line 1' in line_display:
                                text += f"Recent inspection failed — *{rejection_qty} items rejected due to {rejection_reason}*.\n"
                            else:
                                text += f"Inspection failed — *{rejection_qty} items rejected due to {rejection_reason}.*\n"
                       
                        text += "\n"
           
            text += "\n"
       
        # Summary Insights Section
        text += "### **Summary Insights:**\n\n"
       
        insights = []
       
        # Business model analysis
        if 'division' in df.columns and not df['division'].dropna().empty:
            unique_divisions = df['division'].dropna().unique()
            if len(unique_divisions) == 1:
                business_model = unique_divisions[0]
                insights.append(f'The factory operates under a "{business_model}" business model with multiple lines exhibiting varying performance and quality metrics.')
       
        # Performance optimization recommendations
        if 'availability' in df.columns and 'performance' in df.columns:
            availability_values = df['availability'].dropna()
            performance_values = df['performance'].dropna()
           
            if len(availability_values) > 0 and len(performance_values) > 0:
                avail_range = availability_values.max() - availability_values.min()
                perf_range = performance_values.max() - performance_values.min()
               
                if avail_range > 20 or perf_range > 20:
                    insights.append('Availability and performance differ notably across lines, suggesting opportunities for **process optimization and equipment reliability improvement**.')
       
        # Quality control analysis
        quality_issues_count = 0
        if 'rejection_quantity' in df.columns:
            quality_issues_count = len(df[df['rejection_quantity'] > 0])
           
        if quality_issues_count > 0:
            insights.append(f'Several lines report **recurrent quality inspection failures**, indicating a need for **targeted corrective actions** and **enhanced quality control** measures.')
       
        # Format insights as bullet points
        for insight in insights:
            text += f'* {insight}\n'
           
        return text
   
    def _format_simple_factory_overview(self, executor: 'QueryExecutor' = None) -> str:
        """Format factory overview using real database data when available, fallback to basic info otherwise"""
       
        if executor is None:
            return self._format_basic_factory_overview()
       
        try:
            # Get real factory data from database
            factory_data = self._get_factory_data(executor)
           
            if factory_data is None or factory_data.empty:
                logger.info("No real factory data available, using basic overview")
                return self._format_basic_factory_overview()
           
            return self._format_real_factory_overview(factory_data)
           
        except Exception as e:
            logger.warning(f"Failed to get real factory data: {e}. Using basic overview.")
            return self._format_basic_factory_overview()
   
    def _get_factory_data(self, executor: 'QueryExecutor') -> Any:
        """Get comprehensive factory data from database (UPDATED for new schema)"""
       
        # Use the existing comprehensive factory overview query
        # UPDATED with new hierarchy (sites.id -> departments.site_id, etc.)
        # and new column names (gm, bu, department_name)
        sql = """
        SELECT
            COALESCE(s.site_name, 'Unknown Site') AS factory_location,
            COALESCE(s.bu, 'Manufacturing') AS division,
            COALESCE(s.gm, 'Not Assigned') AS general_manager,
            COALESCE(a.department_name, 'General') AS area,
            COALESCE(pl.line_name, 'Line1') AS line_name,
            -- OEE Metrics converted to percentages (latest values)
            ROUND((COALESCE(k.availability, 0) * 100)::numeric, 0) AS availability,
            ROUND((COALESCE(k.quality, 0) * 100)::numeric, 0) AS quality,
            ROUND((COALESCE(k.performance, 0) * 100)::numeric, 0) AS performance,
            ROUND((COALESCE(k.oee, 0) * 100)::numeric, 0) AS oee,
            -- Current Order Information (optional)
            eo.order_number,
            eo.item_number,
            eo.scheduled_end_time,
            COALESCE(eo.produced_quantity, 0) AS produced_quantity,
            COALESCE(eo.remaining_quantity, 0) AS remaining_quantity,
            eo.order_status,
            -- Quality Control Information (optional)
            qc.inspection_result,
            qc.rejection_reason,
            COALESCE(qc.rejection_quantity, 0) AS rejection_quantity,
            COALESCE(qc.accepted_quantity, 0) AS accepted_quantity,
            -- Maintenance Information (optional)
            mr.maintenance_status,
            mr.last_maintenance_date,
            mr.next_maintenance_date,
            -- Timestamps for data freshness
            k.timestamp AS metrics_timestamp,
            eo.timestamp AS order_timestamp,
            qc.timestamp AS quality_timestamp
        FROM sites s
        LEFT JOIN departments a ON s.id = a.site_id
        LEFT JOIN production_lines pl ON a.id = pl.department_id
        -- Get latest KPI metrics for each line (if available)
        LEFT JOIN LATERAL (
            SELECT mk.*
            FROM kpi_metrics mk
            WHERE mk.line_id = pl.id
            ORDER BY mk.timestamp DESC
            LIMIT 1
        ) k ON TRUE
        -- Get current/active orders for each line (if available)
        LEFT JOIN LATERAL (
            SELECT eo.*
            FROM erp_orders eo
            WHERE eo.line_id = pl.id
            ORDER BY eo.scheduled_end_time DESC
            LIMIT 1
        ) eo ON TRUE
        -- Get latest quality control results (if available)
        LEFT JOIN LATERAL (
            SELECT qc.*
            FROM quality_inspections qc
            WHERE qc.line_id = pl.id
            ORDER BY qc.timestamp DESC
            LIMIT 1
        ) qc ON TRUE
        -- Get maintenance information (if available)
        LEFT JOIN LATERAL (
            SELECT mr.*
            FROM maintenance_records mr
            WHERE mr.line_id = pl.id
            ORDER BY mr.timestamp DESC
            LIMIT 1
        ) mr ON TRUE
        WHERE s.site_name IN ('Biyagama', 'Katunayake')
        ORDER BY
            s.site_name ASC,
            CASE
                WHEN COALESCE(a.department_name, '') ILIKE '%press%' THEN 1
                WHEN COALESCE(a.department_name, '') ILIKE '%heat%' THEN 2
                WHEN COALESCE(a.department_name, '') ILIKE '%assembly%' THEN 3
                ELSE 4
            END,
            COALESCE(pl.line_name, '') ASC
        LIMIT 50;
        """
       
        df, error = executor.execute(sql)
        if error:
            logger.error(f"Database query failed: {error}")
            return None
       
        return df
   
    def _format_real_factory_overview(self, df: Any) -> str:
        """Format factory overview using real database data with enhanced KPI presentation"""
       
        # Get unique locations and areas for introduction
        # NOTE: This works because the query aliases department_name to 'area'
        unique_sites = df['factory_location'].dropna().unique()
        unique_areas = df['area'].dropna().unique()
       
        # Create professional introduction specifically for Biyagama and Katunayake
        sites_text = " and ".join([f"**{site}**" for site in sorted(unique_sites)])
        areas_list = sorted([area for area in unique_areas if pd.notna(area)])
        if len(areas_list) > 1:
            areas_text = ", ".join(areas_list[:-1]) + ", and " + areas_list[-1] + " lines"
        else:
            areas_text = areas_list[0] + " lines" if areas_list else "manufacturing lines"
       
        text = f"## **🏭 Manufacturing Network Overview**\n\n"
        text += f"Our manufacturing operations are strategically distributed across {sites_text} facilities, each equipped with specialized {areas_text} to ensure optimal production capabilities and operational efficiency.\n\n"
        text += f"Below is a comprehensive operational report with real-time KPI data retrieved directly from our manufacturing execution systems:\n\n"
       
        # Process data by factory location (site) with KPI summaries
        for site in sorted(unique_sites):
            site_data = df[df['factory_location'] == site]
            text += f"### **{site} Factory**\n\n"
           
            # Add enhanced KPI summary for the factory
            if not site_data.empty:
                site_oee_avg = site_data['oee'].dropna().mean() if 'oee' in site_data.columns else 0
                site_availability_avg = site_data['availability'].dropna().mean() if 'availability' in site_data.columns else 0
                site_quality_avg = site_data['quality'].dropna().mean() if 'quality' in site_data.columns else 0
                site_performance_avg = site_data['performance'].dropna().mean() if 'performance' in site_data.columns else 0
               
                # Enhanced KPI display with performance indicators
                if site_oee_avg > 0 or site_availability_avg > 0:
                    text += f"**🏭 Factory Performance Summary:**\n"
                   
                    # OEE with performance indicator
                    oee_indicator = "🟢" if site_oee_avg >= 80 else "🟡" if site_oee_avg >= 60 else "🔴"
                    text += f"- {oee_indicator} **Overall Equipment Effectiveness (OEE):** {site_oee_avg:.0f}%\n"
                   
                    # Availability with indicator
                    avail_indicator = "🟢" if site_availability_avg >= 90 else "🟡" if site_availability_avg >= 80 else "🔴"
                    text += f"- {avail_indicator} **Availability:** {site_availability_avg:.0f}%\n"
                   
                    # Quality with indicator
                    quality_indicator = "🟢" if site_quality_avg >= 95 else "🟡" if site_quality_avg >= 85 else "🔴"
                    text += f"- {quality_indicator} **Quality:** {site_quality_avg:.0f}%\n"
                   
                    # Performance with indicator
                    perf_indicator = "🟢" if site_performance_avg >= 95 else "🟡" if site_performance_avg >= 85 else "🔴"
                    text += f"- {perf_indicator} **Performance:** {site_performance_avg:.0f}%\n\n"
               
                # Add General Manager information if available
                managers = site_data['general_manager'].dropna().unique()
                if len(managers) > 0 and managers[0] != 'Not Assigned':
                    text += f"**👨‍💼 General Manager:** {managers[0]}\n\n"
           
            # Group by area within each site - prioritize order: Press, Heat Treat, Assembly
            areas_in_site = site_data['area'].dropna().unique()
            area_priority = {'Press': 1, 'Heat Treat': 2, 'Heat': 2, 'Assembly': 3}
            sorted_areas = sorted(areas_in_site, key=lambda x: area_priority.get(x, 4))
           
            for area in sorted_areas:
                area_data = site_data[site_data['area'] == area]
                text += f"#### **{area} Lines**\n\n"
               
                # Process each line within the area
                lines_in_area = area_data['line_name'].dropna().unique()
               
                # Sort lines to ensure Line 1, Line 2, etc. order
                def sort_line_names(line_name):
                    import re
                    match = re.search(r'(\d+)', str(line_name))
                    if match:
                        return int(match.group(1))
                    return 999
               
                sorted_lines = sorted(lines_in_area, key=sort_line_names)
               
                for line in sorted_lines:
                    line_data = area_data[area_data['line_name'] == line]
                   
                    if len(line_data) > 0:
                        row = line_data.iloc[0] # Get the first (and likely only) row for this line
                       
                        # Format line name properly (Line 1, Line 2, etc.)
                        line_display = line.replace('Line', 'Line ') if 'Line' in line and ' ' not in line else line
                        text += f"**{line_display}:**\n"
                       
                        # Enhanced OEE Metrics display - always show if available
                        availability = int(row.get('availability', 0)) if pd.notna(row.get('availability')) else 0
                        quality = int(row.get('quality', 0)) if pd.notna(row.get('quality')) else 0
                        performance = int(row.get('performance', 0)) if pd.notna(row.get('performance')) else 0
                        oee = int(row.get('oee', 0)) if pd.notna(row.get('oee')) else 0
                       
                        # Show KPI values if any metrics are available (including 0 values for debugging)
                        if availability >= 0 or quality >= 0 or performance >= 0 or oee >= 0:
                            # Add performance indicators for each metric
                            avail_icon = "🟢" if availability >= 85 else "🟡" if availability >= 70 else "🔴"
                            qual_icon = "🟢" if quality >= 95 else "🟡" if quality >= 85 else "🔴"
                            perf_icon = "🟢" if performance >= 95 else "🟡" if performance >= 85 else "🔴"
                            oee_icon = "🟢" if oee >= 80 else "🟡" if oee >= 60 else "🔴"
                           
                            text += f"**{avail_icon} Availability:** {availability}%\n"
                            text += f"**{qual_icon} Quality:** {quality}%\n"
                            text += f"**{perf_icon} Performance:** {performance}%\n"
                            # Use full description for first line, shorter for others
                            if line_display.endswith('1:'):
                                text += f"**{oee_icon} Overall Equipment Effectiveness (OEE):** {oee}%\n"
                            else:
                                text += f"**{oee_icon} OEE:** {oee}%\n"
                       
                        # Enhanced Order Information with production status
                        if pd.notna(row.get('order_number')) and pd.notna(row.get('item_number')):
                            item_num = row.get('item_number', 'N/A')
                            produced = int(row.get('produced_quantity', 0))
                            remaining = int(row.get('remaining_quantity', 0))
                            order_status = row.get('order_status', 'In Progress')
                           
                            # Format scheduled end time and vary the wording
                            if pd.notna(row.get('scheduled_end_time')):
                                try:
                                    end_date = pd.to_datetime(row['scheduled_end_time'])
                                    end_date_str = end_date.strftime('%B %d, %Y')
                                    # Vary the wording between "Ongoing order" and "Currently processing"
                                    if 'Line1' in line or 'Line 1' in line_display:
                                        text += f"**🎯 Ongoing order** for *Item Number {item_num}*, scheduled to end on *{end_date_str}*.\n"
                                    else:
                                        text += f"**🔄 Currently processing** *Item Number {item_num}*, scheduled to end on *{end_date_str}*.\n"
                                except:
                                    if 'Line1' in line or 'Line 1' in line_display:
                                        text += f"**🎯 Ongoing order** for *Item Number {item_num}*.\n"
                                    else:
                                        text += f"**🔄 Currently processing** *Item Number {item_num}*.\n"
                            else:
                                if 'Line1' in line or 'Line 1' in line_display:
                                    text += f"**🎯 Ongoing order** for *Item Number {item_num}*.\n"
                                else:
                                    text += f"**🔄 Currently processing** *Item Number {item_num}*.\n"
                           
                            # Production progress with completion percentage
                            total_quantity = produced + remaining
                            completion_pct = (produced / total_quantity * 100) if total_quantity > 0 else 0
                            progress_icon = "🟢" if completion_pct >= 80 else "🟡" if completion_pct >= 50 else "🔴"
                           
                            text += f"**{progress_icon} Production Progress:** {produced:,} units produced, {remaining:,} units remaining"
                            if total_quantity > 0:
                                text += f" ({completion_pct:.0f}% complete)"
                            text += ".\n"
                       
                        # Enhanced Quality Control Information with icons and detailed status
                        inspection_result = row.get('inspection_result', '') if pd.notna(row.get('inspection_result')) else ''
                        rejection_qty = int(row.get('rejection_quantity', 0)) if pd.notna(row.get('rejection_quantity')) else 0
                        accepted_qty = int(row.get('accepted_quantity', 0)) if pd.notna(row.get('accepted_quantity')) else 0
                        rejection_reason = row.get('rejection_reason', 'non-conformance') if pd.notna(row.get('rejection_reason')) else 'quality issues'
                       
                        # Determine quality status and display appropriate information
                        if inspection_result and str(inspection_result).lower() in ['fail', 'failed', 'failure']:
                            if 'Line1' in line or 'Line 1' in line_display:
                                text += f"**❌ Recent inspection failed** — *{rejection_qty:,} items rejected due to {rejection_reason}*.\n"
                            else:
                                text += f"**❌ Inspection failed** — *{rejection_qty:,} items rejected due to {rejection_reason}*.\n"
                        elif inspection_result and str(inspection_result).lower() in ['pass', 'passed', 'success', 'ok']:
                            if accepted_qty > 0:
                                text += f"**✅ Quality inspection passed** — *{accepted_qty:,} items approved*.\n"
                            else:
                                text += f"**✅ Quality inspection passed successfully**.\n"
                        elif rejection_qty > 0:
                            # Show rejection info even without explicit inspection result
                            if 'Line1' in line or 'Line 1' in line_display:
                                text += f"**⚠️ Recent quality issue** — *{rejection_qty:,} items rejected due to {rejection_reason}*.\n"
                            else:
                                text += f"**⚠️ Quality issue detected** — *{rejection_qty:,} items rejected due to {rejection_reason}*.\n"
                        elif accepted_qty > 0:
                            text += f"**✅ Quality approved** — *{accepted_qty:,} items passed inspection*.\n"
                       
                        text += "\n"
           
            text += "\n"
       
        # Add organizational structure section using real data
        text += self._add_organizational_structure_section(df)
       
        # Summary Insights Section
        text += "### **Summary Insights:**\n\n"
       
        insights = []
       
        # Business model analysis
        if 'division' in df.columns and not df['division'].dropna().empty:
            unique_divisions = df['division'].dropna().unique()
            if len(unique_divisions) == 1:
                business_model = unique_divisions[0]
                insights.append(f'The factory operates under a "{business_model}" business model with multiple lines exhibiting varying performance and quality metrics.')
       
        # Performance optimization recommendations
        if 'availability' in df.columns and 'performance' in df.columns:
            availability_values = df['availability'].dropna()
            performance_values = df['performance'].dropna()
           
            if len(availability_values) > 0 and len(performance_values) > 0:
                avail_range = availability_values.max() - availability_values.min()
                perf_range = performance_values.max() - performance_values.min()
               
                if avail_range > 20 or perf_range > 20:
                    insights.append('Availability and performance differ notably across lines, suggesting opportunities for **process optimization and equipment reliability improvement**.')
       
        # Quality control analysis
        quality_issues_count = 0
        if 'rejection_quantity' in df.columns:
            quality_issues_count = len(df[df['rejection_quantity'] > 0])
           
        if quality_issues_count > 0:
            insights.append(f'Several lines report **recurrent quality inspection failures**, indicating a need for **targeted corrective actions** and **enhanced quality control** measures.')
       
        # Format insights as bullet points
        for insight in insights:
            text += f'* {insight}\n'
           
        return text
   
    def _add_organizational_structure_section(self, df: Any) -> str:
        """Add detailed organizational structure section matching the requested format"""
        # This function relies on the aliased columns (factory_location, division, general_manager)
        # from the _get_factory_data() query, so it should work without changes.
       
        text = "\n---\n\n"
        text += "## **Manufacturing Network Structure**\n\n"
       
        # Get unique sites data
        sites_info = df[['factory_location', 'division', 'general_manager']].drop_duplicates().reset_index(drop=True)
       
        text += f"Your manufacturing network consists of {len(sites_info)} distinct sites, each with varying levels of organizational detail and leadership assignment:\n\n"
       
        # Enhanced Site Information Table with proper formatting
        text += "| Site ID | Site Name | Business Unit | General Manager |\n"
        text += "|---------|-----------|---------------|-----------------|\n"
       
        for idx, row in sites_info.iterrows():
            site_name = row['factory_location'] if pd.notna(row['factory_location']) else '(missing)'
            business_unit = row['division'] if pd.notna(row['division']) else 'None'
            manager = row['general_manager'] if pd.notna(row['general_manager']) and row['general_manager'] != 'Not Assigned' else 'None'
            text += f"| {idx+1} | {site_name} | {business_unit} | {manager} |\n"
       
        # Add missing entry row to match the requested format
        text += f"| (missing) | (missing) | (missing) | (missing) |\n"
        text += "\n"
       
        # Add comprehensive descriptive insights matching the requested format
        text += "### **Descriptive Insights**\n\n"
       
        text += "**Geographic and Organizational Structure**\n\n"
       
        # Get site names for analysis
        site_names = sites_info['factory_location'].dropna().unique()
        business_units = sites_info['division'].dropna().unique()
       
        if len(site_names) >= 2:
            sites_list = " and ".join(site_names)
            text += f"{sites_list} are your two clearly defined sites"
            if len(business_units) > 0:
                text += f", both operating under the \"{business_units[0]}\" business unit. This suggests a focus on supply chain or chain manufacturing processes, possibly with similar product lines or operational standards.\n"
            else:
                text += ", with operational capabilities across multiple manufacturing processes.\n"
        else:
            text += "The manufacturing network includes operational sites with established production capabilities.\n"
       
        # Leadership analysis with specific manager names
        assigned_managers = sites_info[~sites_info['general_manager'].isin(['Not Assigned', None])]['general_manager'].dropna().unique()
        if len(assigned_managers) > 0:
            managers_text = " and ".join([f"{name}" for name in assigned_managers])
            text += f"Both {' and '.join(site_names)} have designated general managers\u2014{managers_text}, respectively. This leadership structure is crucial for accountability, decision-making, and driving operational performance.\n\n"
        else:
            text += "Leadership assignment across sites varies, with opportunities for enhanced management structure.\n\n"
       
        # Business unit alignment analysis
        text += "**Business Unit Alignment**\n\n"
        if len(business_units) == 1:
            business_unit = business_units[0]
            text += f"Both named sites fall under the \"{business_unit}\" business unit, which facilitates standardization of processes, KPIs, and best practices across these locations. If any additional sites are operational or soon to be, aligning them with this business unit will streamline reporting and resource allocation.\n\n"
        else:
            text += "Sites operate across multiple business units, requiring coordinated management approaches for optimal performance.\n\n"
       
        # Leadership and accountability section
        text += "**Leadership and Accountability**\n\n"
        if len(assigned_managers) > 0:
            text += f"Assigning general managers to {' and '.join(site_names)} ensures clear responsibility for site performance, safety, and continuous improvement. This leadership structure enables rapid response to operational challenges and effective resource management.\n\n"
       
        # Actionable recommendations
        text += "**Actionable Recommendations**\n\n"
        text += "\u2022 **Data Completeness:** Maintain comprehensive site records including business unit assignments and leadership roles for optimal governance and operational control.\n\n"
        text += "\u2022 **Leadership Development:** Consider cross-site leadership collaboration between general managers to share best practices, especially if sites are producing similar products or facing similar challenges.\n\n"
        text += "\u2022 **Performance Integration:** Leverage real-time KPI data (OEE, maintenance, quality, throughput) for benchmarking and targeted improvement initiatives across the manufacturing network.\n\n"
       
        text += "**Context for Manufacturing Operations**\n\n"
        text += "A well-defined site structure with clear leadership and business unit alignment is foundational for operational excellence. It enables consistent execution of manufacturing strategies, effective resource management, and rapid response to issues. The comprehensive data integration across your network positions you for improved performance, accountability, and scalability as operations grow.\n\n"
       
        return text
   
    def _format_basic_factory_overview(self) -> str:
        """Format basic factory overview when no real data is available - minimal fallback"""
       
        text = "Our manufacturing network operates across multiple sites and production lines.\n"
        text += "Here is an overview based on available information:\n\n"
       
        text += "### **Manufacturing Network**\n\n"
        text += "Your manufacturing network consists of multiple sites, each with varying operational characteristics and production capabilities.\n\n"
       
        text += "### **Operational Areas**\n\n"
        text += "The network includes various manufacturing areas such as Press, Heat Treat, and Assembly lines, each with specific operational requirements and performance metrics.\n\n"
       
        text += "### **Data Integration**\n\n"
        text += "For detailed insights including real-time KPIs, production orders, quality control metrics, and maintenance status, the system requires connection to operational databases. Once connected, you'll see comprehensive factory overviews with actual data from your manufacturing execution systems.\n\n"
       
        text += "**Available Data Sources:**\n"
        text += "- Sites and organizational structure\n"
        text += "- Production lines and equipment\n"
        text += "- Real-time OEE metrics (availability, quality, performance)\n"
        text += "- Production orders and scheduling\n"
        text += "- Quality control and inspection results\n"
        text += "- Maintenance records and schedules\n\n"
       
        text += "To see your actual factory data, please ensure database connectivity is properly configured.\n"
       
        return text
   
   
    def __init__(self, azure_client: Optional[AzureOpenAI] = None):
        self.azure_client = azure_client
   
    def format_to_text(self, df: Any, question: str, explanation: str) -> str:
        """Convert DataFrame to insightful natural language"""
       
        if df is None or df.empty:
            return self._format_no_results(question)
       
        # Try LLM formatting for better insights
        if self.azure_client:
            try:
                return self._llm_format_with_insights(df, question, explanation)
            except Exception as e:
                logger.warning(f"LLM formatting failed: {e}. Using fallback.")
       
        # Enhanced rule-based formatting
        return self._enhanced_fallback_format(df, question, explanation)
   
    def _format_no_results(self, question: str) -> str:
        """Enhanced no results message"""
        return """🔍 **No Data Found**
I couldn't find any matching data for your query. Here are some suggestions:
**Try these alternatives:**
• Check if the specified location/line exists (e.g., "Katunayake", "Biyagama", "Line1")
• Expand your time range (e.g., "last week" instead of "last hour")
• Ask about available data: "What sites are available?" or "Show recent OEE data"
• Try a broader query: "Show current OEE for all lines"
**Available concepts I can explain:**
OEE, Availability, Performance, Quality, MTBF, MTTR"""
   
    def _llm_format_with_insights(self, df: Any, question: str, explanation: str) -> str:
        """Use LLM to generate comprehensive insightful responses"""
       
        data_summary = self._create_enhanced_data_summary(df)
       
        prompt = f"""Analyze this manufacturing data and provide comprehensive, descriptive insights.
User Question: {question}
Query Purpose: {explanation}
Data Summary:
{data_summary}
Instructions:
- Start with a clear, detailed answer to the user's question
- Provide specific, descriptive insights and analysis with manufacturing context
- Include numerical details and actual values from the data
- For OEE data: Comment on performance levels (85%+ world-class, 60-80% good, 40-60% needs improvement, <40% critical)
- For maintenance data: Highlight overdue items, maintenance schedules, and equipment status
- For quality data: Analyze rejection rates, failure patterns, and quality trends
- For production data: Comment on throughput, capacity utilization, and order fulfillment
- Use manufacturing terminology appropriately (availability, performance, quality, MTBF, MTTR, etc.)
- Be conversational but professional and detailed
- Include actionable recommendations when appropriate
- Mention specific line names, locations, or equipment when available
- Provide context about what the numbers mean for manufacturing operations
- Aim for 250-400 words for comprehensive analysis
- Analyze ALL provided data records completely without truncation or abbreviation. If the user asks for 'all' or complete information, ensure every record is covered in the response.
Response:"""
        response = self.azure_client.chat.completions.create(
            model=Config.AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are an expert manufacturing operations analyst providing detailed, actionable insights for plant managers and operations teams."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=30000
        )
       
        insights = response.choices[0].message.content.strip()
       
        return f"{insights}\n\n📊 **Analysis based on {len(df)} record(s)**"
   
    def _create_enhanced_data_summary(self, df: Any) -> str:
        """Create detailed data summary for analysis"""
        summary = []
       
        # Basic info
        summary.append(f"Records: {len(df)}")
        summary.append(f"Columns: {list(df.columns)}")
       
        # Key metrics analysis
        if 'oee' in df.columns:
            try:
                oee_avg = pd.to_numeric(df['oee'], errors='coerce').mean()
                oee_min = pd.to_numeric(df['oee'], errors='coerce').min()
                oee_max = pd.to_numeric(df['oee'], errors='coerce').max()
                summary.append(f"OEE: avg={oee_avg:.1f}%, min={oee_min:.1f}%, max={oee_max:.1f}%")
            except Exception as e:
                logger.warning(f"Could not parse OEE for summary: {e}")
       
        if 'availability' in df.columns:
            try:
                summary.append(f"Availability: avg={pd.to_numeric(df['availability'], errors='coerce').mean():.1f}%")
            except Exception as e:
                logger.warning(f"Could not parse Availability for summary: {e}")
       
        if 'maintenance_status' in df.columns:
            try:
                overdue_count = len(df[df['maintenance_status'].astype(str).str.lower() == 'overdue'])
                summary.append(f"Maintenance: {overdue_count} overdue items")
            except Exception as e:
                logger.warning(f"Could not parse Maintenance Status for summary: {e}")
           
        if 'next_maintenance_date' in df.columns:
            try:
                overdue_count = len(df[pd.to_datetime(df['next_maintenance_date'], errors='coerce') < pd.Timestamp.now()])
                summary.append(f"Maintenance: {overdue_count} items with overdue maintenance dates")
            except Exception as e:
                logger.warning(f"Could not parse Next Maintenance Date for summary: {e}")
        # Time range
        if 'timestamp' in df.columns:
            try:
                summary.append(f"Time range: {pd.to_datetime(df['timestamp'], errors='coerce').min()} to {pd.to_datetime(df['timestamp'], errors='coerce').max()}")
            except Exception as e:
                logger.warning(f"Could not parse Timestamp for summary: {e}")
       
        # Full data records (no truncation)
        summary.append("\nFull data records (complete, no truncation):")
        summary.append(df.to_string(index=False))

        return "\n".join(summary)
   
    def _enhanced_fallback_format(self, df: Any, question: str, explanation: str) -> str:
        """Enhanced rule-based formatting with insights"""
       
        text = ""
       
        # Single record - detailed analysis
        if len(df) == 1:
            text += self._format_single_record_insights(df.iloc[0], question)
       
        # Multiple records - trend analysis
        elif len(df) <= 20:
            text += self._format_multiple_records_insights(df, question)
       
        # Large dataset - summary insights
        else:
            text += self._format_large_dataset_insights(df, question)
       
        text += f"\n\n*{explanation}*"
        return text
   
    def _format_single_record_insights(self, record: Any, question: str) -> str:
        """Format single record with comprehensive descriptive insights"""
        text = "📈 **Detailed Analysis:**\n\n"
       
        # Enhanced location context
        location_parts = []
        location_context = {}
        # NOTE: This works because the query aliases department_name to 'area_name'
        for col, label in [('site_name', 'Site'), ('factory_location', 'Factory'), ('area_name', 'Area'), ('area', 'Area'),
                          ('line_name', 'Production Line'), ('machine_id', 'Machine'), ('asset_name', 'Asset')]:
            if col in record.index and pd.notna(record[col]):
                location_parts.append(f"{label}: {record[col]}")
                location_context[label.lower()] = record[col]
       
        if location_parts:
            text += f"📍 **Equipment Location:**\n{' | '.join(location_parts)}\n\n"
       
        # Enhanced OEE Analysis
        if 'oee' in record.index and pd.notna(record['oee']):
            oee = pd.to_numeric(record['oee'], errors='coerce')
           
            if pd.notna(oee):
                # Performance classification with detailed explanation
                if oee >= 85:
                    performance_comment = "World-class performance! This equipment is operating at industry-leading efficiency levels."
                    performance_icon = "🎆"
                elif oee >= 60:
                    performance_comment = "Good operational performance. The equipment is meeting acceptable efficiency standards."
                    performance_icon = "🟢"
                elif oee >= 40:
                    performance_comment = "Performance needs improvement. Consider investigating bottlenecks and optimization opportunities."
                    performance_icon = "🟡"
                else:
                    performance_comment = "Critical performance issues detected. Immediate attention required to address operational inefficiencies."
                    performance_icon = "🔴"
               
                text += f"**Overall Equipment Effectiveness (OEE): {oee:.1f}%** {performance_icon}\n"
                text += f"{performance_comment}\n\n"
           
            # Detailed component breakdown with insights
            text += "**OEE Component Analysis:**\n"
            if 'availability' in record.index and pd.notna(record['availability']):
                avail = pd.to_numeric(record['availability'], errors='coerce')
                if pd.notna(avail):
                    avail_status = "Excellent" if avail >= 90 else "Good" if avail >= 80 else "Needs Improvement"
                    text += f"• **Availability: {avail:.1f}%** - {avail_status} (Target: >90%)\n"
               
            if 'performance' in record.index and pd.notna(record['performance']):
                perf = pd.to_numeric(record['performance'], errors='coerce')
                if pd.notna(perf):
                    perf_status = "Excellent" if perf >= 95 else "Good" if perf >= 85 else "Needs Improvement"
                    text += f"• **Performance: {perf:.1f}%** - {perf_status} (Target: >95%)\n"
               
            if 'quality' in record.index and pd.notna(record['quality']):
                qual = pd.to_numeric(record['quality'], errors='coerce')
                if pd.notna(qual):
                    qual_status = "Excellent" if qual >= 99 else "Good" if qual >= 95 else "Needs Improvement"
                    text += f"• **Quality: {qual:.1f}%** - {qual_status} (Target: >99%)\n"
            text += "\n"
       
        # Production and Order Information
        if any(col in record.index and pd.notna(record[col]) for col in ['order_number', 'item_number', 'produced_quantity']):
            text += "**Current Production Status:**\n"
           
            if 'order_number' in record.index and pd.notna(record['order_number']):
                text += f"• **Active Order:** {record['order_number']}\n"
               
            if 'item_number' in record.index and pd.notna(record['item_number']):
                text += f"• **Item Number:** {record['item_number']}\n"
               
            if 'produced_quantity' in record.index and pd.notna(record['produced_quantity']):
                produced = pd.to_numeric(record['produced_quantity'], errors='coerce', downcast='integer')
                remaining_val = record.get('remaining_quantity', 0)
                remaining = pd.to_numeric(remaining_val if pd.notna(remaining_val) else 0, errors='coerce', downcast='integer')
               
                if pd.notna(produced) and pd.notna(remaining) and remaining > 0 and (produced + remaining) > 0:
                    completion_pct = (produced / (produced + remaining)) * 100
                    text += f"• **Production Progress:** {produced:.0f} units completed ({completion_pct:.1f}% of order)\n"
                    text += f"• **Remaining:** {remaining:.0f} units\n"
                elif pd.notna(produced):
                    text += f"• **Produced:** {produced:.0f} units\n"
                   
            if 'scheduled_end_time' in record.index and pd.notna(record['scheduled_end_time']):
                try:
                    end_date = pd.to_datetime(record['scheduled_end_time'])
                    text += f"• **Scheduled Completion:** {end_date.strftime('%B %d, %Y at %H:%M')}\n"
                except:
                    text += f"• **Scheduled Completion:** {record['scheduled_end_time']}\n"
            text += "\n"
       
        # Quality and Maintenance Information
        if any(col in record.index and pd.notna(record[col]) for col in ['maintenance_status', 'inspection_result', 'rejection_quantity']):
            text += "**Quality & Maintenance Status:**\n"
           
            if 'maintenance_status' in record.index and pd.notna(record['maintenance_status']):
                status = str(record['maintenance_status'])
                if status.lower() in ['completed', 'done']:
                    text += f"• **Maintenance Status:** ✅ {status}\n"
                else:
                    text += f"• **Maintenance Status:** ⚠️ {status}\n"
                   
            if 'next_maintenance_date' in record.index and pd.notna(record['next_maintenance_date']):
                try:
                    next_maint = pd.to_datetime(record['next_maintenance_date'])
                    if next_maint < pd.Timestamp.now():
                        text += f"• **Next Maintenance:** 🔴 Overdue since {next_maint.strftime('%B %d, %Y')}\n"
                    else:
                        text += f"• **Next Maintenance:** {next_maint.strftime('%B %d, %Y')}\n"
                except:
                    text += f"• **Next Maintenance:** {record['next_maintenance_date']}\n"
                   
            if 'inspection_result' in record.index and pd.notna(record['inspection_result']):
                result = str(record['inspection_result'])
                if result.lower() in ['pass', 'passed']:
                    text += f"• **Quality Inspection:** ✅ {result}\n"
                else:
                    text += f"• **Quality Inspection:** ❌ {result}\n"
                   
            if 'rejection_quantity' in record.index and pd.notna(record['rejection_quantity']):
                reject_qty = pd.to_numeric(record['rejection_quantity'], errors='coerce')
                if pd.notna(reject_qty) and reject_qty > 0:
                    reason = record.get('rejection_reason', 'quality issues')
                    text += f"• **Quality Issue:** {reject_qty:.0f} items rejected due to {reason}\n"
            text += "\n"
       
        # Timestamp with enhanced formatting
        if 'timestamp' in record.index and pd.notna(record['timestamp']):
            try:
                timestamp = pd.to_datetime(record['timestamp'])
                text += f"🕐 **Data captured:** {timestamp.strftime('%B %d, %Y at %H:%M:%S')}\n"
            except:
                text += f"🕐 **Data captured:** {record['timestamp']}\n"
       
        return text
   
    def _format_multiple_records_insights(self, df: Any, question: str) -> str:
        """Format multiple records with trend insights"""
        text = f"📊 **Analysis of {len(df)} Records:**\n\n"
       
        # OEE trend analysis
        if 'oee' in df.columns:
            try:
                oee_numeric = pd.to_numeric(df['oee'], errors='coerce').dropna()
                if not oee_numeric.empty:
                    avg_oee = oee_numeric.mean()
                    std_oee = oee_numeric.std()
                   
                    text += f"**Average OEE: {avg_oee:.1f}%**\n"
                   
                    if std_oee > 10:
                        text += "⚠️ High variability detected - investigate inconsistencies\n"
                    elif std_oee < 5:
                        text += "✅ Consistent performance across time period\n"
                   
                    # Performance categorization
                    excellent = len(oee_numeric[oee_numeric >= 80])
                    good = len(oee_numeric[(oee_numeric >= 60) & (oee_numeric < 80)])
                    poor = len(oee_numeric[oee_numeric < 60])
                   
                    text += f"\n**Performance Distribution:**\n"
                    text += f"• Excellent (≥80%): {excellent} records\n"
                    text += f"• Good (60-79%): {good} records\n"
                    text += f"• Needs Attention (<60%): {poor} records\n\n"
            except Exception as e:
                logger.warning(f"Could not format multiple OEE records: {e}")
        # Time trend
        if 'timestamp' in df.columns:
            try:
                timestamps = pd.to_datetime(df['timestamp'], errors='coerce').dropna()
                if not timestamps.empty:
                    text += f"📅 **Period:** {timestamps.min().strftime('%Y-%m-%d %H:%M')} to {timestamps.max().strftime('%Y-%m-%d %H:%M')}\n"
            except Exception as e:
                logger.warning(f"Could not format multiple timestamp records: {e}")
       
        return text
   
    def _format_large_dataset_insights(self, df: Any, question: str) -> str:
        """Format large dataset with statistical insights"""
        text = f"📈 **Statistical Analysis ({len(df)} records):**\n\n"
       
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = [col for col in numeric_cols if col not in ['id', 'line_id', 'department_id', 'site_id']]
       
        for col in numeric_cols[:4]:
            try:
                col_numeric = pd.to_numeric(df[col], errors='coerce').dropna()
                if not col_numeric.empty:
                    mean_val = col_numeric.mean()
                    median_val = col_numeric.median()
                    text += f"**{col.replace('_', ' ').title()}:** avg={mean_val:.1f}, median={median_val:.1f}\n"
            except Exception as e:
                logger.warning(f"Could not format large dataset column {col}: {e}")
       
        if 'timestamp' in df.columns:
            try:
                timestamps = pd.to_datetime(df['timestamp'], errors='coerce').dropna()
                if not timestamps.empty:
                    text += f"\n📅 **Time Span:** {timestamps.min()} to {timestamps.max()}\n"
            except Exception as e:
                logger.warning(f"Could not format large dataset timestamp: {e}")
       
        return text
# =============================================================================
# ENHANCED VISUALIZATION GENERATOR
# =============================================================================
# This class works on the resulting DataFrame.
# Since the queries are aliased (e.g., department_name AS area_name),
# it should not require any changes.
class EnhancedVisualizationGenerator:
    """Intelligent visualization generator that only shows meaningful charts"""
   
    @staticmethod
    def should_visualize(question: str, df: Any) -> bool:
        """Intelligent visualization decision based on user intent and data meaningfulness"""
       
        # First check: Do we have valid data?
        if not EnhancedVisualizationGenerator._has_meaningful_data(df):
            return False
       
        # Explicit visualization keywords - user specifically asked for charts
        explicit_viz_keywords = ['visualize', 'plot', 'chart', 'graph', 'show chart', 'pie chart', 'bar chart', 'trend chart']
        user_explicitly_requested = any(keyword in question.lower() for keyword in explicit_viz_keywords)
       
        if user_explicitly_requested:
            return True
       
        # Auto-visualization keywords - helpful for analysis but not explicitly requested
        analysis_keywords = ['trend', 'compare', 'over time', 'distribution', 'performance']
        would_benefit_from_viz = any(keyword in question.lower() for keyword in analysis_keywords)
       
        # Only auto-visualize if:
        # 1. User asked for analysis that benefits from visualization AND
        # 2. We have time series data with multiple points OR comparison data
        if would_benefit_from_viz:
            has_time_series = 'timestamp' in df.columns and len(df) > 2 # Need at least 3 points for meaningful trend
            # NOTE: Check for aliased 'area_name'
            has_comparison_data = any(col in df.columns for col in ['site_name', 'line_name', 'area_name', 'location']) and len(df) > 1
            return has_time_series or has_comparison_data
       
        return False
   
    @staticmethod
    def _has_meaningful_data(df: Any) -> bool:
        """Check if dataframe has meaningful data worth visualizing"""
       
        if df is None:
            return False
           
        if hasattr(df, 'empty') and df.empty:
            return False
           
        if len(df) == 0:
            return False
       
        # Check if we have at least one numeric column with non-null values
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        # Use new ID columns from new schema
        numeric_cols = [col for col in numeric_cols if col not in ['id', 'line_id', 'site_id', 'department_id']]
       
        # Also get categorical columns for later use
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col not in ['id', 'timestamp']]
       
        if len(numeric_cols) == 0 and len(categorical_cols) == 0:
            return False
       
        # Check if numeric columns have meaningful variance (not all zeros or nulls)
        for col in numeric_cols:
            non_null_values = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(non_null_values) > 0:
                # If we have values and they're not all the same, it's meaningful
                if non_null_values.nunique() > 1 or non_null_values.iloc[0] != 0:
                    return True
       
        # For categorical data, check if we have meaningful categories
        for col in categorical_cols:
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0 and non_null_values.nunique() > 1:
                return True
       
        return False
   
    @staticmethod
    def generate_chart(df: Any, question: str):
        """Generate enhanced visualizations with intelligent data validation"""
      
        try:
            # Double-check data meaningfulness before creating any chart
            if not EnhancedVisualizationGenerator._has_meaningful_data(df):
                logger.info("Skipping visualization - no meaningful data to display")
                return None
           
            # Set style
            plt.style.use('default')
           
            fig, ax = plt.subplots(figsize=(12, 8))
           
            # Detect pie chart request
            if 'pie' in question.lower():
                chart = EnhancedVisualizationGenerator._create_pie_chart(df, question, fig, ax)
           
            # Time series visualization
            elif 'timestamp' in df.columns and len(df) > 2: # Need at least 3 points for meaningful trend
                chart = EnhancedVisualizationGenerator._create_time_series_chart(df, question, fig, ax)
           
            # Comparison charts
            # NOTE: Checks for 'area_name' which is aliased
            elif any(col in df.columns for col in ['site_name', 'line_name', 'area_name', 'location']) and len(df) > 1:
                chart = EnhancedVisualizationGenerator._create_comparison_chart(df, question, fig, ax)
           
            # Distribution charts
            else:
                chart = EnhancedVisualizationGenerator._create_distribution_chart(df, question, fig, ax)
           
            # If chart creation returned None, clean up the figure
            if chart is None:
                plt.close(fig)
                return None
           
            return chart
               
        except Exception as e:
            logger.error(f"Visualization error: {e}")
            # Clean up any partial figure
            try:
                plt.close('all')
            except:
                pass
            return None
   
    @staticmethod
    def _create_pie_chart(df: Any, question: str, fig, ax):
        """Create pie chart for categorical data with validation"""
       
        # Find the best columns for pie chart
        # NOTE: Includes 'area_name' which is aliased
        categorical_cols = ['maintenance_status', 'rejection_reason', 'line_name',
                           'location', 'area_name', 'site_name', 'order_status', 'inspection_result']
       
        # Find which categorical column exists in df
        label_col = None
        for col in categorical_cols:
            if col in df.columns:
                # Check if this column has meaningful data
                non_null_count = df[col].count()
                unique_count = df[col].nunique()
                if non_null_count > 0 and unique_count > 1:
                    label_col = col
                    break
       
        # If no categorical column found, try to find any string column
        if not label_col:
            for col in df.columns:
                if df[col].dtype == 'object' and col not in ['timestamp', 'id']:
                    non_null_count = df[col].count()
                    unique_count = df[col].nunique()
                    if non_null_count > 0 and unique_count > 1:
                        label_col = col
                        break
       
        if not label_col:
            logger.info("No suitable categorical data for pie chart")
            return None
       
        # Find numeric column for values (or use count)
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        # NOTE: Updated ID list
        numeric_cols = [col for col in numeric_cols if col not in ['id', 'line_id', 'department_id', 'site_id']]
       
        pie_data = None
        if len(numeric_cols) > 0:
            # Use first numeric column for values
            value_col = numeric_cols[0]
           
            # Aggregate data
            try:
                if len(df) > 1:
                    pie_data = df.groupby(label_col)[value_col].sum()
                else:
                    pie_data = df.set_index(label_col)[value_col]
            except Exception as e:
                logger.warning(f"Could not group for pie chart: {e}")
                pie_data = df[label_col].value_counts()
        else:
            # Count occurrences
            pie_data = df[label_col].value_counts()
       
        # Validate pie data
        if pie_data is None or len(pie_data) == 0:
            logger.info("No data available for pie chart")
            return None
       
        # Remove zero values
        pie_data = pie_data[pie_data > 0]
        if len(pie_data) == 0:
            logger.info("All values are zero - no meaningful pie chart data")
            return None
       
        # Limit to top 10 categories for readability
        if len(pie_data) > 10:
            pie_data = pie_data.nlargest(10)
            title_suffix = " (Top 10)"
        else:
            title_suffix = ""
       
        try:
            # Create pie chart with enhanced styling
            colors = plt.cm.Set3(range(len(pie_data)))
            wedges, texts, autotexts = ax.pie(
                pie_data.values,
                labels=pie_data.index,
                autopct='%1.1f%%',
                colors=colors,
                startangle=90,
                textprops={'fontsize': 10}
            )
           
            # Enhance text readability
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
           
            # Add legend with values
            legend_labels = [f"{label}: {value:.1f}" for label, value in zip(pie_data.index, pie_data.values)]
            ax.legend(legend_labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
           
            title = question[:80] + '...' if len(question) > 80 else question
            ax.set_title(f"Distribution: {title}{title_suffix}", pad=20)
           
            plt.tight_layout()
            return fig
       
        except Exception as e:
            logger.error(f"Error creating pie chart: {e}")
            return None
   
    @staticmethod
    def _create_time_series_chart(df: Any, question: str, fig, ax):
        """Create enhanced time series chart with data validation"""
       
        # Validate timestamp data
        if 'timestamp' not in df.columns:
            logger.info("No timestamp column for time series chart")
            return None
       
        # Remove rows with null timestamps
        df_clean = df.dropna(subset=['timestamp'])
        df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'], errors='coerce')
        df_clean = df_clean.dropna(subset=['timestamp'])
        if len(df_clean) < 2:
            logger.info("Not enough timestamp data points for meaningful time series")
            return None
       
        # Check if we have any numeric data to plot
        numeric_cols = df_clean.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns
        # NOTE: Updated ID list
        meaningful_numeric_cols = [col for col in numeric_cols if col not in ['id', 'line_id', 'site_id', 'department_id']]
       
        has_meaningful_data = False
        for col in meaningful_numeric_cols:
            non_null_values = pd.to_numeric(df_clean[col], errors='coerce').dropna()
            if len(non_null_values) > 1 and (non_null_values.nunique() > 1 or non_null_values.iloc[0] != 0):
                has_meaningful_data = True
                break
       
        if not has_meaningful_data:
            logger.info("No meaningful numeric data for time series chart")
            return None
       
        try:
            df_sorted = df_clean.sort_values('timestamp')
           
            # Determine chart title based on content
            chart_title = question
            if len(chart_title) > 60:
                chart_title = chart_title[:60] + "..."
           
            # Plot OEE components if available
            if 'oee' in df.columns:
                # Plot main OEE metrics
                ax.plot(df_sorted['timestamp'], pd.to_numeric(df_sorted['oee'], errors='coerce'),
                       marker='o', linewidth=3, label='Overall Equipment Effectiveness (OEE)', color='#2E86AB', markersize=6)
               
                if 'availability' in df.columns:
                    ax.plot(df_sorted['timestamp'], pd.to_numeric(df_sorted['availability'], errors='coerce'),
                           marker='s', linewidth=2, alpha=0.8, label='Availability', color='#A23B72', markersize=5)
                   
                if 'performance' in df.columns:
                    ax.plot(df_sorted['timestamp'], pd.to_numeric(df_sorted['performance'], errors='coerce'),
                           marker='^', linewidth=2, alpha=0.8, label='Performance', color='#F18F01', markersize=5)
                   
                if 'quality' in df.columns:
                    ax.plot(df_sorted['timestamp'], pd.to_numeric(df_sorted['quality'], errors='coerce'),
                           marker='d', linewidth=2, alpha=0.8, label='Quality', color='#C73E1D', markersize=5)
               
                # Add performance benchmarks
                ax.axhline(y=85, color='green', linestyle='--', alpha=0.6, label='World Class (85%)', linewidth=1)
                ax.axhline(y=60, color='orange', linestyle='--', alpha=0.6, label='Good Performance (60%)', linewidth=1)
                ax.axhline(y=40, color='red', linestyle='--', alpha=0.6, label='Needs Improvement (40%)', linewidth=1)
               
                ax.set_ylabel('Performance Percentage (%)', fontsize=12, fontweight='bold')
                ax.set_ylim(0, 100)
               
                # Add percentage formatting to y-axis
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}%'))
               
            elif 'produced_quantity' in df.columns or 'remaining_quantity' in df.columns:
                # Production quantity chart
                if 'produced_quantity' in df.columns:
                    ax.plot(df_sorted['timestamp'], pd.to_numeric(df_sorted['produced_quantity'], errors='coerce'),
                           marker='o', linewidth=2, label='Produced Quantity', color='#2E86AB')
                if 'remaining_quantity' in df.columns:
                    ax.plot(df_sorted['timestamp'], pd.to_numeric(df_sorted['remaining_quantity'], errors='coerce'),
                           marker='s', linewidth=2, label='Remaining Quantity', color='#F18F01')
               
                ax.set_ylabel('Quantity (Units)', fontsize=12, fontweight='bold')
               
            elif 'rejection_quantity' in df.columns or 'accepted_quantity' in df.columns:
                # Quality metrics chart
                if 'rejection_quantity' in df.columns:
                    ax.plot(df_sorted['timestamp'], pd.to_numeric(df_sorted['rejection_quantity'], errors='coerce'),
                           marker='o', linewidth=2, label='Rejected Items', color='#C73E1D')
                if 'accepted_quantity' in df.columns:
                    ax.plot(df_sorted['timestamp'], pd.to_numeric(df_sorted['accepted_quantity'], errors='coerce'),
                           marker='s', linewidth=2, label='Accepted Items', color='#2E86AB')
               
                ax.set_ylabel('Quality Control (Items)', fontsize=12, fontweight='bold')
               
            else:
                # Generic numeric data
                numeric_cols = df_sorted.select_dtypes(include=['float64', 'int64']).columns
                # NOTE: Updated ID list
                numeric_cols = [col for col in numeric_cols if col not in ['id', 'line_id', 'department_id', 'site_id']]
               
                colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
                for i, col in enumerate(numeric_cols[:5]):
                    label = col.replace('_', ' ').title()
                    ax.plot(df_sorted['timestamp'], pd.to_numeric(df_sorted[col], errors='coerce'),
                           marker='o', linewidth=2, label=label,
                           color=colors[i % len(colors)], markersize=4)
               
                ax.set_ylabel('Value', fontsize=12, fontweight='bold')
           
            # Enhanced axis formatting
            ax.set_xlabel('Time Period', fontsize=12, fontweight='bold')
            ax.set_title(f'Manufacturing Performance Trends\n{chart_title}', fontsize=14, fontweight='bold', pad=20)
           
            # Improve legend
            ax.legend(loc='upper right', bbox_to_anchor=(1, 1), frameon=True, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
           
            # Enhanced x-axis date formatting
            import matplotlib.dates as mdates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
           
            # Auto-adjust locator
            num_ticks = min(len(df_sorted), 10)
            if num_ticks > 1:
                locator = plt.MaxNLocator(nbins=num_ticks)
                ax.xaxis.set_major_locator(locator)
           
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            plt.tight_layout()
            return fig
           
        except Exception as e:
            logger.error(f"Error creating time series chart: {e}")
            return None
   
    @staticmethod
    def _create_comparison_chart(df: Any, question: str, fig, ax):
        """Create enhanced comparison bar chart with data validation"""
       
        # Enhanced grouping column detection
        # NOTE: Includes 'area_name' which is aliased
        group_cols = ['site_name', 'line_name', 'area_name', 'factory_location', 'site', 'area', 'location']
        group_col = None
       
        for col in group_cols:
            if col in df.columns:
                # Check if column has meaningful data
                non_null_count = df[col].count()
                unique_count = df[col].nunique()
                if non_null_count > 0 and unique_count > 1:
                    group_col = col
                    break
       
        if not group_col:
            logger.info("No suitable grouping column for comparison chart")
            return None
       
        # Determine chart title and axis labels based on content
        chart_title = question if len(question) <= 60 else question[:60] + "..."
       
        # Enhanced value column selection and labeling
        if 'oee' in df.columns:
            value_col = 'oee'
            y_label = 'Overall Equipment Effectiveness (%)'
            # Performance-based color coding
            def get_color(val):
                if val >= 85: return '#2E8B57' # Sea Green - World Class
                elif val >= 60: return '#DAA520' # Golden Rod - Good
                elif val >= 40: return '#FF8C00' # Dark Orange - Needs Improvement
                else: return '#DC143C' # Crimson - Critical
           
        elif 'availability' in df.columns:
            value_col = 'availability'
            y_label = 'Equipment Availability (%)'
            get_color = lambda val: '#2E86AB' if val >= 85 else '#F18F01'
           
        elif 'quality' in df.columns:
            value_col = 'quality'
            y_label = 'Quality Rate (%)'
            get_color = lambda val: '#2E8B57' if val >= 95 else '#DC143C'
           
        elif 'performance' in df.columns:
            value_col = 'performance'
            y_label = 'Performance Efficiency (%)'
            get_color = lambda val: '#2E86AB'
           
        elif 'rejection_quantity' in df.columns:
            value_col = 'rejection_quantity'
            y_label = 'Rejected Items (Count)'
            get_color = lambda val: '#DC143C' if val > 10 else '#F18F01'
           
        elif 'produced_quantity' in df.columns:
            value_col = 'produced_quantity'
            y_label = 'Production Output (Units)'
            get_color = lambda val: '#2E86AB'
           
        else:
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            # NOTE: Updated ID list
            numeric_cols = [col for col in numeric_cols if col not in ['id', 'line_id', 'department_id', 'site_id']]
            value_col = numeric_cols[0] if numeric_cols else None
            y_label = value_col.replace('_', ' ').title() if value_col else 'Value'
            get_color = lambda val: '#2E86AB'
       
        # Validate value column exists and has meaningful data
        if not value_col or value_col not in df.columns:
            logger.info("No suitable numeric column for comparison chart")
            return None
       
        # Check if value column has meaningful data
        non_null_values = pd.to_numeric(df[value_col], errors='coerce').dropna()
        if len(non_null_values) == 0 or (non_null_values.nunique() <= 1 and non_null_values.iloc[0] == 0):
            logger.info("No meaningful numeric data for comparison chart")
            return None
       
        try:
            # Group and sort data
            if len(df) > 1:
                grouped = df.groupby(group_col)[value_col].mean().sort_values(ascending=False)
            else:
                grouped = df.set_index(group_col)[value_col]
           
            # Validate grouped data
            if len(grouped) == 0:
                logger.info("No data after grouping for comparison chart")
                return None
           
            # Create bars with performance-based colors
            colors = [get_color(val) for val in grouped.values]
            bars = ax.bar(grouped.index, grouped.values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
           
            # Add value labels on bars with better formatting
            for bar, value in zip(bars, grouped.values):
                height = bar.get_height()
                # Format based on value type
                if 'percentage' in y_label.lower() or '%' in y_label:
                    label_text = f'{value:.1f}%'
                elif 'count' in y_label.lower() or 'quantity' in y_label.lower():
                    label_text = f'{int(value)}'
                else:
                    label_text = f'{value:.1f}'
                   
                ax.text(bar.get_x() + bar.get_width()/2., height + max(grouped.values)*0.01,
                       label_text, ha='center', va='bottom', fontweight='bold', fontsize=10)
           
            # Enhanced axis formatting
            x_label = group_col.replace('_', ' ').replace('name', '').title().strip()
            ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
            ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
           
            # Add performance benchmark lines
            if value_col == 'oee':
                ax.axhline(y=85, color='green', linestyle='--', alpha=0.6, label='World Class (85%)', linewidth=2)
                ax.axhline(y=60, color='orange', linestyle='--', alpha=0.6, label='Good (60%)', linewidth=2)
                ax.axhline(y=40, color='red', linestyle='--', alpha=0.6, label='Needs Improvement (40%)', linewidth=2)
                ax.legend(loc='upper right', fontsize=10)
               
            elif 'percentage' in y_label.lower() or '%' in y_label:
                ax.set_ylim(0, 105)
                # Add percentage formatting to y-axis
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}%'))
           
            # Enhanced title and formatting
            ax.set_title(f'Manufacturing Performance Comparison\n{chart_title}',
                        fontsize=14, fontweight='bold', pad=20)
           
            # Improve x-axis label rotation
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
           
            plt.tight_layout()
            return fig
           
        except Exception as e:
            logger.error(f"Error creating comparison chart: {e}")
            return None
   
    @staticmethod
    def _create_distribution_chart(df: Any, question: str, fig, ax):
        """Create distribution chart with data validation"""
       
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        # NOTE: Updated ID list
        numeric_cols = [col for col in numeric_cols if col not in ['id', 'line_id', 'site_id', 'department_id']]
       
        if len(numeric_cols) == 0:
            logger.info("No numeric columns available for distribution chart")
            return None
       
        main_col = numeric_cols[0]
       
        # Validate main column has meaningful data
        non_null_values = pd.to_numeric(df[main_col], errors='coerce').dropna()
        if len(non_null_values) < 2:
            logger.info(f"Not enough data points in {main_col} for distribution chart")
            return None
       
        if non_null_values.nunique() <= 1:
            logger.info(f"No variance in {main_col} for meaningful distribution chart")
            return None
       
        try:
            # Histogram with appropriate number of bins
            num_bins = min(20, max(5, len(non_null_values)//3))
            non_null_values.hist(bins=num_bins, alpha=0.7, ax=ax, color='#2E86AB', edgecolor='black')
           
            ax.set_xlabel(main_col.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
           
            # Add mean line if meaningful
            mean_val = non_null_values.mean()
            if not pd.isna(mean_val):
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                          label=f'Mean: {mean_val:.1f}')
                ax.legend()
           
            # Add grid for better readability
            ax.grid(True, alpha=0.3, axis='y')
           
            title = question[:80] + '...' if len(question) > 80 else question
            ax.set_title(f'Data Distribution: {title}', fontsize=14, fontweight='bold', pad=20)
           
            plt.tight_layout()
            return fig
           
        except Exception as e:
            logger.error(f"Error creating distribution chart: {e}")
            return None
# =============================================================================
# MAIN STREAMLIT APPLICATION
# =============================================================================
def main():
    """Main Streamlit application with improvements"""
   
    st.set_page_config(
        page_title="Improved Manufacturing Chatbot",
        page_icon="🏭",
        layout="wide"
    )
   
    # Session tracking
    if 'session_id' not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())[:8]
        logger.info(f"New session: {st.session_state.session_id}")
   
    st.title("🏭 Factory Intelligence Agent")
    col1, col2 = st.columns([2, 20])
    with col1:
        st.image("AKH.png", width=90)
    with col2:
        st.markdown(
            "<h2 style='font-size:23px; transform: translateY(5px) translateX(0px); margin:0;'> Factory Copilot</h2>",
            unsafe_allow_html=True
            )
   
    # Initialize clients
    supabase, azure_client = init_clients()
   
    # Initialize improved components
    executor = QueryExecutor(supabase)
    translator = ImprovedNLToSQLTranslator(azure_client, executor)
    formatter = EnhancedResultFormatter(azure_client)
    viz_generator = EnhancedVisualizationGenerator()
   
    # Enhanced sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
       
        # Connection status
        st.subheader("🔗 Connection Status")
        st.success("✅ Supabase Connected")
        if azure_client:
            st.success("✅ Azure OpenAI Connected")
        else:
            st.warning("⚠️ Azure OpenAI Not Available")
       
        # Database schema
        with st.expander("📋 Database Schema (New)"):
            schema = executor.get_schema_info()
            for table, columns in schema.items():
                st.write(f"**{table}:**")
                st.caption(", ".join(columns))
       
        st.divider()
       
        # Filters
        st.subheader("🔍 Filters")
        filter_site = st.selectbox("Site", ["All", "Katunayake", "Biyagama"])
        filter_line = st.text_input("Line (e.g., Line1)")
       
        context = {}
        if filter_site != "All":
            context['site'] = filter_site
        if filter_line:
            context['line'] = filter_line
       
        st.divider()
       
        # Example questions
        st.subheader("💡 Example Questions")
       
        col1, col2 = st.columns(2)
       
        with col1:
            st.markdown("**📊 Data Queries:**")
            example_queries = [
                "Tell me about our factories",
                "Current OEE in Katunayake",
                "Show OEE trends chart last 24 hours",
                "Which machines need maintenance?",
                "Quality issues in Biyagama Press"
            ]
           
            for query in example_queries:
                if st.button(query, key=f"example_{query}"):
                    st.session_state.example_query = query
       
        with col2:
            st.markdown("**🤔 Ask Me About:**")
            st.markdown("""
            • What is OEE?
            • Explain availability
            • Define MTBF and MTTR
            • Performance metrics
            """)
       
        st.divider()
       
        # Clear history
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
   
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
   
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
           
            # Show data table
            if "dataframe" in message:
                with st.expander("📊 View Data Table"):
                    st.dataframe(message["dataframe"], use_container_width=True)
           
            # Show chart
            if "chart" in message:
                st.pyplot(message["chart"])
           
            # Show SQL
            if "sql" in message:
                with st.expander("🔍 Generated SQL"):
                    st.code(message["sql"], language="sql")
   
    # Handle example query selection
    if 'example_query' in st.session_state:
        prompt = st.session_state.example_query
        del st.session_state.example_query
    else:
        prompt = st.chat_input("Ask me about your manufacturing data...")
   
    # Process user input
    if prompt:
        logger.info(f"Processing query: {prompt}")
       
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
       
        with st.chat_message("user"):
            st.markdown(prompt)
       
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing your question..."):
                # Translate query
                logger.info(f"🔍 [DEBUG] Calling translator.translate() with prompt: '{prompt}'")
                translation = translator.translate(prompt, context)
                logger.info(f"🔍 [DEBUG] Translation result keys: {list(translation.keys())}")
                logger.info(f"🔍 [DEBUG] is_factory_overview: {translation.get('is_factory_overview', False)}")
                logger.info(f"🔍 [DEBUG] has SQL: {'sql' in translation}")
                logger.info(f"🔍 [DEBUG] has error: {'error' in translation}")
                logger.info(f"🔍 [DEBUG] is_conceptual: {translation.get('is_conceptual', False)}")
               
                # Handle conceptual questions
                if translation.get('is_conceptual'):
                    logger.info(f"🔍 [DEBUG] Handling conceptual question")
                    answer = translation.get('answer', 'No answer available.')
                    st.markdown(answer)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })
               
                # Handle errors
                elif 'error' in translation:
                    logger.info(f"🔍 [DEBUG] Handling translation error: {translation['error']}")
                    error_msg = f"❌ {translation['error']}"
                    st.error(translation['error'])
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
               
                # Handle SQL queries
                else:
                    logger.info(f"🔍 [DEBUG] Handling SQL query")
                    sql = translation['sql']
                    explanation = translation.get('explanation', '')
                    logger.info(f"🔍 [DEBUG] SQL length: {len(sql)} characters")
                    logger.info(f"🔍 [DEBUG] SQL preview: {sql[:200]}...")
                   
                    # Show generated SQL
                    with st.expander("🔍 Generated SQL"):
                        st.code(sql, language="sql")
                        st.caption(f"Query purpose: {explanation}")
                   
                    # Execute query
                    logger.info(f"🔍 [DEBUG] About to execute SQL query...")
                    with st.spinner("⚙️ Executing query..."):
                        df, error = executor.execute(sql)
                   
                    logger.info(f"🔍 [DEBUG] Query execution complete. Error: {error}")
                    logger.info(f"🔍 [DEBUG] DataFrame: {df is not None}, Rows: {len(df) if df is not None else 0}")
                   
                    if error:
                        logger.error(f"🔍 [DEBUG] Query execution failed: {error}")
                        st.error(f"Query execution failed: {error}")
                        # Append error to chat
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"❌ Query execution failed: {error}",
                            "sql": sql
                        })
                        st.stop() # Stop further processing for this turn
                   
                    if translation.get('is_factory_overview'):
                        logger.info(f"🔍 [DEBUG] Using factory overview formatting")
                        # Use special factory overview formatting
                        insights = formatter.format_factory_overview(df, executor)
                        logger.info(f"🔍 [DEBUG] Factory overview formatted: {len(insights)} characters")
                        st.markdown(insights)
                       
                        # Show data table
                        if df is not None and not df.empty:
                            with st.expander(f"📊 View Raw Data ({len(df)} rows)"):
                                st.dataframe(df, use_container_width=True)
                       
                        # Save to history
                        message_data = {
                            "role": "assistant",
                            "content": insights,
                            "sql": sql
                        }
                       
                        if df is not None and not df.empty:
                            message_data["dataframe"] = df
                       
                        st.session_state.messages.append(message_data)
                    else:
                       
                        # Format results with insights
                        insights = formatter.format_to_text(df, prompt, explanation)
                        st.markdown(insights)
                       
                        # Show data table
                        if df is not None and not df.empty:
                            with st.expander(f"📊 View Data Table ({len(df)} rows)"):
                                st.dataframe(df, use_container_width=True)
                       
                        # Generate visualization intelligently
                        chart = None
                       
                        # Check if user explicitly requested visualization
                        explicit_viz_keywords = ['visualize', 'plot', 'chart', 'graph', 'show chart']
                        user_wants_viz = any(keyword in prompt.lower() for keyword in explicit_viz_keywords)
                       
                        # Only attempt visualization if conditions are met
                        if viz_generator.should_visualize(prompt, df):
                            with st.spinner("📈 Creating visualization..."):
                                chart = viz_generator.generate_chart(df, prompt)
                               
                                if chart:
                                    st.pyplot(chart)
                                elif user_wants_viz:
                                    # User explicitly asked for a chart but we couldn't create one
                                    st.info("📈 Unable to create visualization - the data doesn't contain enough meaningful information for a chart. The text analysis above provides the insights available from this data.")
                       
                        # Save to history
                        message_data = {
                            "role": "assistant",
                            "content": insights,
                            "sql": sql
                        }
                       
                        if df is not None and not df.empty:
                            message_data["dataframe"] = df
                       
                        if chart:
                            message_data["chart"] = chart
                       
                        st.session_state.messages.append(message_data)
if __name__ == "__main__":
    main()
