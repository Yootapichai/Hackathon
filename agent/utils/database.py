"""
Database Connection Module

This module contains the custom database wrapper for supply chain analytics.
Separated from the main agent for better code organization and maintainability.
"""

import os
from langchain_community.utilities import SQLDatabase
from .logger import log_error, log_agent_response, log_dataframe_operation


class QueryCapturingSQLDatabase(SQLDatabase):
    """Custom SQLDatabase wrapper that captures executed queries"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_executed_query = None
        self.last_query_result = None
    
    def run(self, command: str, fetch: str = "all", **kwargs) -> str:
        """Override run method to capture queries"""
        try:
            # Store the query being executed
            self.last_executed_query = command.strip()
            
            # Execute the query using parent method with all parameters
            result = super().run(command, fetch, **kwargs)
            
            # Store the result
            self.last_query_result = result
            
            return result
        except Exception as e:
            # Reset on error
            self.last_executed_query = None
            self.last_query_result = None
            raise e
    
    def get_last_query_info(self):
        """Get the last executed query and its results"""
        return {
            "query": self.last_executed_query,
            "result": self.last_query_result
        }
    
    def clear_query_cache(self):
        """Clear stored query information"""
        self.last_executed_query = None
        self.last_query_result = None


def create_database_connection():
    """
    Create connection to Supabase PostgreSQL database with multiple fallback options
    
    Returns:
        QueryCapturingSQLDatabase: Configured database connection
        
    Raises:
        ValueError: If environment variables are missing
        Exception: If database connection fails
    """
    try:
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_password = os.environ.get("SUPABASE_DB_PASSWORD")
        
        if not supabase_url or not supabase_password:
            raise ValueError("SUPABASE_URL and SUPABASE_DB_PASSWORD environment variables are required")
        
        # Extract project ID from Supabase URL
        project_id = supabase_url.split("//")[1].split(".")[0]
        
        # Try multiple connection patterns with updated hostname patterns
        # Based on current Supabase documentation (2024/2025)
        connection_patterns = [
            # New pooler patterns (most likely to work)
            f"postgresql://postgres.{project_id}:{supabase_password}@aws-0-{project_id}.pooler.supabase.com:5432/postgres",
            f"postgresql://postgres.{project_id}:{supabase_password}@aws-0-{project_id}.pooler.supabase.com:6543/postgres",
            
            # Alternative new patterns
            f"postgresql://postgres:{supabase_password}@{project_id}.supabase.co:5432/postgres",
            f"postgresql://postgres:{supabase_password}@{project_id}.supabase.co:6543/postgres",
            
            # Legacy patterns (for older projects)
            f"postgresql://postgres:{supabase_password}@db.{project_id}.supabase.co:5432/postgres",
            f"postgresql://postgres:{supabase_password}@aws-0-{project_id}.pooler.supabase.com:5432/postgres",
            f"postgresql://postgres:{supabase_password}@aws-0-{project_id}.pooler.supabase.com:6543/postgres",
            
            # With SSL mode specified
            f"postgresql://postgres.{project_id}:{supabase_password}@aws-0-{project_id}.pooler.supabase.com:5432/postgres?sslmode=require",
            f"postgresql://postgres:{supabase_password}@{project_id}.supabase.co:5432/postgres?sslmode=require",
            
            # Direct connection with SSL
            f"postgresql://postgres:{supabase_password}@db.{project_id}.supabase.co:5432/postgres?sslmode=require"
        ]
        
        last_error = None
        
        for i, db_uri in enumerate(connection_patterns):
            try:
                # Extract hostname for display (hide password)
                display_uri = db_uri.replace(supabase_password, "***")
                print(f"Attempting connection pattern {i+1}: {display_uri}")
                
                db = QueryCapturingSQLDatabase.from_uri(db_uri)
                
                # Test the connection by getting table names
                table_names = db.get_usable_table_names()
                print(f"✓ Connection successful! Found {len(table_names)} tables")
                
                # Log successful connection
                try:
                    log_agent_response("sql_connection_initialized", len(table_names), 0)
                    for table in table_names:
                        log_dataframe_operation("sql_table_available", table, (0, 0))
                except Exception as e:
                    log_error(e, {"component": "database_logging"})
                
                return db
                
            except Exception as e:
                last_error = e
                print(f"✗ Connection pattern {i+1} failed: {str(e)}")
                continue
        
        # If all patterns failed, raise the last error with helpful message
        error_msg = f"Failed to connect to Supabase database after trying {len(connection_patterns)} connection patterns.\n"
        error_msg += f"Last error: {str(last_error)}\n\n"
        error_msg += "Common solutions:\n"
        error_msg += "1. Check if your Supabase project is active (visit https://supabase.com/dashboard)\n"
        error_msg += "2. Verify the database password is correct\n"
        error_msg += "3. Check if your IP is allowed in Supabase dashboard > Settings > Database > Network restrictions\n"
        error_msg += "4. Try pausing and resuming your Supabase project\n"
        error_msg += "5. Check if direct database access is enabled in your project settings\n"
        error_msg += f"6. Your project ID is: {project_id}\n"
        error_msg += "7. Verify your internet connection is stable"
        
        raise Exception(error_msg)
        
    except Exception as e:
        log_error(e, {"component": "database_connection"})
        raise