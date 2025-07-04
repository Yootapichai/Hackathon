"""
Database Connection Module

This module contains the custom database wrapper for supply chain analytics.
Separated from the main agent for better code organization and maintainability.
"""

import os
from langchain_community.utilities import SQLDatabase
from loguru import logger


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
    Create connection to Supabase PostgreSQL database
    
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
        
        # PostgreSQL connection string for Supabase
        db_uri = f"postgresql://postgres:{supabase_password}@db.{project_id}.supabase.co:5432/postgres"
        
        db = QueryCapturingSQLDatabase.from_uri(db_uri)
        
        # Log successful connection
        try:
            table_names = db.get_usable_table_names()
            logger.info(f"SQL connection initialized with {len(table_names)} tables")
            for table in table_names:
                logger.debug(f"SQL table available: {table}")
        except Exception as e:
            logger.error(f"Database logging error: {e}")
        
        return db
        
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise