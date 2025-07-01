import logging
import sys
import os
from datetime import datetime
from typing import Dict, Any
import json

class SupplyChainLogger:
    def __init__(self, log_level: str = "INFO"):
        self.setup_logging(log_level)
        
    def setup_logging(self, log_level: str):
        """Setup comprehensive logging configuration"""
        
        # Create logs directory if it doesn't exist
        if not os.path.exists("logs"):
            os.makedirs("logs")
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            handlers=[
                logging.FileHandler(f"logs/supply_chain_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Create specialized loggers
        self.agent_logger = logging.getLogger("agent")
        self.tool_logger = logging.getLogger("tools")
        self.streamlit_logger = logging.getLogger("streamlit")
        self.error_logger = logging.getLogger("errors")
        
        # Create separate error log file
        error_handler = logging.FileHandler(f"logs/errors_{datetime.now().strftime('%Y%m%d')}.log")
        error_handler.setLevel(logging.ERROR)
        error_formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s | %(exc_info)s"
        )
        error_handler.setFormatter(error_formatter)
        self.error_logger.addHandler(error_handler)
        
        # Create tool execution log
        tool_handler = logging.FileHandler(f"logs/tools_{datetime.now().strftime('%Y%m%d')}.log")
        tool_formatter = logging.Formatter(
            "%(asctime)s | TOOL | %(levelname)s | %(message)s"
        )
        tool_handler.setFormatter(tool_formatter)
        self.tool_logger.addHandler(tool_handler)
    
    def log_query(self, query: str, user_id: str = "default"):
        """Log incoming user query"""
        self.agent_logger.info(f"USER_QUERY | {user_id} | {query}")
    
    def log_tool_call(self, tool_name: str, input_data: str, execution_time: float = None):
        """Log tool execution"""
        log_data = {
            "tool": tool_name,
            "input": input_data[:200] + "..." if len(input_data) > 200 else input_data,
            "execution_time": execution_time
        }
        self.tool_logger.info(f"TOOL_CALL | {json.dumps(log_data)}")
    
    def log_tool_result(self, tool_name: str, result_type: str, success: bool, error: str = None):
        """Log tool execution result"""
        log_data = {
            "tool": tool_name,
            "result_type": result_type,
            "success": success,
            "error": error
        }
        self.tool_logger.info(f"TOOL_RESULT | {json.dumps(log_data)}")
    
    def log_agent_response(self, response_type: str, response_length: int, processing_time: float = None):
        """Log agent response"""
        log_data = {
            "response_type": response_type,
            "response_length": response_length,
            "processing_time": processing_time
        }
        self.agent_logger.info(f"AGENT_RESPONSE | {json.dumps(log_data)}")
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log errors with context"""
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        }
        self.error_logger.error(f"ERROR | {json.dumps(error_data)}", exc_info=error)
    
    def log_dataframe_operation(self, operation: str, dataframe_name: str, result_shape: tuple = None):
        """Log pandas dataframe operations"""
        log_data = {
            "operation": operation,
            "dataframe": dataframe_name,
            "result_shape": result_shape
        }
        self.tool_logger.info(f"DATAFRAME_OP | {json.dumps(log_data)}")
    
    def log_visualization(self, chart_type: str, data_points: int, success: bool):
        """Log visualization generation"""
        log_data = {
            "chart_type": chart_type,
            "data_points": data_points,
            "success": success
        }
        self.tool_logger.info(f"VISUALIZATION | {json.dumps(log_data)}")
    
    def log_streamlit_event(self, event: str, details: Dict[str, Any] = None):
        """Log Streamlit UI events"""
        log_data = {
            "event": event,
            "details": details or {}
        }
        self.streamlit_logger.info(f"UI_EVENT | {json.dumps(log_data)}")

# Global logger instance
supply_chain_logger = SupplyChainLogger()

# Convenience functions
def log_query(query: str, user_id: str = "default"):
    supply_chain_logger.log_query(query, user_id)

def log_tool_call(tool_name: str, input_data: str, execution_time: float = None):
    supply_chain_logger.log_tool_call(tool_name, input_data, execution_time)

def log_tool_result(tool_name: str, result_type: str, success: bool, error: str = None):
    supply_chain_logger.log_tool_result(tool_name, result_type, success, error)

def log_agent_response(response_type: str, response_length: int, processing_time: float = None):
    supply_chain_logger.log_agent_response(response_type, response_length, processing_time)

def log_error(error: Exception, context: Dict[str, Any] = None):
    supply_chain_logger.log_error(error, context)

def log_dataframe_operation(operation: str, dataframe_name: str, result_shape: tuple = None):
    supply_chain_logger.log_dataframe_operation(operation, dataframe_name, result_shape)

def log_visualization(chart_type: str, data_points: int, success: bool):
    supply_chain_logger.log_visualization(chart_type, data_points, success)

def log_streamlit_event(event: str, details: Dict[str, Any] = None):
    supply_chain_logger.log_streamlit_event(event, details)