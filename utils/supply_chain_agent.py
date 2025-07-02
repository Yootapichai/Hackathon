"""
Supply Chain Agent - Refactored Version

This module contains the main SupplyChainAgent class with separated concerns:
- Database connection logic moved to database.py
- Tool definitions moved to tools.py
- Cleaner, more maintainable code structure
"""

import os
import time
from typing import Dict, Any, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing import TypedDict, Annotated
from dotenv import load_dotenv

# Import custom modules
from .database import create_database_connection
from .tools import create_supply_chain_tools
from .logger import (
    log_query, log_agent_response, log_error
)

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


class SupplyChainAgent:
    """
    Main Supply Chain Analytics Agent
    
    Provides natural language interface to supply chain database with:
    - SQL-powered data analysis
    - Interactive visualizations
    - Conversation memory
    - Session persistence
    """
    
    def __init__(self):
        try:
            log_agent_response("agent_init", 0, 0)
            
            # Validate API key
            self.google_api_key = os.environ.get('GOOGLE_API_KEY')
            if not self.google_api_key:
                error_msg = "GOOGLE_API_KEY environment variable is required"
                log_error(ValueError(error_msg), {"component": "agent_init"})
                raise ValueError(error_msg)
            
            # Initialize LLM
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=self.google_api_key,
                temperature=0.0,
                verbose=True
            )
            
            # Initialize database connection
            self.db = create_database_connection()
            
            # Initialize LangChain memory for conversation context
            self.memory = ConversationBufferWindowMemory(
                k=5,  # Keep last 5 conversation exchanges
                memory_key="chat_history",
                return_messages=True
            )
            
            # Initialize LangGraph memory saver with SQLite persistence
            try:
                import sqlite3
                conn = sqlite3.connect("conversations.db", check_same_thread=False)
                self.langgraph_memory = SqliteSaver(conn)
                log_agent_response("sqlite_memory_initialized", 0, 0)
            except Exception as e:
                # Fallback to in-memory storage
                log_error(e, {"fallback": "using in-memory storage"})
                self.langgraph_memory = MemorySaver()
            
            # Setup tools and agent
            self._setup_tools()
            self._setup_agent()
            
            log_agent_response("agent_init_complete", 0, 0)
            
        except Exception as e:
            log_error(e, {"component": "agent_init"})
            raise
    
    def _setup_tools(self):
        """Setup tools using the tools module"""
        self.analyze_tool, self.visualize_tool = create_supply_chain_tools(
            db=self.db,
            llm=self.llm,
            memory=self.memory
        )
    
    def _setup_agent(self):
        """Setup the LangGraph agent with proper memory management"""
        agent_node = create_react_agent(
            model=self.llm,
            tools=[self.analyze_tool, self.visualize_tool]
        )
        
        graph = StateGraph(AgentState)
        graph.add_node("agent", agent_node)
        graph.set_entry_point("agent")
        graph.add_edge("agent", END)
        
        # Compile with MemorySaver for persistence
        self.app = graph.compile(checkpointer=self.langgraph_memory)
    
    def process_query(self, query: str, thread_id: str = "default") -> Dict[str, Any]:
        """
        Process user query with proper memory management
        
        Args:
            query: User's natural language query
            thread_id: Session identifier for conversation persistence
            
        Returns:
            Dict with response type, content, and optional chart/SQL data
        """
        start_time = time.time()
        log_query(query)
        
        try:
            # Create thread configuration for session persistence
            config = {"configurable": {"thread_id": thread_id}}
            
            # Determine if this is a visualization request
            visualization_keywords = ['plot', 'chart', 'graph', 'show', 'visualize', 'display']
            is_viz_request = any(keyword in query.lower() for keyword in visualization_keywords)
            
            if is_viz_request:
                # Handle visualization requests directly
                viz_result = self.visualize_tool.invoke(query)
                if viz_result["type"] == "plotly":
                    response = {
                        "type": "text_with_chart",
                        "text": f"Here's a visualization based on your request: {viz_result['description']}",
                        "chart": viz_result["chart"]
                    }
                    processing_time = time.time() - start_time
                    log_agent_response("text_with_chart", len(response["text"]), processing_time)
                    
                    # Store in LangChain memory for context
                    self.memory.save_context(
                        {"input": query},
                        {"output": response["text"]}
                    )
                    
                    return response
                else:
                    error_msg = viz_result.get("message", "Error creating visualization")
                    processing_time = time.time() - start_time
                    log_agent_response("error", len(error_msg), processing_time)
                    return {"type": "error", "content": error_msg}
            
            # For data analysis questions and conversation memory
            else:
                # Use enhanced SQL tool that captures query and DataFrame
                analysis_result = self.analyze_tool.invoke(query)
                
                processing_time = time.time() - start_time
                
                # Handle the enhanced response with SQL and DataFrame
                if isinstance(analysis_result, dict) and analysis_result.get("type") == "text_with_sql_and_dataframe":
                    log_agent_response("text_with_sql_and_dataframe", len(analysis_result["text"]), processing_time)
                    
                    # Store in LangChain memory for context
                    self.memory.save_context(
                        {"input": query},
                        {"output": analysis_result["text"]}
                    )
                    
                    return analysis_result
                    
                # Handle error responses
                elif isinstance(analysis_result, dict) and analysis_result.get("type") == "text":
                    log_agent_response("error", len(analysis_result["content"]), processing_time)
                    return {"type": "error", "content": analysis_result["content"]}
                    
                # Fallback for unexpected response format
                else:
                    # If tool returns just text (old format), wrap it
                    response_text = str(analysis_result)
                    log_agent_response("text", len(response_text), processing_time)
                    
                    self.memory.save_context(
                        {"input": query},
                        {"output": response_text}
                    )
                    
                    return {"type": "text", "content": response_text}
                
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error processing query: {str(e)}"
            log_error(e, {"query": query, "processing_time": processing_time})
            log_agent_response("error", len(error_msg), processing_time)
            return {"type": "error", "content": error_msg}
    
    def get_conversation_history(self, thread_id: str = "default") -> List[BaseMessage]:
        """Get conversation history for a specific thread"""
        try:
            config = {"configurable": {"thread_id": thread_id}}
            state = self.app.get_state(config)
            return state.values.get("messages", [])
        except Exception as e:
            log_error(e, {"method": "get_conversation_history", "thread_id": thread_id})
            return []
    
    def clear_memory(self, thread_id: str = "default"):
        """Clear conversation memory for a specific thread"""
        try:
            # Clear LangChain memory
            self.memory.clear()
            
            # Clear LangGraph state (note: MemorySaver doesn't have direct clear method)
            # Memory will naturally expire or can be handled at the application level
            log_agent_response("memory_cleared", 0, 0)
            
        except Exception as e:
            log_error(e, {"method": "clear_memory", "thread_id": thread_id})