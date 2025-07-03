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
                model="gemini-2.5-flash",
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
        (
            self.analyze_tool,
            self.sql_tool,
            self.bar_chart_tool,
            self.line_chart_tool,
            self.scatter_plot_tool,
            self.histogram_tool,
            self.monthly_trends_tool
        ) = create_supply_chain_tools(
            db=self.db,
            llm=self.llm,
            memory=self.memory
        )
    
    def _setup_agent(self):
        """Setup the LangGraph agent with proper memory management"""
        tools = [
            self.analyze_tool,
            self.sql_tool,
            self.bar_chart_tool,
            self.line_chart_tool,
            self.scatter_plot_tool,
            self.histogram_tool,
            self.monthly_trends_tool
        ]
        
        agent_node = create_react_agent(
            model=self.llm,
            tools=tools
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
            # Clear any previous query cache to avoid stale data
            self.db.clear_query_cache()
            
            # Create thread configuration for session persistence
            config = {"configurable": {"thread_id": thread_id}}
            
            # Let the LLM agent decide which tools to use based on the query
            messages = [HumanMessage(content=query)]
            result = self.app.invoke({"messages": messages}, config)
            
            processing_time = time.time() - start_time
            
            # Extract final response text
            final_message = result["messages"][-1]
            response_text = final_message.content if hasattr(final_message, 'content') else str(final_message)
            
            # Check for tool calls in the conversation to extract charts and SQL data
            chart_data = None
            sql_query = "No SQL query captured"
            dataframe = None
            
            # Debug: Log the full result structure
            log_agent_response(f"langgraph_messages_count_{len(result['messages'])}", 0, 0)
            
            # Find the index of the current user message (the one we just sent)
            user_message_index = -1
            for i, msg in enumerate(result["messages"]):
                if hasattr(msg, 'content') and msg.content == query:
                    user_message_index = i
                    break
            
            # Only look at messages AFTER the current user input
            if user_message_index >= 0:
                current_request_messages = result["messages"][user_message_index + 1:]
            else:
                # Fallback: just look at the last few messages
                current_request_messages = result["messages"][-4:] if len(result["messages"]) > 4 else result["messages"]
            
            log_agent_response(f"processing_messages_from_index_{user_message_index}", len(current_request_messages), 0)
            
            for i, msg in enumerate(current_request_messages):
                # Debug logging
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        log_agent_response(f"found_tool_call_{tc['name']}", 0, 0)
                
                # Check for analyze_supply_chain_data tool calls
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        if tool_call['name'] == 'analyze_supply_chain_data':
                            # Find the corresponding tool response
                            if i + 1 < len(current_request_messages):
                                next_msg = current_request_messages[i + 1]
                                if hasattr(next_msg, 'content'):
                                    try:
                                        # The content might be a string representation of the dict
                                        import ast
                                        if isinstance(next_msg.content, str) and next_msg.content.startswith('{'):
                                            tool_result = ast.literal_eval(next_msg.content)
                                        else:
                                            tool_result = next_msg.content
                                        
                                        if isinstance(tool_result, dict):
                                            sql_query = tool_result.get('sql_query', 'No SQL query captured')
                                            dataframe = tool_result.get('dataframe')
                                    except:
                                        # If parsing fails, try to get from database cache
                                        query_info = self.db.get_last_query_info()
                                        sql_query = query_info.get("query", "No SQL query captured")
                                        if sql_query and sql_query != "No SQL query captured":
                                            try:
                                                import pandas as pd
                                                dataframe = pd.read_sql_query(sql_query, self.db._engine)
                                            except:
                                                pass
                        
                        # Check for chart tool calls
                        elif tool_call['name'] in ['create_bar_chart', 'create_line_chart', 'create_scatter_plot', 'create_histogram', 'plot_monthly_transaction_trends']:
                            # Find the corresponding tool response
                            if i + 1 < len(current_request_messages):
                                next_msg = current_request_messages[i + 1]
                                if hasattr(next_msg, 'content'):
                                    log_agent_response(f"chart_tool_response_type_{type(next_msg.content)}", 0, 0)
                                    try:
                                        import ast
                                        if isinstance(next_msg.content, str) and next_msg.content.startswith('{'):
                                            tool_result = ast.literal_eval(next_msg.content)
                                            log_agent_response(f"parsed_tool_result_type_{tool_result.get('type')}", 0, 0)
                                            if isinstance(tool_result, dict) and tool_result.get('type') == 'plotly':
                                                # Convert JSON back to Plotly figure
                                                try:
                                                    import plotly.io as pio
                                                    chart_json = tool_result.get('chart')
                                                    if isinstance(chart_json, str):
                                                        chart_fig = pio.from_json(chart_json)
                                                        tool_result['chart'] = chart_fig
                                                    chart_data = tool_result
                                                except Exception as chart_error:
                                                    log_error(chart_error, {"context": "chart_reconstruction"})
                                                    chart_data = tool_result  # Use as-is if conversion fails
                                    except:
                                        pass
            
            # Store in memory
            self.memory.save_context(
                {"input": query},
                {"output": response_text}
            )
            
            # Check if this was a chart request by looking for chart-related keywords and tool calls
            chart_keywords = ['plot', 'chart', 'graph', 'trend', 'visuali']
            is_chart_request = any(keyword in query.lower() for keyword in chart_keywords)
            
            # If it's a chart request but we didn't extract chart data, try calling the tool directly
            # BUT only if we don't already have SQL/dataframe from analyze_supply_chain_data
            if is_chart_request and not chart_data and sql_query == "No SQL query captured":
                log_agent_response("attempting_direct_chart_call", 0, 0)
                if 'monthly' in query.lower() and ('trend' in query.lower() or 'transaction' in query.lower()):
                    try:
                        direct_chart = self.monthly_trends_tool.invoke({})
                        if direct_chart.get('type') == 'plotly':
                            # Convert JSON to Plotly figure
                            import plotly.io as pio
                            chart_json = direct_chart.get('chart')
                            if isinstance(chart_json, str):
                                chart_fig = pio.from_json(chart_json)
                                chart_data = {
                                    "type": "plotly",
                                    "chart": chart_fig,
                                    "description": direct_chart.get("description", "Monthly trends chart")
                                }
                                
                                # Also get the SQL query and dataframe from the tool execution
                                query_info = self.db.get_last_query_info()
                                if query_info.get("query"):
                                    sql_query = query_info["query"]
                                    try:
                                        import pandas as pd
                                        dataframe = pd.read_sql_query(sql_query, self.db._engine)
                                    except Exception as df_error:
                                        log_error(df_error, {"context": "direct_chart_dataframe"})
                                
                                log_agent_response("direct_chart_success", 0, 0)
                    except Exception as e:
                        log_error(e, {"context": "direct_chart_call"})
            
            # Return appropriate format based on what was found
            if chart_data:
                log_agent_response("text_with_chart", len(response_text), processing_time)
                return {
                    "type": "text_with_chart",
                    "text": response_text,
                    "chart": chart_data.get("chart"),
                    "sql_query": sql_query,
                    "dataframe": dataframe
                }
            elif dataframe is not None or sql_query != "No SQL query captured":
                log_agent_response("text_with_sql_and_dataframe", len(response_text), processing_time)
                return {
                    "type": "text_with_sql_and_dataframe",
                    "text": response_text,
                    "sql_query": sql_query,
                    "dataframe": dataframe
                }
            else:
                log_agent_response("text", len(response_text), processing_time)
                return {
                    "type": "text",
                    "content": response_text
                }
                
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