"""
Supply Chain Agent - Refactored Version

This module contains the main SupplyChainAgent class with separated concerns:
- Database connection logic moved to database.py
- Tool definitions moved to tools.py
- Cleaner, more maintainable code structure
"""
import os
import time
import sqlite3
import pandas as pd
import plotly.express as px
from typing import Dict, Any, List, TypedDict, Annotated

from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

# Import custom modules
from .database import QueryCapturingSQLDatabase, create_database_connection
from .tools import create_supply_chain_tools
from .logger import (
    log_query, log_agent_response, log_error
)
from loguru import logger

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
            logger.info("Agent initialization started")
            
            # Validate API key
            self.google_api_key = os.environ.get('GOOGLE_API_KEY')
            if not self.google_api_key:
                error_msg = "GOOGLE_API_KEY environment variable is required"
                logger.error(f"Agent initialization failed: {error_msg}")
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
            
            # Log database connection info
            try:
                table_names = self.db.get_usable_table_names()
                logger.info(f"SQL connection initialized with {len(table_names)} tables")
                for table in table_names:
                    logger.debug(f"SQL table available: {table}")
            except Exception as e:
                logger.error(f"Database logging error: {e}")
            
            # Initialize LangChain memory for pandas agent
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
                logger.info("SQLite memory initialized")
            except Exception as e:
                # Fallback to in-memory storage
                logger.error(f"SQLite memory initialization failed, using in-memory storage: {e}")
                self.langgraph_memory = MemorySaver()
            
            # Initialize chart data memory for follow-up questions
            self.chart_memory = {}  # Store recent chart data for follow-ups
            self.max_stored_charts = 3  # Keep last 3 charts
            
            # Setup tools and agent
            self._setup_tools()
            self._setup_agent()
            
            logger.info("Agent initialization complete")
            
        except Exception as e:
            logger.error(f"Agent initialization failed: {e}")
            raise
    
    def store_chart_data(self, chart_type: str, dataframe, sql_query: str, description: str = ""):
        """
        Store chart data for follow-up questions
        
        Args:
            chart_type: Type of chart (e.g., 'monthly_trends', 'bar_chart')
            dataframe: The pandas DataFrame used to create the chart
            sql_query: The SQL query used to generate the data
            description: Optional description of the chart
        """
        try:
            import time
            
            chart_id = f"{chart_type}_{int(time.time())}"
            
            chart_data = {
                "id": chart_id,
                "type": chart_type,
                "dataframe": dataframe,
                "sql_query": sql_query,
                "description": description,
                "timestamp": time.time(),
                "row_count": len(dataframe) if dataframe is not None else 0,
                "columns": list(dataframe.columns) if dataframe is not None else []
            }
            
            # Store the chart data
            self.chart_memory[chart_id] = chart_data
            
            # Keep only the most recent charts
            if len(self.chart_memory) > self.max_stored_charts:
                # Remove oldest chart
                oldest_id = min(self.chart_memory.keys(), 
                              key=lambda k: self.chart_memory[k]["timestamp"])
                del self.chart_memory[oldest_id]
            
            log_agent_response(f"stored_chart_data_{chart_type}", len(dataframe) if dataframe is not None else 0, 0)
        except Exception as e:
            log_error(e, {"context": "store_chart_data", "chart_type": chart_type})
            return chart_id
    def _create_database_connection(self):
        """Create connection to Supabase PostgreSQL database"""
        try:
            supabase_url = os.environ.get("SUPABASE_URL")
            supabase_password = os.environ.get("SUPABASE_DB_PASSWORD")
            
            if not supabase_url or not supabase_password:
                raise ValueError("SUPABASE_URL and SUPABASE_DB_PASSWORD environment variables are required")
            
            # Extract project ID from Supabase URL
            project_id = supabase_url.split("//")[1].split(".")[0]
            
            # PostgreSQL connection string for Supabase
            db_uri = f"postgresql://postgres:{supabase_password}@db.{project_id}.supabase.co:5432/postgres"
            
            return QueryCapturingSQLDatabase.from_uri(db_uri)
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def _setup_tools(self):
        @tool
        def analyze_supply_chain_data(query: str) -> Dict[str, Any]:
            """Analyze supply chain data using SQL queries. Returns SQL, DataFrame, and natural language answer."""
            
            start_time = time.time()
            logger.info(f"Starting supply chain data analysis: {query[:100]}...")
            
            try:
                # Clear previous query cache
                self.db.clear_query_cache()
                
                # Enhanced system prompt for SQL agent with business context
                system_prefix = """
You are an expert supply chain data analyst with access to a PostgreSQL database containing:

Tables:
- material_master: Material information (material_name, polymer_type, shelf_life_in_month, downgrade_value_lost_percent)
- inventory: Current stock levels (balance_as_of_date, plant_name, material_name, batch_number, unrestricted_stock, stock_unit, stock_sell_value, currency)
- inbound: Incoming shipments (inbound_date, plant_name, material_name, net_quantity_mt)
- outbound: Outgoing shipments (outbound_date, plant_name, material_name, customer_number, mode_of_transport, net_quantity_mt)
- operation_costs: Storage and transfer costs (operation_category, cost_type, entity_name, entity_type, cost_amount, cost_unit, container_capacity_mt, currency)

Key Business Rules:
- Stock quantities are in KG for inventory, MT for inbound/outbound
- Different plants use different currencies (CNY for China, SGD for Singapore)
- Batch numbers track specific material lots
- Materials have shelf life and degradation rates

Always provide actionable insights and consider business context in your analysis.
Your answer must end with smile emoji
"""
                
                # Create SQL toolkit
                toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
                
                # Create SQL agent with proper memory integration
                try:
                    agent = create_sql_agent(
                        llm=self.llm,
                        toolkit=toolkit,
                        verbose=True,
                        agent_type="openai-tools",  # Changed from zero-shot-react-description
                        prefix=system_prefix,  # Use prefix parameter for openai-tools agent
                        handle_parsing_errors=True,
                        memory=self.memory  # Use proper LangChain memory
                    )
                except Exception as e:
                    logger.error(f"SQL agent creation with memory failed, trying without memory: {e}")
                    # Fallback without memory if it causes issues
                    agent = create_sql_agent(
                        llm=self.llm,
                        toolkit=toolkit,
                        verbose=True,
                        agent_type="openai-tools",  # Changed from zero-shot-react-description
                        prefix=system_prefix,  # Use prefix parameter for openai-tools agent
                        handle_parsing_errors=True
                    )
                
                # Use memory-aware invocation
                result = agent.invoke({"input": query})["output"]
                
                # Get the captured SQL query and create DataFrame
                query_info = self.db.get_last_query_info()
                sql_query = query_info.get("query", "No SQL query captured")
                
                # Create DataFrame from the executed query if available
                dataframe = None
                if sql_query and sql_query != "No SQL query captured":
                    try:
                        # Execute the same query to get DataFrame
                        dataframe = pd.read_sql_query(sql_query, self.db._engine)
                        logger.debug(f"SQL query result captured with shape: {dataframe.shape}")
                    except Exception as df_error:
                        logger.error(f"DataFrame creation failed for SQL: {sql_query[:100]}... Error: {df_error}")
                        # If DataFrame creation fails, still return the text result
                        pass
                
                execution_time = time.time() - start_time
                logger.info(f"Supply chain data analysis completed successfully in {execution_time:.2f}s")
                
                # Return structured response
                return {
                    "type": "text_with_sql_and_dataframe",
                    "text": result,
                    "sql_query": sql_query,
                    "dataframe": dataframe
                }
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Supply chain data analysis failed in {execution_time:.2f}s: {e}")
                
                # Return error as text-only response
                return {
                    "type": "text",
                    "content": f"Error analyzing data: {str(e)}"
                }
        
        @tool 
        def create_visualization(query: str) -> Dict[str, Any]:
            """Create Plotly visualizations for supply chain data. Use this when users ask for charts, plots, or visual analysis."""
            
            start_time = time.time()
            logger.info(f"Starting visualization creation: {query[:100]}...")
            
            try:
                # Determine what type of visualization is needed
                query_lower = query.lower()
                
                if any(word in query_lower for word in ['trend', 'time', 'monthly', 'daily', 'over time']):
                    result = self._create_time_series_chart(query)
                elif any(word in query_lower for word in ['top', 'highest', 'largest', 'ranking']):
                    result = self._create_ranking_chart(query)
                elif any(word in query_lower for word in ['inventory', 'stock', 'levels']):
                    result = self._create_inventory_chart(query)
                elif any(word in query_lower for word in ['cost', 'expense', 'storage', 'transfer']):
                    result = self._create_cost_chart(query)
                else:
                    result = self._create_general_chart(query)
                
                execution_time = time.time() - start_time
                
                if result["type"] == "plotly":
                    logger.info(f"Visualization created successfully: {result.get('description', 'unknown')}")
                else:
                    logger.error(f"Visualization creation failed: {result.get('message')}")
                
                logger.info(f"Visualization creation completed in {execution_time:.2f}s")
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Visualization creation failed in {execution_time:.2f}s: {e}")
                return {"type": "error", "message": f"Error creating visualization: {str(e)}"}
        
        self.analyze_tool = analyze_supply_chain_data
        self.visualize_tool = create_visualization
    
    def _create_time_series_chart(self, query: str) -> Dict[str, Any]:
        """Create time-series visualizations"""
        try:
            # SQL query for monthly transaction trends (lowercase columns)
            sql_query = """
            WITH combined_transactions AS (
                SELECT 
                    inbound_date as transaction_date,
                    'INBOUND' as transaction_type,
                    net_quantity_mt as net_quantity_mt
                FROM inbound
                UNION ALL
                SELECT 
                    outbound_date as transaction_date,
                    'OUTBOUND' as transaction_type,
                    net_quantity_mt as net_quantity_mt
                FROM outbound
            ),
            monthly_data AS (
                SELECT 
                    DATE_TRUNC('month', transaction_date::DATE) as month,
                    transaction_type,
                    SUM(net_quantity_mt) as total_quantity
                FROM combined_transactions
                WHERE transaction_date IS NOT NULL
                GROUP BY DATE_TRUNC('month', transaction_date::DATE), transaction_type
                ORDER BY month, transaction_type
            )
            SELECT 
                TO_CHAR(month, 'YYYY-MM') as month_str,
                transaction_type,
                total_quantity
            FROM monthly_data;
            """
            
            # Execute query and get results
            df = pd.read_sql_query(sql_query, self.db._engine)
            
            if df.empty:
                return {"type": "error", "message": "No transaction data found for time series"}
            
            fig = px.line(
                df, 
                x='month_str', 
                y='total_quantity', 
                color='transaction_type',
                title='Monthly Transaction Trends',
                labels={'total_quantity': 'Quantity (MT)', 'month_str': 'Month', 'transaction_type': 'Transaction Type'}
            )
            
            fig.update_layout(xaxis_tickangle=-45)
            return {"type": "plotly", "chart": fig, "description": "Monthly transaction trends showing inbound vs outbound volumes"}
            
        except Exception as e:
            return {"type": "error", "message": f"Error creating time series chart: {str(e)}"}
    
    def _create_ranking_chart(self, query: str) -> Dict[str, Any]:
        """Create ranking/top N visualizations"""
        try:
            if 'material' in query.lower():
                # Top materials by volume (lowercase columns)
                sql_query = """
                WITH combined_transactions AS (
                    SELECT 
                        material_name, 
                        net_quantity_mt 
                    FROM inbound
                    UNION ALL
                    SELECT 
                        material_name, 
                        net_quantity_mt 
                    FROM outbound
                ),
                material_totals AS (
                    SELECT 
                        material_name,
                        SUM(net_quantity_mt) as total_volume
                    FROM combined_transactions
                    WHERE material_name IS NOT NULL AND net_quantity_mt IS NOT NULL
                    GROUP BY material_name
                    ORDER BY total_volume DESC
                    LIMIT 10
                )
                SELECT material_name, total_volume FROM material_totals;
                """
                
                df = pd.read_sql_query(sql_query, self.db._engine)
                
                fig = px.bar(
                    df, 
                    x='total_volume', 
                    y='material_name',
                    orientation='h',
                    title='Top 10 Materials by Total Volume',
                    labels={'total_volume': 'Total Volume (MT)', 'material_name': 'Material'}
                )
                
            elif 'plant' in query.lower():
                # Top plants by volume (lowercase columns)
                sql_query = """
                WITH combined_transactions AS (
                    SELECT 
                        plant_name, 
                        net_quantity_mt 
                    FROM inbound
                    UNION ALL
                    SELECT 
                        plant_name, 
                        net_quantity_mt 
                    FROM outbound
                ),
                plant_totals AS (
                    SELECT 
                        plant_name,
                        SUM(net_quantity_mt) as total_volume
                    FROM combined_transactions
                    WHERE plant_name IS NOT NULL AND net_quantity_mt IS NOT NULL
                    GROUP BY plant_name
                    ORDER BY total_volume DESC
                    LIMIT 10
                )
                SELECT plant_name, total_volume FROM plant_totals;
                """
                
                df = pd.read_sql_query(sql_query, self.db._engine)
                
                fig = px.bar(
                    df, 
                    x='plant_name', 
                    y='total_volume',
                    title='Top 10 Plants by Total Volume',
                    labels={'total_volume': 'Total Volume (MT)', 'plant_name': 'Plant'}
                )
                
            else:
                # Default: top materials (lowercase columns)
                sql_query = """
                WITH combined_transactions AS (
                    SELECT 
                        material_name, 
                        net_quantity_mt 
                    FROM inbound
                    UNION ALL
                    SELECT 
                        material_name, 
                        net_quantity_mt 
                    FROM outbound
                ),
                material_totals AS (
                    SELECT 
                        material_name,
                        SUM(net_quantity_mt) as total_volume
                    FROM combined_transactions
                    WHERE material_name IS NOT NULL AND net_quantity_mt IS NOT NULL
                    GROUP BY material_name
                    ORDER BY total_volume DESC
                    LIMIT 10
                )
                SELECT material_name, total_volume FROM material_totals;
                """
                
                df = pd.read_sql_query(sql_query, self.db._engine)
                
                fig = px.bar(df, x='material_name', y='total_volume', title='Top 10 Materials by Volume')
            
            if df.empty:
                return {"type": "error", "message": "No data found for ranking chart"}
            
            fig.update_layout(xaxis_tickangle=-45)
            return {"type": "plotly", "chart": fig, "description": "Ranking chart based on your query"}
            
        except Exception as e:
            log_error(e, {"context": "_create_ranking_chart", "query": query})
            return {"type": "error", "message": f"Error creating ranking chart: {str(e)}"}
    
    def get_latest_chart_data(self):
        """Get the most recently stored chart data"""
        if not self.chart_memory:
            return None
        
        # Return the most recent chart
        latest_id = max(self.chart_memory.keys(), 
                       key=lambda k: self.chart_memory[k]["timestamp"])
        return self.chart_memory[latest_id]
    
    def get_chart_data_by_type(self, chart_type: str):
        """Get the most recent chart data of a specific type"""
        matching_charts = [
            chart for chart in self.chart_memory.values() 
            if chart["type"] == chart_type
        ]
        
        if not matching_charts:
            return None
        
        # Return the most recent of this type
        return max(matching_charts, key=lambda c: c["timestamp"])
    
    def has_recent_chart_data(self) -> bool:
        """Check if there's any recent chart data available"""
        return len(self.chart_memory) > 0
    
    def _setup_tools(self):
        """Setup tools using the tools module"""
        (
            self.analyze_tool,
            self.sql_tool,
            self.bar_chart_tool,
            self.line_chart_tool,
            self.scatter_plot_tool,
            self.histogram_tool,
            self.monthly_trends_tool,
            self.analyze_chart_data_tool
        ) = create_supply_chain_tools(
            db=self.db,
            llm=self.llm,
            memory=self.memory,
            agent=self
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
            self.monthly_trends_tool,
            self.analyze_chart_data_tool
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
        logger.info(f"Processing query: {query[:100]}...")
        
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
            logger.error(f"Query processing failed in {processing_time:.2f}s: {e}")
            return {"type": "error", "content": error_msg}
    
    def get_conversation_history(self, thread_id: str = "default") -> List[BaseMessage]:
        """Get conversation history for a specific thread"""
        try:
            config = {"configurable": {"thread_id": thread_id}}
            state = self.app.get_state(config)
            return state.values.get("messages", [])
        except Exception as e:
            logger.error(f"Failed to get conversation history for thread {thread_id}: {e}")
            return []
    
    def clear_memory(self, thread_id: str = "default"):
        """Clear conversation memory for a specific thread"""
        try:
            # Clear LangChain memory
            self.memory.clear()
            
            # Clear LangGraph state (note: MemorySaver doesn't have direct clear method)
            # Memory will naturally expire or can be handled at the application level
            logger.info("Memory cleared successfully")
            
        except Exception as e:
            logger.error(f"Failed to clear memory for thread {thread_id}: {e}")