"""
Dynamic Supply Chain Analysis Tools

This module contains LangChain tools for analyzing supply chain data with dynamic, 
LLM-driven chart generation. The LLM chooses appropriate chart types and data 
based on user context rather than keyword matching.
"""

import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional
from langchain.tools import tool
from loguru import logger


def create_supply_chain_tools(db, llm, memory, agent=None):
    """
    Factory function to create supply chain analysis tools with database and LLM dependencies.
    
    Args:
        db: QueryCapturingSQLDatabase instance
        llm: Language model instance
        memory: ConversationBufferWindowMemory instance
        agent: SupplyChainAgent instance for chart memory access
    
    Returns:
        tuple: (analyze_tool, sql_tool, bar_tool, line_tool, scatter_tool, histogram_tool, trends_tool, chart_analysis_tool)
    """
    
    @tool
    def analyze_supply_chain_data(query: str) -> Dict[str, Any]:
        """Analyze supply chain data using SQL queries. Returns SQL, DataFrame, and natural language answer."""
        
        start_time = time.time()
        logger.info(f"Tool call: analyze_supply_chain_data with query: {query[:100]}")
        
        try:
            from langchain_community.agent_toolkits.sql.base import create_sql_agent
            from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
            
            # Clear previous query cache
            db.clear_query_cache()
            
            # Enhanced system prompt for SQL agent with business context
            system_prefix = """
You are an expert supply chain data analyst with access to a PostgreSQL database containing:

SQL Rules:
- Column names are lowercase without quotes
- Example: SELECT material_name, SUM(net_quantity_mt) FROM outbound

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

IMPORTANT FOR CHARTS: When user requests charts/visualizations:
1. First use analyze_supply_chain_data to get the data
2. Then use the appropriate chart tool with data_query='use_last'
3. Never create SQL queries for chart tools - use existing analyzed data

IMPORTANT FOR FOLLOW-UP QUESTIONS: When user asks follow-up questions about recently generated charts:
1. Use analyze_existing_chart_data tool for questions like:
   - "What's the highest value in the chart?"
   - "Which month had the peak?"
   - "What's the trend shown?"
   - "What date shows the maximum value?"
2. This is more efficient than making new SQL queries
3. Only use if there's a recently generated chart to analyze

Your answer must end with smile emoji
"""
            
            # Create SQL toolkit
            toolkit = SQLDatabaseToolkit(db=db, llm=llm)
            
            # Create SQL agent with proper memory integration
            try:
                agent = create_sql_agent(
                    llm=llm,
                    toolkit=toolkit,
                    verbose=True,
                    agent_type="openai-tools",
                    prefix=system_prefix,
                    handle_parsing_errors=True,
                    memory=memory
                )
            except Exception as e:
                logger.error(f"SQL agent creation failed, trying without memory: {e}")
                # Fallback without memory if it causes issues
                agent = create_sql_agent(
                    llm=llm,
                    toolkit=toolkit,
                    verbose=True,
                    agent_type="openai-tools",
                    prefix=system_prefix,
                    handle_parsing_errors=True
                )
            
            # Use memory-aware invocation
            result = agent.invoke({"input": query})["output"]
            
            # Get the captured SQL query and create DataFrame
            query_info = db.get_last_query_info()
            sql_query = query_info.get("query", "No SQL query captured")
            
            # Create DataFrame from the executed query if available
            dataframe = None
            if sql_query and sql_query != "No SQL query captured":
                try:
                    # Execute the same query to get DataFrame
                    dataframe = pd.read_sql_query(sql_query, db._engine)
                    logger.info(f"Created DataFrame from SQL query result: shape {dataframe.shape}")
                except Exception as df_error:
                    logger.error(f"DataFrame creation failed for SQL query: {df_error}")
                    # If DataFrame creation fails, still return the text result
                    pass
            
            execution_time = time.time() - start_time
            logger.info(f"Tool analyze_supply_chain_data completed successfully in {execution_time:.2f}s")
            
            # Return structured response
            return {
                "type": "text_with_sql_and_dataframe",
                "text": result,
                "sql_query": sql_query,
                "dataframe": dataframe
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Tool analyze_supply_chain_data failed for query '{query[:100]}': {e}")
            
            # Return error as text-only response
            return {
                "type": "text",
                "content": f"Error analyzing data: {str(e)}"
            }
    
    @tool
    def execute_sql_for_chart(sql_query: str) -> Dict[str, Any]:
        """
        Execute a custom PostgreSQL query and return the results as a DataFrame for chart creation.
        Use this tool to get data for visualization when you need specific data combinations.
        
        IMPORTANT: This uses PostgreSQL syntax, NOT SQLite. Use PostgreSQL functions like:
        - DATE_TRUNC() for date truncation
        - TO_CHAR() for date formatting
        - EXTRACT() for date parts
        
        Args:
            sql_query: The PostgreSQL SQL query to execute
            
        Returns:
            Dict with DataFrame data or error message
        """
        start_time = time.time()
        
        # Debug logging
        logger.info(f"Tool call: execute_sql_for_chart with SQL: {str(sql_query)[:100]}")
        
        try:
            # Ensure sql_query is a string (handle immutabledict from LangGraph)
            if hasattr(sql_query, 'get'):
                # If it's a dict-like object, extract the actual query
                sql_query = str(sql_query)
            elif not isinstance(sql_query, str):
                sql_query = str(sql_query)
            
            # Execute the SQL query
            df = pd.read_sql_query(sql_query, db._engine)
            
            execution_time = time.time() - start_time
            logger.info(f"Tool execute_sql_for_chart completed successfully: shape {df.shape} in {execution_time:.2f}s")
            
            return {
                "type": "dataframe",
                "data": df,
                "sql_query": sql_query,
                "shape": df.shape
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Tool execute_sql_for_chart failed: {e}")
            
            return {
                "type": "error",
                "message": f"Error executing SQL query: {str(e)}. Query was: {sql_query[:200]}..."
            }
    
    @tool
    def create_bar_chart(
        data_query: str,
        x_column: str, 
        y_column: str, 
        title: str,
        color_column: Optional[str] = None,
        orientation: str = "v"
    ) -> Dict[str, Any]:
        """
        Create a bar chart visualization from supply chain data. Use this when the user asks for:
        - Bar charts, bar graphs, or bar plots
        - Comparing quantities, costs, or counts across categories
        - Ranking data (top plants, materials, etc.)
        - Any categorical comparison visualization
        
        Args:
            data_query: SQL query to get the data, or 'use_last' to use last query result
            x_column: Column name for x-axis (categories)
            y_column: Column name for y-axis (values)
            title: Chart title
            color_column: Optional column for color coding bars
            orientation: 'v' for vertical, 'h' for horizontal bars
            
        Returns:
            Dict with Plotly chart or error message
        """
        start_time = time.time()
        
        # Ensure all parameters are strings (handle immutabledict from LangGraph)
        data_query = str(data_query) if not isinstance(data_query, str) else data_query
        x_column = str(x_column) if not isinstance(x_column, str) else x_column
        y_column = str(y_column) if not isinstance(y_column, str) else y_column
        title = str(title) if not isinstance(title, str) else title
        
        logger.info(f"Tool call: create_bar_chart - {title} ({x_column} vs {y_column})")
        
        try:
            # Ensure data_query is a string (handle immutabledict from LangGraph)
            if hasattr(data_query, 'get'):
                data_query = str(data_query)
            elif not isinstance(data_query, str):
                data_query = str(data_query)
            
            # Get data  
            if data_query == "use_last":
                query_info = db.get_last_query_info()
                if not query_info.get("query"):
                    return {"type": "error", "message": "No previous query to use"}
                df = pd.read_sql_query(query_info["query"], db._engine)
            elif data_query.startswith("SELECT") or data_query.startswith("WITH"):
                # It's a SQL query - but validate it first
                try:
                    df = pd.read_sql_query(data_query, db._engine)
                except Exception as sql_error:
                    # SQL failed, fall back to using last query
                    logger.error(f"Chart SQL fallback failed: {sql_error}")
                    query_info = db.get_last_query_info()
                    if query_info.get("query"):
                        df = pd.read_sql_query(query_info["query"], db._engine)
                    else:
                        return {"type": "error", "message": f"SQL query failed and no previous data available: {str(sql_error)}"}
            else:
                # It might be malformed, try to use last query instead
                query_info = db.get_last_query_info()
                if query_info.get("query"):
                    df = pd.read_sql_query(query_info["query"], db._engine)
                else:
                    return {"type": "error", "message": f"Invalid data_query. Use 'use_last' to use existing data, or provide valid SQL. Got: {data_query[:100]}"}
            
            if df.empty:
                return {"type": "error", "message": "No data found for bar chart"}
            
            # Validate columns exist
            if x_column not in df.columns:
                return {"type": "error", "message": f"Column '{x_column}' not found in data"}
            if y_column not in df.columns:
                return {"type": "error", "message": f"Column '{y_column}' not found in data"}
            
            # Create bar chart
            fig = px.bar(
                df,
                x=x_column if orientation == "v" else y_column,
                y=y_column if orientation == "v" else x_column,
                color=color_column if color_column and color_column in df.columns else None,
                title=title,
                orientation=orientation
            )
            
            # Improve layout
            fig.update_layout(
                xaxis_tickangle=-45 if orientation == "v" else 0,
                showlegend=bool(color_column)
            )
            
            execution_time = time.time() - start_time
            logger.info(f"Tool create_bar_chart completed successfully: {len(df)} data points in {execution_time:.2f}s")
            
            # Store chart data in agent memory if agent is available
            if agent:
                # Get the SQL query for storage
                if data_query == "use_last":
                    query_info = db.get_last_query_info()
                    storage_sql = query_info.get("query", "use_last")
                else:
                    storage_sql = data_query
                
                agent.store_chart_data(
                    chart_type="bar_chart",
                    dataframe=df,
                    sql_query=storage_sql,
                    description=f"Bar chart: {title}"
                )
            
            return {
                "type": "plotly",
                "chart": fig.to_json(),  # Convert to JSON for LangGraph compatibility
                "description": f"Bar chart: {title}"
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Tool create_bar_chart failed for '{title}': {e}")
            return {"type": "error", "message": f"Error creating bar chart: {str(e)}"}
    
    @tool
    def create_line_chart(
        data_query: str,
        x_column: str,
        y_column: str, 
        title: str,
        color_column: Optional[str] = None,
        sort_by_x: bool = True
    ) -> Dict[str, Any]:
        """
        Create a line chart visualization from supply chain data. Use this when the user asks for:
        - Line charts, line graphs, or line plots
        - Time series analysis (trends over time)
        - Monthly, daily, or weekly trends
        - Plotting trends, patterns, or changes over time
        - Any time-based or sequential data visualization
        
        IMPORTANT: For monthly transaction trends, first use analyze_supply_chain_data to get the data,
        then use data_query='use_last' to create the chart from that data.
        
        Args:
            data_query: Use 'use_last' to chart the most recent analyze_supply_chain_data result (RECOMMENDED)
            x_column: Column name for x-axis (usually time/date or sequential data)  
            y_column: Column name for y-axis (numeric values to track)
            title: Chart title
            color_column: Optional column for multiple lines with different colors
            sort_by_x: Whether to sort data by x-axis values
            
        Returns:
            Dict with Plotly chart or error message
        """
        start_time = time.time()
        
        # Ensure all parameters are strings (handle immutabledict from LangGraph)
        data_query = str(data_query) if not isinstance(data_query, str) else data_query
        x_column = str(x_column) if not isinstance(x_column, str) else x_column
        y_column = str(y_column) if not isinstance(y_column, str) else y_column
        title = str(title) if not isinstance(title, str) else title
        
        logger.info(f"Tool call: create_line_chart - {title} ({x_column} vs {y_column})")
        
        try:
            # Ensure data_query is a string (handle immutabledict from LangGraph)
            if hasattr(data_query, 'get'):
                data_query = str(data_query)
            elif not isinstance(data_query, str):
                data_query = str(data_query)
            
            # Get data  
            if data_query == "use_last":
                query_info = db.get_last_query_info()
                if not query_info.get("query"):
                    return {"type": "error", "message": "No previous query to use"}
                df = pd.read_sql_query(query_info["query"], db._engine)
            elif data_query.startswith("SELECT") or data_query.startswith("WITH"):
                # It's a SQL query - but validate it first
                try:
                    df = pd.read_sql_query(data_query, db._engine)
                except Exception as sql_error:
                    # SQL failed, fall back to using last query
                    logger.error(f"Chart SQL fallback failed: {sql_error}")
                    query_info = db.get_last_query_info()
                    if query_info.get("query"):
                        df = pd.read_sql_query(query_info["query"], db._engine)
                    else:
                        return {"type": "error", "message": f"SQL query failed and no previous data available: {str(sql_error)}"}
            else:
                # It might be malformed, try to use last query instead
                query_info = db.get_last_query_info()
                if query_info.get("query"):
                    df = pd.read_sql_query(query_info["query"], db._engine)
                else:
                    return {"type": "error", "message": f"Invalid data_query. Use 'use_last' to use existing data, or provide valid SQL. Got: {data_query[:100]}"}
            
            if df.empty:
                return {"type": "error", "message": "No data found for line chart"}
            
            # Validate columns exist
            if x_column not in df.columns:
                return {"type": "error", "message": f"Column '{x_column}' not found in data"}
            if y_column not in df.columns:
                return {"type": "error", "message": f"Column '{y_column}' not found in data"}
            
            # Sort data if requested
            if sort_by_x:
                df = df.sort_values(by=x_column)
            
            # Create line chart
            fig = px.line(
                df,
                x=x_column,
                y=y_column,
                color=color_column if color_column and color_column in df.columns else None,
                title=title,
                markers=True
            )
            
            # Improve layout
            fig.update_layout(
                xaxis_tickangle=-45,
                showlegend=bool(color_column)
            )
            
            execution_time = time.time() - start_time
            logger.info(f"Tool create_line_chart completed successfully: {len(df)} data points in {execution_time:.2f}s")
            
            # Store chart data in agent memory if agent is available
            if agent:
                # Get the SQL query for storage
                if data_query == "use_last":
                    query_info = db.get_last_query_info()
                    storage_sql = query_info.get("query", "use_last")
                else:
                    storage_sql = data_query
                
                agent.store_chart_data(
                    chart_type="line_chart",
                    dataframe=df,
                    sql_query=storage_sql,
                    description=f"Line chart: {title}"
                )
            
            return {
                "type": "plotly",
                "chart": fig.to_json(),  # Convert to JSON for LangGraph compatibility
                "description": f"Line chart: {title}"
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Tool create_line_chart failed for '{title}': {e}")
            return {"type": "error", "message": f"Error creating line chart: {str(e)}"}
    
    @tool
    def create_scatter_plot(
        data_query: str,
        x_column: str,
        y_column: str,
        title: str,
        color_column: Optional[str] = None,
        size_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a scatter plot from supply chain data to show relationships between variables.
        
        Args:
            data_query: SQL query to get the data, or 'use_last' to use last query result
            x_column: Column name for x-axis
            y_column: Column name for y-axis
            title: Chart title
            color_column: Optional column for color coding points
            size_column: Optional column for sizing points
            
        Returns:
            Dict with Plotly chart or error message
        """
        start_time = time.time()
        
        # Ensure all parameters are strings (handle immutabledict from LangGraph)
        data_query = str(data_query) if not isinstance(data_query, str) else data_query
        x_column = str(x_column) if not isinstance(x_column, str) else x_column
        y_column = str(y_column) if not isinstance(y_column, str) else y_column
        title = str(title) if not isinstance(title, str) else title
        
        logger.info(f"Tool call: create_scatter_plot - {title} ({x_column} vs {y_column})")
        
        try:
            # Ensure data_query is a string (handle immutabledict from LangGraph)
            if hasattr(data_query, 'get'):
                data_query = str(data_query)
            elif not isinstance(data_query, str):
                data_query = str(data_query)
            
            # Get data  
            if data_query == "use_last":
                query_info = db.get_last_query_info()
                if not query_info.get("query"):
                    return {"type": "error", "message": "No previous query to use"}
                df = pd.read_sql_query(query_info["query"], db._engine)
            elif data_query.startswith("SELECT") or data_query.startswith("WITH"):
                # It's a SQL query - but validate it first
                try:
                    df = pd.read_sql_query(data_query, db._engine)
                except Exception as sql_error:
                    # SQL failed, fall back to using last query
                    logger.error(f"Chart SQL fallback failed: {sql_error}")
                    query_info = db.get_last_query_info()
                    if query_info.get("query"):
                        df = pd.read_sql_query(query_info["query"], db._engine)
                    else:
                        return {"type": "error", "message": f"SQL query failed and no previous data available: {str(sql_error)}"}
            else:
                # It might be malformed, try to use last query instead
                query_info = db.get_last_query_info()
                if query_info.get("query"):
                    df = pd.read_sql_query(query_info["query"], db._engine)
                else:
                    return {"type": "error", "message": f"Invalid data_query. Use 'use_last' to use existing data, or provide valid SQL. Got: {data_query[:100]}"}
            
            if df.empty:
                return {"type": "error", "message": "No data found for scatter plot"}
            
            # Validate columns exist
            if x_column not in df.columns:
                return {"type": "error", "message": f"Column '{x_column}' not found in data"}
            if y_column not in df.columns:
                return {"type": "error", "message": f"Column '{y_column}' not found in data"}
            
            # Create scatter plot
            fig = px.scatter(
                df,
                x=x_column,
                y=y_column,
                color=color_column if color_column and color_column in df.columns else None,
                size=size_column if size_column and size_column in df.columns else None,
                title=title,
                hover_data=df.columns.tolist()  # Show all columns on hover
            )
            
            # Improve layout
            fig.update_layout(showlegend=bool(color_column))
            
            execution_time = time.time() - start_time
            logger.info(f"Tool create_scatter_plot completed successfully: {len(df)} data points in {execution_time:.2f}s")
            
            # Store chart data in agent memory if agent is available
            if agent:
                # Get the SQL query for storage
                if data_query == "use_last":
                    query_info = db.get_last_query_info()
                    storage_sql = query_info.get("query", "use_last")
                else:
                    storage_sql = data_query
                
                agent.store_chart_data(
                    chart_type="scatter_plot",
                    dataframe=df,
                    sql_query=storage_sql,
                    description=f"Scatter plot: {title}"
                )
            
            return {
                "type": "plotly",
                "chart": fig.to_json(),  # Convert to JSON for LangGraph compatibility
                "description": f"Scatter plot: {title}"
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Tool create_scatter_plot failed for '{title}': {e}")
            return {"type": "error", "message": f"Error creating scatter plot: {str(e)}"}
    
    @tool
    def create_histogram(
        data_query: str,
        column: str,
        title: str,
        bins: int = 20,
        color_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a histogram from supply chain data to show distribution of values.
        
        Args:
            data_query: SQL query to get the data, or 'use_last' to use last query result
            column: Column name to create histogram for
            title: Chart title
            bins: Number of bins for the histogram
            color_column: Optional column for color coding histogram bars
            
        Returns:
            Dict with Plotly chart or error message
        """
        start_time = time.time()
        
        # Ensure all parameters are strings (handle immutabledict from LangGraph)
        data_query = str(data_query) if not isinstance(data_query, str) else data_query
        column = str(column) if not isinstance(column, str) else column
        title = str(title) if not isinstance(title, str) else title
        
        logger.info(f"Tool call: create_histogram - {title} ({column})")
        
        try:
            # Ensure data_query is a string (handle immutabledict from LangGraph)
            if hasattr(data_query, 'get'):
                data_query = str(data_query)
            elif not isinstance(data_query, str):
                data_query = str(data_query)
            
            # Get data  
            if data_query == "use_last":
                query_info = db.get_last_query_info()
                if not query_info.get("query"):
                    return {"type": "error", "message": "No previous query to use"}
                df = pd.read_sql_query(query_info["query"], db._engine)
            elif data_query.startswith("SELECT") or data_query.startswith("WITH"):
                # It's a SQL query - but validate it first
                try:
                    df = pd.read_sql_query(data_query, db._engine)
                except Exception as sql_error:
                    # SQL failed, fall back to using last query
                    logger.error(f"Chart SQL fallback failed: {sql_error}")
                    query_info = db.get_last_query_info()
                    if query_info.get("query"):
                        df = pd.read_sql_query(query_info["query"], db._engine)
                    else:
                        return {"type": "error", "message": f"SQL query failed and no previous data available: {str(sql_error)}"}
            else:
                # It might be malformed, try to use last query instead
                query_info = db.get_last_query_info()
                if query_info.get("query"):
                    df = pd.read_sql_query(query_info["query"], db._engine)
                else:
                    return {"type": "error", "message": f"Invalid data_query. Use 'use_last' to use existing data, or provide valid SQL. Got: {data_query[:100]}"}
            
            if df.empty:
                return {"type": "error", "message": "No data found for histogram"}
            
            # Validate column exists
            if column not in df.columns:
                return {"type": "error", "message": f"Column '{column}' not found in data"}
            
            # Create histogram
            fig = px.histogram(
                df,
                x=column,
                color=color_column if color_column and color_column in df.columns else None,
                title=title,
                nbins=bins
            )
            
            # Improve layout
            fig.update_layout(
                bargap=0.1,
                showlegend=bool(color_column)
            )
            
            execution_time = time.time() - start_time
            logger.info(f"Tool create_histogram completed successfully: {len(df)} data points in {execution_time:.2f}s")
            
            # Store chart data in agent memory if agent is available
            if agent:
                # Get the SQL query for storage
                if data_query == "use_last":
                    query_info = db.get_last_query_info()
                    storage_sql = query_info.get("query", "use_last")
                else:
                    storage_sql = data_query
                
                agent.store_chart_data(
                    chart_type="histogram",
                    dataframe=df,
                    sql_query=storage_sql,
                    description=f"Histogram: {title}"
                )
            
            return {
                "type": "plotly", 
                "chart": fig.to_json(),  # Convert to JSON for LangGraph compatibility
                "description": f"Histogram: {title}"
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Tool create_histogram failed for '{title}': {e}")
            return {"type": "error", "message": f"Error creating histogram: {str(e)}"}
    
    @tool
    def analyze_existing_chart_data(question: str) -> Dict[str, Any]:
        """
        Analyze data from recently generated charts without making new SQL queries.
        Use this for follow-up questions about charts like:
        - "What's the highest value in the chart?"
        - "Which month had the peak inbound?"
        - "From the chart, what's the trend?"
        - "What date shows the maximum value?"
        
        This tool is efficient for chart follow-ups as it uses existing data.
        Only use this if there's a recently generated chart to analyze.
        
        Args:
            question: The question about the existing chart data
        """
        start_time = time.time()
        logger.info(f"Tool call: analyze_existing_chart_data - {question[:100]}")
        
        try:
            # Check if agent is available and has chart data
            if not agent or not agent.has_recent_chart_data():
                return {
                    "type": "error",
                    "message": "No recent chart data available. Please generate a chart first."
                }
            
            # Get the most recent chart data
            chart_data = agent.get_latest_chart_data()
            df = chart_data["dataframe"]
            chart_type = chart_data["type"]
            
            if df is None or df.empty:
                return {
                    "type": "error", 
                    "message": "Chart data is empty or unavailable."
                }
            
            # Analyze the dataframe based on the question
            analysis_result = ""
            
            # Common analysis patterns
            question_lower = question.lower()
            
            if any(word in question_lower for word in ['highest', 'maximum', 'peak', 'max']):
                # Find maximum values
                if 'inbound' in question_lower:
                    # Look for inbound-specific max
                    if 'transaction_type' in df.columns:
                        inbound_data = df[df['transaction_type'] == 'Inbound']
                        if not inbound_data.empty:
                            max_row = inbound_data.loc[inbound_data['total_quantity'].idxmax()]
                            analysis_result = f"The highest inbound value is {max_row['total_quantity']:.2f} MT in {max_row['month']}."
                    elif 'total_inbound_mt' in df.columns:
                        max_row = df.loc[df['total_inbound_mt'].idxmax()]
                        analysis_result = f"The highest inbound value is {max_row['total_inbound_mt']:.2f} MT in {max_row['month']}."
                elif 'outbound' in question_lower:
                    # Look for outbound-specific max
                    if 'transaction_type' in df.columns:
                        outbound_data = df[df['transaction_type'] == 'Outbound']
                        if not outbound_data.empty:
                            max_row = outbound_data.loc[outbound_data['total_quantity'].idxmax()]
                            analysis_result = f"The highest outbound value is {max_row['total_quantity']:.2f} MT in {max_row['month']}."
                    elif 'total_outbound_mt' in df.columns:
                        max_row = df.loc[df['total_outbound_mt'].idxmax()]
                        analysis_result = f"The highest outbound value is {max_row['total_outbound_mt']:.2f} MT in {max_row['month']}."
                else:
                    # General maximum - find numeric columns
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        main_col = numeric_cols[0]
                        max_idx = df[main_col].idxmax()
                        max_row = df.loc[max_idx]
                        date_col = [col for col in df.columns if 'month' in col.lower() or 'date' in col.lower()]
                        if date_col:
                            analysis_result = f"The maximum value is {max_row[main_col]:.2f} in {max_row[date_col[0]]}."
                        else:
                            analysis_result = f"The maximum value is {max_row[main_col]:.2f}."
            
            elif any(word in question_lower for word in ['lowest', 'minimum', 'min']):
                # Find minimum values
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    main_col = numeric_cols[0]
                    min_idx = df[main_col].idxmin()
                    min_row = df.loc[min_idx]
                    date_col = [col for col in df.columns if 'month' in col.lower() or 'date' in col.lower()]
                    if date_col:
                        analysis_result = f"The minimum value is {min_row[main_col]:.2f} in {min_row[date_col[0]]}."
                    else:
                        analysis_result = f"The minimum value is {min_row[main_col]:.2f}."
            
            elif any(word in question_lower for word in ['trend', 'pattern', 'direction']):
                # Analyze trends
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    main_col = numeric_cols[0]
                    if len(df) > 1:
                        start_val = df[main_col].iloc[0]
                        end_val = df[main_col].iloc[-1]
                        if end_val > start_val:
                            analysis_result = f"The trend shows an overall increase from {start_val:.2f} to {end_val:.2f}."
                        elif end_val < start_val:
                            analysis_result = f"The trend shows an overall decrease from {start_val:.2f} to {end_val:.2f}."
                        else:
                            analysis_result = f"The trend is relatively stable around {start_val:.2f}."
            
            elif any(word in question_lower for word in ['total', 'sum', 'aggregate']):
                # Calculate totals
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    totals = []
                    for col in numeric_cols:
                        total = df[col].sum()
                        totals.append(f"{col}: {total:.2f}")
                    analysis_result = f"Totals: {', '.join(totals)}"
            
            else:
                # General summary
                analysis_result = f"Chart contains {len(df)} data points"
                if 'month' in df.columns:
                    date_range = f" from {df['month'].min()} to {df['month'].max()}"
                    analysis_result += date_range
                
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    main_col = numeric_cols[0]
                    analysis_result += f". Values range from {df[main_col].min():.2f} to {df[main_col].max():.2f}."
            
            # If no specific analysis was done, provide general info
            if not analysis_result:
                analysis_result = f"Chart data has {len(df)} rows with columns: {', '.join(df.columns)}. Please ask a more specific question about the data."
            
            execution_time = time.time() - start_time
            logger.info(f"Tool analyze_existing_chart_data completed successfully in {execution_time:.2f}s")
            
            return {
                "type": "chart_analysis",
                "analysis": analysis_result,
                "chart_type": chart_type,
                "data_points": len(df),
                "columns": list(df.columns)
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Tool analyze_existing_chart_data failed for question '{question[:100]}': {e}")
            return {"type": "error", "message": f"Error analyzing chart data: {str(e)}"}
    
    @tool
    def plot_monthly_transaction_trends() -> Dict[str, Any]:
        """
        Create a line chart showing monthly inbound vs outbound transaction trends.
        Use this specifically when user asks for monthly transaction trends, inbound/outbound trends over time.
        This tool handles the data query and chart creation automatically.
        """
        start_time = time.time()
        logger.info("Tool call: plot_monthly_transaction_trends")
        
        try:
            # SQL query for monthly transaction trends using proper PostgreSQL syntax
            sql_query = """
            WITH monthly_inbound AS (
                SELECT 
                    TO_CHAR(TO_DATE(inbound_date, 'YYYY/MM/DD'), 'YYYY-MM') AS month,
                    'Inbound' as transaction_type,
                    SUM(net_quantity_mt) AS total_quantity
                FROM inbound 
                WHERE inbound_date IS NOT NULL
                GROUP BY TO_CHAR(TO_DATE(inbound_date, 'YYYY/MM/DD'), 'YYYY-MM')
            ),
            monthly_outbound AS (
                SELECT 
                    TO_CHAR(TO_DATE(outbound_date, 'YYYY/MM/DD'), 'YYYY-MM') AS month,
                    'Outbound' as transaction_type,
                    SUM(net_quantity_mt) AS total_quantity
                FROM outbound 
                WHERE outbound_date IS NOT NULL
                GROUP BY TO_CHAR(TO_DATE(outbound_date, 'YYYY/MM/DD'), 'YYYY-MM')
            )
            SELECT month, transaction_type, total_quantity 
            FROM monthly_inbound
            UNION ALL
            SELECT month, transaction_type, total_quantity 
            FROM monthly_outbound
            ORDER BY month, transaction_type;
            """
            
            # Execute query
            df = pd.read_sql_query(sql_query, db._engine)
            
            if df.empty:
                return {"type": "error", "message": "No transaction data found"}
            
            # Create line chart
            fig = px.line(
                df,
                x='month',
                y='total_quantity',
                color='transaction_type',
                title='Monthly Transaction Trends: Inbound vs Outbound',
                labels={'total_quantity': 'Quantity (MT)', 'month': 'Month', 'transaction_type': 'Transaction Type'},
                markers=True
            )
            
            fig.update_layout(
                xaxis_tickangle=-45,
                showlegend=True
            )
            
            execution_time = time.time() - start_time
            logger.info(f"Tool plot_monthly_transaction_trends completed successfully: {len(df)} data points in {execution_time:.2f}s")
            
            # Store chart data in agent memory if agent is available
            if agent:
                agent.store_chart_data(
                    chart_type="monthly_trends",
                    dataframe=df,
                    sql_query=sql_query,
                    description="Monthly transaction trends showing inbound vs outbound volumes over time"
                )
            
            return {
                "type": "plotly",
                "chart": fig.to_json(),
                "description": "Monthly transaction trends showing inbound vs outbound volumes over time"
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Tool plot_monthly_transaction_trends failed: {e}")
            return {"type": "error", "message": f"Error creating monthly trends chart: {str(e)}"}
    
    return (
        analyze_supply_chain_data,
        execute_sql_for_chart, 
        create_bar_chart,
        create_line_chart,
        create_scatter_plot,
        create_histogram,
        plot_monthly_transaction_trends,
        analyze_existing_chart_data
    )