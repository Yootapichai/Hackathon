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
from .logger import log_tool_call, log_tool_result, log_error, log_dataframe_operation, log_visualization


def create_supply_chain_tools(db, llm, memory):
    """
    Factory function to create supply chain analysis tools with database and LLM dependencies.
    
    Args:
        db: QueryCapturingSQLDatabase instance
        llm: Language model instance
        memory: ConversationBufferWindowMemory instance
    
    Returns:
        tuple: (analyze_tool, sql_tool, bar_tool, line_tool, scatter_tool, histogram_tool)
    """
    
    @tool
    def analyze_supply_chain_data(query: str) -> Dict[str, Any]:
        """Analyze supply chain data using SQL queries. Returns SQL, DataFrame, and natural language answer."""
        
        start_time = time.time()
        log_tool_call("analyze_supply_chain_data", query)
        
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
                log_error(e, {"fallback": "trying without memory"})
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
                    log_dataframe_operation("sql_query_result", "captured", dataframe.shape)
                except Exception as df_error:
                    log_error(df_error, {"context": "dataframe_creation", "sql": sql_query})
                    # If DataFrame creation fails, still return the text result
                    pass
            
            execution_time = time.time() - start_time
            log_tool_result("analyze_supply_chain_data", "enhanced_sql_response", True)
            log_tool_call("analyze_supply_chain_data", query, execution_time)
            
            # Return structured response
            return {
                "type": "text_with_sql_and_dataframe",
                "text": result,
                "sql_query": sql_query,
                "dataframe": dataframe
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            log_tool_result("analyze_supply_chain_data", "error", False, str(e))
            log_error(e, {"tool": "analyze_supply_chain_data", "query": query})
            
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
        log_tool_call("execute_sql_for_chart", f"Type: {type(sql_query)}, Value: {str(sql_query)[:100]}")
        
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
            log_dataframe_operation("sql_chart_query", "executed", df.shape)
            log_tool_result("execute_sql_for_chart", "success", True)
            log_tool_call("execute_sql_for_chart", sql_query, execution_time)
            
            return {
                "type": "dataframe",
                "data": df,
                "sql_query": sql_query,
                "shape": df.shape
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            log_tool_result("execute_sql_for_chart", "error", False, str(e))
            log_error(e, {"tool": "execute_sql_for_chart", "sql": sql_query})
            
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
        
        log_tool_call("create_bar_chart", f"{title} | {x_column} vs {y_column}")
        
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
                    log_error(sql_error, {"context": "chart_sql_fallback", "sql": data_query[:200]})
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
            log_visualization("bar_chart", len(df), True)
            log_tool_result("create_bar_chart", "success", True)
            log_tool_call("create_bar_chart", title, execution_time)
            
            return {
                "type": "plotly",
                "chart": fig.to_json(),  # Convert to JSON for LangGraph compatibility
                "description": f"Bar chart: {title}"
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            log_tool_result("create_bar_chart", "error", False, str(e))
            log_error(e, {"tool": "create_bar_chart", "title": title})
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
        
        log_tool_call("create_line_chart", f"{title} | {x_column} vs {y_column}")
        
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
                    log_error(sql_error, {"context": "chart_sql_fallback", "sql": data_query[:200]})
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
            log_visualization("line_chart", len(df), True)
            log_tool_result("create_line_chart", "success", True)
            log_tool_call("create_line_chart", title, execution_time)
            
            return {
                "type": "plotly",
                "chart": fig.to_json(),  # Convert to JSON for LangGraph compatibility
                "description": f"Line chart: {title}"
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            log_tool_result("create_line_chart", "error", False, str(e))
            log_error(e, {"tool": "create_line_chart", "title": title})
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
        
        log_tool_call("create_scatter_plot", f"{title} | {x_column} vs {y_column}")
        
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
                    log_error(sql_error, {"context": "chart_sql_fallback", "sql": data_query[:200]})
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
            log_visualization("scatter_plot", len(df), True)
            log_tool_result("create_scatter_plot", "success", True)
            log_tool_call("create_scatter_plot", title, execution_time)
            
            return {
                "type": "plotly",
                "chart": fig.to_json(),  # Convert to JSON for LangGraph compatibility
                "description": f"Scatter plot: {title}"
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            log_tool_result("create_scatter_plot", "error", False, str(e))
            log_error(e, {"tool": "create_scatter_plot", "title": title})
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
        
        log_tool_call("create_histogram", f"{title} | {column}")
        
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
                    log_error(sql_error, {"context": "chart_sql_fallback", "sql": data_query[:200]})
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
            log_visualization("histogram", len(df), True)
            log_tool_result("create_histogram", "success", True)
            log_tool_call("create_histogram", title, execution_time)
            
            return {
                "type": "plotly", 
                "chart": fig.to_json(),  # Convert to JSON for LangGraph compatibility
                "description": f"Histogram: {title}"
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            log_tool_result("create_histogram", "error", False, str(e))
            log_error(e, {"tool": "create_histogram", "title": title})
            return {"type": "error", "message": f"Error creating histogram: {str(e)}"}
    
    @tool
    def plot_monthly_transaction_trends() -> Dict[str, Any]:
        """
        Create a line chart showing monthly inbound vs outbound transaction trends.
        Use this specifically when user asks for monthly transaction trends, inbound/outbound trends over time.
        This tool handles the data query and chart creation automatically.
        """
        start_time = time.time()
        log_tool_call("plot_monthly_transaction_trends", "Monthly transaction trends")
        
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
            log_visualization("monthly_trends", len(df), True)
            log_tool_result("plot_monthly_transaction_trends", "success", True)
            log_tool_call("plot_monthly_transaction_trends", "Monthly transaction trends", execution_time)
            
            return {
                "type": "plotly",
                "chart": fig.to_json(),
                "description": "Monthly transaction trends showing inbound vs outbound volumes over time"
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            log_tool_result("plot_monthly_transaction_trends", "error", False, str(e))
            log_error(e, {"tool": "plot_monthly_transaction_trends"})
            return {"type": "error", "message": f"Error creating monthly trends chart: {str(e)}"}
    
    return (
        analyze_supply_chain_data,
        execute_sql_for_chart, 
        create_bar_chart,
        create_line_chart,
        create_scatter_plot,
        create_histogram,
        plot_monthly_transaction_trends
    )