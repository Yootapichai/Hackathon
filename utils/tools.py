"""
Supply Chain Analysis Tools

This module contains LangChain tools for analyzing supply chain data and creating visualizations.
Separated from the main agent for better code organization and maintainability.
"""

import time
import pandas as pd
import plotly.express as px
from typing import Dict, Any
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
        tuple: (analyze_tool, visualize_tool)
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

CRITICAL SQL Rules:
- ALWAYS use double quotes around ALL column names exactly as shown above
- Column names are case-sensitive - use the exact format listed
- Example: SELECT "MATERIAL_NAME", SUM("NET_QUANTITY_MT") FROM outbound

Tables:
- material_master: Material information ("MATERIAL_NAME", "POLYMER_TYPE", "SHELF_LIFE_IN_MONTH", "DOWNGRADE_VALUE_LOST_PERCENT")
- inventory: Current stock levels ("BALANCE_AS_OF_DATE", "PLANT_NAME", "MATERIAL_NAME", "BATCH_NUMBER", "UNRESTRICTED_STOCK", "STOCK_UNIT", "STOCK_SELL_VALUE", "CURRENCY")
- inbound: Incoming shipments ("INBOUND_DATE", "PLANT_NAME", "MATERIAL_NAME", "NET_QUANTITY_MT")
- outbound: Outgoing shipments ("OUTBOUND_DATE", "PLANT_NAME", "MATERIAL_NAME", "CUSTOMER_NUMBER", "MODE_OF_TRANSPORT', "NET_QUANTITY_MT")
- operation_costs: Storage and transfer costs ("OPERATION_CATEGORY", "COST_TYPE", "ENTITY_NAME", "ENTITY_TYPE", "COST_AMOUNT", "COST_UNIT", "CONTAINER_CAPACITY_MT", "CURRENCY")

Key Business Rules:
- Stock quantities are in KG for inventory, MT for inbound/outbound
- Different plants use different currencies (CNY for China, SGD for Singapore)
- Batch numbers track specific material lots
- Materials have shelf life and degradation rates

Always provide actionable insights and consider business context in your analysis.
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
    def create_visualization(query: str) -> Dict[str, Any]:
        """Create Plotly visualizations for supply chain data. Use this when users ask for charts, plots, or visual analysis."""
        
        start_time = time.time()
        log_tool_call("create_visualization", query)
        
        try:
            # Determine what type of visualization is needed
            query_lower = query.lower()
            
            if any(word in query_lower for word in ['trend', 'time', 'monthly', 'daily', 'over time']):
                result = create_time_series_chart(db, query)
            elif any(word in query_lower for word in ['top', 'highest', 'largest', 'ranking']):
                result = create_ranking_chart(db, query)
            elif any(word in query_lower for word in ['inventory', 'stock', 'levels']):
                result = create_inventory_chart(db, query)
            elif any(word in query_lower for word in ['cost', 'expense', 'storage', 'transfer']):
                result = create_cost_chart(db, query)
            else:
                result = create_general_chart(db, query)
            
            execution_time = time.time() - start_time
            
            if result["type"] == "plotly":
                log_visualization(result.get("description", "unknown"), 0, True)
                log_tool_result("create_visualization", "plotly", True)
            else:
                log_tool_result("create_visualization", "error", False, result.get("message"))
            
            log_tool_call("create_visualization", query, execution_time)
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            log_tool_result("create_visualization", "error", False, str(e))
            log_error(e, {"tool": "create_visualization", "query": query})
            return {"type": "error", "message": f"Error creating visualization: {str(e)}"}
    
    return analyze_supply_chain_data, create_visualization


def create_time_series_chart(db, query: str) -> Dict[str, Any]:
    """Create time-series visualizations"""
    try:
        # SQL query for monthly transaction trends (uppercase quoted columns)
        sql_query = """
        WITH combined_transactions AS (
            SELECT 
                "INBOUND_DATE" as transaction_date,
                'INBOUND' as transaction_type,
                "NET_QUANTITY_MT" as net_quantity_mt
            FROM inbound
            UNION ALL
            SELECT 
                "OUTBOUND_DATE" as transaction_date,
                'OUTBOUND' as transaction_type,
                "NET_QUANTITY_MT" as net_quantity_mt
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
        df = pd.read_sql_query(sql_query, db._engine)
        
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


def create_ranking_chart(db, query: str) -> Dict[str, Any]:
    """Create ranking/top N visualizations"""
    try:
        if 'material' in query.lower():
            # Top materials by volume (uppercase quoted columns)
            sql_query = """
            WITH combined_transactions AS (
                SELECT 
                    "MATERIAL_NAME" as material_name, 
                    "NET_QUANTITY_MT" as net_quantity_mt 
                FROM inbound
                UNION ALL
                SELECT 
                    "MATERIAL_NAME" as material_name, 
                    "NET_QUANTITY_MT" as net_quantity_mt 
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
            
            df = pd.read_sql_query(sql_query, db._engine)
            
            fig = px.bar(
                df, 
                x='total_volume', 
                y='material_name',
                orientation='h',
                title='Top 10 Materials by Total Volume',
                labels={'total_volume': 'Total Volume (MT)', 'material_name': 'Material'}
            )
            
        elif 'plant' in query.lower():
            # Top plants by volume (uppercase quoted columns)
            sql_query = """
            WITH combined_transactions AS (
                SELECT 
                    "PLANT_NAME" as plant_name, 
                    "NET_QUANTITY_MT" as net_quantity_mt 
                FROM inbound
                UNION ALL
                SELECT 
                    "PLANT_NAME" as plant_name, 
                    "NET_QUANTITY_MT" as net_quantity_mt 
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
            
            df = pd.read_sql_query(sql_query, db._engine)
            
            fig = px.bar(
                df, 
                x='plant_name', 
                y='total_volume',
                title='Top 10 Plants by Total Volume',
                labels={'total_volume': 'Total Volume (MT)', 'plant_name': 'Plant'}
            )
            
        else:
            # Default: top materials (uppercase quoted columns)
            sql_query = """
            WITH combined_transactions AS (
                SELECT 
                    "MATERIAL_NAME" as material_name, 
                    "NET_QUANTITY_MT" as net_quantity_mt 
                FROM inbound
                UNION ALL
                SELECT 
                    "MATERIAL_NAME" as material_name, 
                    "NET_QUANTITY_MT" as net_quantity_mt 
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
            
            df = pd.read_sql_query(sql_query, db._engine)
            
            fig = px.bar(df, x='material_name', y='total_volume', title='Top 10 Materials by Volume')
        
        if df.empty:
            return {"type": "error", "message": "No data found for ranking chart"}
        
        fig.update_layout(xaxis_tickangle=-45)
        return {"type": "plotly", "chart": fig, "description": "Ranking chart based on your query"}
        
    except Exception as e:
        return {"type": "error", "message": f"Error creating ranking chart: {str(e)}"}


def create_inventory_chart(db, query: str) -> Dict[str, Any]:
    """Create inventory-related visualizations"""
    try:
        # SQL query for inventory levels by plant (cast text to numeric)
        sql_query = """
        SELECT 
            "PLANT_NAME" as plant_name,
            SUM("UNRESTRICTED_STOCK") as total_inventory
        FROM inventory
        WHERE "UNRESTRICTED_STOCK" > 0
        GROUP BY "PLANT_NAME"
        ORDER BY total_inventory DESC;
        """
        
        df = pd.read_sql_query(sql_query, db._engine)
        
        if df.empty:
            return {"type": "error", "message": "No inventory data found"}
        
        fig = px.pie(
            df, 
            values='total_inventory', 
            names='plant_name',
            title='Inventory Distribution by Plant'
        )
        
        return {"type": "plotly", "chart": fig, "description": "Current inventory distribution across plants"}
        
    except Exception as e:
        return {"type": "error", "message": f"Error creating inventory chart: {str(e)}"}


def create_cost_chart(db, query: str) -> Dict[str, Any]:
    """Create cost-related visualizations"""
    try:
        if 'storage' in query.lower():
            # Storage costs by plant (uppercase quoted columns)
            sql_query = """
            SELECT 
                "ENTITY_NAME" as plant_name,
                "COST_AMOUNT" as storage_cost
            FROM operation_costs
            WHERE "COST_TYPE" = 'inventory_storage'
            ORDER BY storage_cost DESC;
            """
            
            df = pd.read_sql_query(sql_query, db._engine)
            
            if df.empty:
                return {"type": "error", "message": "No storage cost data found"}
            
            fig = px.bar(
                df, 
                x='plant_name', 
                y='storage_cost',
                title='Storage Costs by Plant',
                labels={'storage_cost': 'Cost per MT per Day', 'plant_name': 'Plant'}
            )
            
        elif 'transfer' in query.lower():
            # Transfer costs by transport mode (uppercase quoted columns)
            sql_query = """
            SELECT 
                "ENTITY_NAME" as transport_mode,
                "COST_AMOUNT" as transfer_cost
            FROM operation_costs
            WHERE "COST_TYPE" = 'truck_transfer'
            ORDER BY transfer_cost DESC;
            """
            
            df = pd.read_sql_query(sql_query, db._engine)
            
            if df.empty:
                return {"type": "error", "message": "No transfer cost data found"}
            
            fig = px.bar(
                df, 
                x='transport_mode', 
                y='transfer_cost',
                title='Transfer Costs by Transport Mode',
                labels={'transfer_cost': 'Cost per Container', 'transport_mode': 'Transport Mode'}
            )
            
        else:
            # Default storage costs (uppercase quoted columns)
            sql_query = """
            SELECT 
                "ENTITY_NAME" as plant_name,
                "COST_AMOUNT" as storage_cost
            FROM operation_costs
            WHERE "COST_TYPE" = 'inventory_storage'
            ORDER BY storage_cost DESC;
            """
            
            df = pd.read_sql_query(sql_query, db._engine)
            
            if df.empty:
                return {"type": "error", "message": "No cost data found"}
            
            fig = px.bar(df, x='plant_name', y='storage_cost', title='Storage Costs by Plant')
        
        fig.update_layout(xaxis_tickangle=-45)
        return {"type": "plotly", "chart": fig, "description": "Cost analysis visualization"}
        
    except Exception as e:
        return {"type": "error", "message": f"Error creating cost chart: {str(e)}"}


def create_general_chart(db, query: str) -> Dict[str, Any]:
    """Create general visualizations"""
    try:
        # Default: transaction type distribution
        sql_query = """
        WITH transaction_counts AS (
            SELECT 'INBOUND' as transaction_type, COUNT(*) as count FROM inbound
            UNION ALL
            SELECT 'OUTBOUND' as transaction_type, COUNT(*) as count FROM outbound
        )
        SELECT transaction_type, count FROM transaction_counts;
        """
        
        df = pd.read_sql_query(sql_query, db._engine)
        
        if df.empty:
            return {"type": "error", "message": "No transaction data found"}
        
        fig = px.pie(
            df, 
            values='count', 
            names='transaction_type',
            title='Transaction Type Distribution'
        )
        
        return {"type": "plotly", "chart": fig, "description": "General overview of transaction types"}
        
    except Exception as e:
        return {"type": "error", "message": f"Error creating general chart: {str(e)}"}