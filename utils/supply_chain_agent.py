import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import traceback
from typing import Dict, Any, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from .dataframe import transactions_master_df, inventory_master_df, storage_cost_df, transfer_cost_df
from .logger import (
    log_query, log_tool_call, log_tool_result, log_agent_response, 
    log_error, log_dataframe_operation, log_visualization
)

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    last_query: str
    context: Dict[str, Any]

class SupplyChainAgent:
    def __init__(self):
        try:
            log_agent_response("agent_init", 0, 0)
            
            self.google_api_key = os.environ.get('GOOGLE_API_KEY')
            if not self.google_api_key:
                error_msg = "GOOGLE_API_KEY environment variable is required"
                log_error(ValueError(error_msg), {"component": "agent_init"})
                raise ValueError(error_msg)
            
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=self.google_api_key,
                temperature=0.0,
                verbose=True
            )
            
            self.dataframes = {
                'transactions': transactions_master_df,
                'inventory': inventory_master_df, 
                'storage_costs': storage_cost_df,
                'transfer_costs': transfer_cost_df
            }
            
            # Log dataframe info
            for name, df in self.dataframes.items():
                log_dataframe_operation("load", name, df.shape)
            
            self._setup_tools()
            self._setup_agent()
            
            log_agent_response("agent_init_complete", 0, 0)
            
        except Exception as e:
            log_error(e, {"component": "agent_init"})
            raise
    
    def _setup_tools(self):
        @tool
        def analyze_dataframe(query: str) -> str:
            """Analyze supply chain data using pandas operations. Use this for data analysis questions."""
            
            start_time = time.time()
            log_tool_call("analyze_dataframe", query)
            
            try:
                data_dictionary_prefix = """
You are a supply chain data analyst expert.
You have access to pandas dataframes with supply chain data.
The dataframes are available as df1, df2, df3, and df4.

Dataframe mapping:
- df1: transactions_master_df - Transaction log (inbound/outbound) with material details
- df2: inventory_master_df - Monthly inventory snapshots with material details  
- df3: storage_cost_df - Storage costs per MT per day by plant
- df4: transfer_cost_df - Transfer costs per container by transport mode

Important columns:
- TRANSACTION_DATE: Transaction date
- PLANT_NAME: Plant/warehouse name
- MATERIAL_NAME: Specific product name
- NET_QUANTITY_MT: Quantity in Metric Tons
- TRANSACTION_TYPE: 'INBOUND' or 'OUTBOUND'
- POLYMER_TYPE: Material category
- BALANCE_AS_OF_DATE: Inventory snapshot date
- UNRESTRICTED_STOCK: Available quantity
- STOCK_SELL_VALUE: Inventory value
- STORAGE_COST_PER_MT_DAY: Storage cost per MT per day
- MODE_OF_TRANSPORT: Transportation method
- TRANSFER_COST_PER_CONTAINER: Cost per container transfer

IMPORTANT: Always use df1, df2, df3, df4 in your Python code. Do not redefine these variables.

When you need to execute Python code, use the python_repl_ast tool.
Format your response as:

Thought: I need to analyze the data to answer this question.
Action: python_repl_ast
Action Input: <your_python_code_here>
Observation: <result_will_appear_here>
... (repeat until you have the final answer)
Final Answer: <your_final_answer>
                """
                
                # Try different agent types that work better with Gemini
                try:
                    agent = create_pandas_dataframe_agent(
                        self.llm,
                        [transactions_master_df, inventory_master_df, storage_cost_df, transfer_cost_df],
                        prefix=data_dictionary_prefix,
                        allow_dangerous_code=True,
                        verbose=True,
                        agent_type="zero-shot-react-description",  # More reliable with Gemini
                        handle_parsing_errors=True
                    )
                except Exception as e:
                    log_error(e, {"fallback": "trying openai-tools agent type"})
                    # Fallback to openai-tools if zero-shot fails
                    agent = create_pandas_dataframe_agent(
                        self.llm,
                        [transactions_master_df, inventory_master_df, storage_cost_df, transfer_cost_df],
                        prefix=data_dictionary_prefix,
                        allow_dangerous_code=True,
                        verbose=True,
                        agent_type="openai-tools",
                        handle_parsing_errors=True
                    )
                
                result = agent.invoke({"input": query})["output"]
                
                execution_time = time.time() - start_time
                log_tool_result("analyze_dataframe", "text", True)
                log_tool_call("analyze_dataframe", query, execution_time)
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                log_tool_result("analyze_dataframe", "error", False, str(e))
                log_error(e, {"tool": "analyze_dataframe", "query": query})
                return f"Error analyzing data: {str(e)}"
        
        @tool 
        def create_visualization(query: str) -> Dict[str, Any]:
            """Create Plotly visualizations for supply chain data. Use this when users ask for charts, plots, or visual analysis."""
            
            start_time = time.time()
            log_tool_call("create_visualization", query)
            
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
        
        self.analyze_tool = analyze_dataframe
        self.visualize_tool = create_visualization
    
    def _create_time_series_chart(self, query: str) -> Dict[str, Any]:
        """Create time-series visualizations"""
        try:
            # Monthly transaction trends
            df = transactions_master_df.copy()
            df['TRANSACTION_DATE'] = pd.to_datetime(df['TRANSACTION_DATE'])
            df['MONTH'] = df['TRANSACTION_DATE'].dt.to_period('M')
            
            monthly_data = df.groupby(['MONTH', 'TRANSACTION_TYPE'])['NET_QUANTITY_MT'].sum().reset_index()
            monthly_data['MONTH'] = monthly_data['MONTH'].astype(str)
            
            fig = px.line(
                monthly_data, 
                x='MONTH', 
                y='NET_QUANTITY_MT', 
                color='TRANSACTION_TYPE',
                title='Monthly Transaction Trends',
                labels={'NET_QUANTITY_MT': 'Quantity (MT)', 'MONTH': 'Month'}
            )
            
            return {"type": "plotly", "chart": fig, "description": "Monthly transaction trends showing inbound vs outbound volumes"}
            
        except Exception as e:
            return {"type": "error", "message": f"Error creating time series chart: {str(e)}"}
    
    def _create_ranking_chart(self, query: str) -> Dict[str, Any]:
        """Create ranking/top N visualizations"""
        try:
            if 'material' in query.lower():
                # Top materials by volume
                df = transactions_master_df.groupby('MATERIAL_NAME')['NET_QUANTITY_MT'].sum().reset_index()
                df = df.nlargest(10, 'NET_QUANTITY_MT')
                
                fig = px.bar(
                    df, 
                    x='NET_QUANTITY_MT', 
                    y='MATERIAL_NAME',
                    orientation='h',
                    title='Top 10 Materials by Total Volume',
                    labels={'NET_QUANTITY_MT': 'Total Volume (MT)', 'MATERIAL_NAME': 'Material'}
                )
                
            elif 'plant' in query.lower():
                # Top plants by volume
                df = transactions_master_df.groupby('PLANT_NAME')['NET_QUANTITY_MT'].sum().reset_index()
                df = df.nlargest(10, 'NET_QUANTITY_MT')
                
                fig = px.bar(
                    df, 
                    x='PLANT_NAME', 
                    y='NET_QUANTITY_MT',
                    title='Top 10 Plants by Total Volume',
                    labels={'NET_QUANTITY_MT': 'Total Volume (MT)', 'PLANT_NAME': 'Plant'}
                )
                
            else:
                # Default: top materials
                df = transactions_master_df.groupby('MATERIAL_NAME')['NET_QUANTITY_MT'].sum().reset_index()
                df = df.nlargest(10, 'NET_QUANTITY_MT')
                
                fig = px.bar(df, x='MATERIAL_NAME', y='NET_QUANTITY_MT', title='Top 10 Materials by Volume')
            
            fig.update_layout(xaxis_tickangle=-45)
            return {"type": "plotly", "chart": fig, "description": "Ranking chart based on your query"}
            
        except Exception as e:
            return {"type": "error", "message": f"Error creating ranking chart: {str(e)}"}
    
    def _create_inventory_chart(self, query: str) -> Dict[str, Any]:
        """Create inventory-related visualizations"""
        try:
            df = inventory_master_df.copy()
            
            # Inventory levels by plant
            plant_inventory = df.groupby('PLANT_NAME')['UNRESTRICTED_STOCK'].sum().reset_index()
            
            fig = px.pie(
                plant_inventory, 
                values='UNRESTRICTED_STOCK', 
                names='PLANT_NAME',
                title='Inventory Distribution by Plant'
            )
            
            return {"type": "plotly", "chart": fig, "description": "Current inventory distribution across plants"}
            
        except Exception as e:
            return {"type": "error", "message": f"Error creating inventory chart: {str(e)}"}
    
    def _create_cost_chart(self, query: str) -> Dict[str, Any]:
        """Create cost-related visualizations"""
        try:
            if 'storage' in query.lower():
                df = storage_cost_df.copy()
                
                fig = px.bar(
                    df, 
                    x='PLANT_NAME', 
                    y='STORAGE_COST_PER_MT_DAY',
                    title='Storage Costs by Plant',
                    labels={'STORAGE_COST_PER_MT_DAY': 'Cost per MT per Day', 'PLANT_NAME': 'Plant'}
                )
                
            elif 'transfer' in query.lower():
                df = transfer_cost_df.copy()
                
                fig = px.bar(
                    df, 
                    x='MODE_OF_TRANSPORT', 
                    y='TRANSFER_COST_PER_CONTAINER',
                    title='Transfer Costs by Transport Mode',
                    labels={'TRANSFER_COST_PER_CONTAINER': 'Cost per Container', 'MODE_OF_TRANSPORT': 'Transport Mode'}
                )
                
            else:
                # Default storage costs
                df = storage_cost_df.copy()
                fig = px.bar(df, x='PLANT_NAME', y='STORAGE_COST_PER_MT_DAY', title='Storage Costs by Plant')
            
            fig.update_layout(xaxis_tickangle=-45)
            return {"type": "plotly", "chart": fig, "description": "Cost analysis visualization"}
            
        except Exception as e:
            return {"type": "error", "message": f"Error creating cost chart: {str(e)}"}
    
    def _create_general_chart(self, query: str) -> Dict[str, Any]:
        """Create general visualizations"""
        try:
            # Default: transaction type distribution
            df = transactions_master_df['TRANSACTION_TYPE'].value_counts().reset_index()
            df.columns = ['TRANSACTION_TYPE', 'COUNT']
            
            fig = px.pie(
                df, 
                values='COUNT', 
                names='TRANSACTION_TYPE',
                title='Transaction Type Distribution'
            )
            
            return {"type": "plotly", "chart": fig, "description": "General overview of transaction types"}
            
        except Exception as e:
            return {"type": "error", "message": f"Error creating general chart: {str(e)}"}
    
    def _setup_agent(self):
        """Setup the LangGraph agent"""
        agent_node = create_react_agent(
            model=self.llm,
            tools=[self.analyze_tool, self.visualize_tool]
        )
        
        graph = StateGraph(AgentState)
        graph.add_node("agent", agent_node)
        graph.set_entry_point("agent")
        graph.add_edge("agent", END)
        
        self.app = graph.compile()
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process user query and return appropriate response"""
        start_time = time.time()
        log_query(query)
        
        try:
            # Determine if this is a visualization request
            visualization_keywords = ['plot', 'chart', 'graph', 'show', 'visualize', 'display']
            is_viz_request = any(keyword in query.lower() for keyword in visualization_keywords)
            
            if is_viz_request:
                viz_result = self.visualize_tool.invoke(query)
                if viz_result["type"] == "plotly":
                    response = {
                        "type": "text_with_chart",
                        "text": f"Here's a visualization based on your request: {viz_result['description']}",
                        "chart": viz_result["chart"]
                    }
                    processing_time = time.time() - start_time
                    log_agent_response("text_with_chart", len(response["text"]), processing_time)
                    return response
                else:
                    error_msg = viz_result.get("message", "Error creating visualization")
                    processing_time = time.time() - start_time
                    log_agent_response("error", len(error_msg), processing_time)
                    return {"type": "error", "content": error_msg}
            
            # For data analysis questions
            else:
                response = self.app.invoke({
                    "messages": [("user", query)],
                    "last_query": query,
                    "context": {}
                })
                
                final_answer = response["messages"][-1].content
                processing_time = time.time() - start_time
                log_agent_response("text", len(final_answer), processing_time)
                
                return {"type": "text", "content": final_answer}
                
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error processing query: {str(e)}"
            log_error(e, {"query": query, "processing_time": processing_time})
            log_agent_response("error", len(error_msg), processing_time)
            return {"type": "error", "content": error_msg}