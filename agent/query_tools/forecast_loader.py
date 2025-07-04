import pandas as pd
import json
from datetime import datetime
import openpyxl

def parse_warehouse_data(df, warehouse_name, capacity_row, outbound_row, inventory_row, date_columns):
    """
    Helper function to parse warehouse data.
    """
    warehouse_data = {
        'warehouse': warehouse_name,
        'capacity': [],
        'predicted_outbound': [],
        'predicted_inventory': [],
        'dates': date_columns
    }
    
    # Extract capacity
    for col in range(2, 2 + len(date_columns)):
        if col < len(df.columns) and pd.notna(df.iloc[capacity_row, col]):
            warehouse_data['capacity'].append(float(df.iloc[capacity_row, col]))
        else:
            warehouse_data['capacity'].append(None)
    
    # Extract predicted outbound
    for col in range(2, 2 + len(date_columns)):
        if col < len(df.columns) and pd.notna(df.iloc[outbound_row, col]):
            warehouse_data['predicted_outbound'].append(float(df.iloc[outbound_row, col]))
        else:
            warehouse_data['predicted_outbound'].append(None)
    
    # Extract predicted inventory
    for col in range(2, 2 + len(date_columns)):
        if col < len(df.columns) and pd.notna(df.iloc[inventory_row, col]):
            warehouse_data['predicted_inventory'].append(float(df.iloc[inventory_row, col]))
        else:
            warehouse_data['predicted_inventory'].append(None)
    
    return warehouse_data

def read_forecast_excel(file_path):
    """
    Read the forecast Excel file and convert it to LLM-friendly format
    """
    # Read the Excel file
    df = pd.read_excel(file_path, header=None)
    
    # Extract date headers (row 3, columns 2 onwards)
    date_columns = []
    for col in range(2, len(df.columns)):
        if pd.notna(df.iloc[2, col]):
            # Convert to readable date format
            date_val = pd.to_datetime(df.iloc[2, col])
            date_columns.append(date_val.strftime('%Y-%m'))
    
    # Structure to hold the parsed data
    warehouses = {}
    
    # Parse Singapore data
    warehouses['SINGAPORE'] = parse_warehouse_data(df, 'SINGAPORE', 1, 5, 6, date_columns)
    
    # Parse China data
    warehouses['CHINA'] = parse_warehouse_data(df, 'CHINA', 13, 17, 18, date_columns)

    return warehouses

def format_for_llm(warehouses_data):
    """
    Format the data in multiple LLM-friendly formats
    """
    
    # Format 1: Structured JSON
    # print("=== JSON FORMAT ===")
    # print(json.dumps(warehouses_data, indent=2))
    
    # Format 2: Tabular CSV-like format
    # print("\n=== TABULAR FORMAT ===")
    # Create a list to collect all rows
    all_rows = []
    
    for warehouse_name, data in warehouses_data.items():
        # print(f"\n{warehouse_name} WAREHOUSE:")
        # print("Date,Capacity_KT,Predicted_Outbound_KT,Predicted_Inventory_KT")
        
        for i, date in enumerate(data['dates']):
            capacity = data['capacity'][i] if i < len(data['capacity']) else None
            outbound = data['predicted_outbound'][i] if i < len(data['predicted_outbound']) else None
            inventory = data['predicted_inventory'][i] if i < len(data['predicted_inventory']) else None
            
            # Add to DataFrame collection
            all_rows.append({
                'Warehouse': warehouse_name,
                'Date': date,
                'Capacity_KT': capacity,
                'Predicted_Outbound_KT': outbound,
                'Predicted_Inventory_KT': inventory
            })
            
            # Print for display
            capacity_str = capacity if capacity is not None else 'N/A'
            outbound_str = outbound if outbound is not None else 'N/A'
            inventory_str = inventory if inventory is not None else 'N/A'
            # print(f"{date},{capacity_str},{outbound_str},{inventory_str}")
    
    # Create DataFrame from collected data
    tabular_df = pd.DataFrame(all_rows)
    # print(f"\nCreated DataFrame with {len(tabular_df)} rows")
    
    # Format 3: Summary statistics
    # print("\n=== SUMMARY STATISTICS ===")
    # for warehouse_name, data in warehouses_data.items():
    #     print(f"\n{warehouse_name} WAREHOUSE SUMMARY:")
        
    #     # Capacity stats
    #     capacity_values = [x for x in data['capacity'] if x is not None]
    #     if capacity_values:
    #         print(f"  Capacity: Min={min(capacity_values):.1f} KT, Max={max(capacity_values):.1f} KT, Avg={sum(capacity_values)/len(capacity_values):.1f} KT")
        
    #     # Outbound stats
    #     outbound_values = [x for x in data['predicted_outbound'] if x is not None]
    #     if outbound_values:
    #         print(f"  Outbound: Min={min(outbound_values):.1f} KT, Max={max(outbound_values):.1f} KT, Avg={sum(outbound_values)/len(outbound_values):.1f} KT")
    #         print(f"  Total Annual Outbound: {sum(outbound_values):.1f} KT")
        
    #     # Inventory stats
    #     inventory_values = [x for x in data['predicted_inventory'] if x is not None]
    #     if inventory_values:
    #         print(f"  Inventory: Min={min(inventory_values):.1f} KT, Max={max(inventory_values):.1f} KT, Avg={sum(inventory_values)/len(inventory_values):.1f} KT")
    
    return tabular_df

def save_to_csv(warehouses_data, output_file='forecast_data.csv'):
    """
    Save data to CSV format for easy analysis
    """
    all_data = []
    
    for warehouse_name, data in warehouses_data.items():
        for i, date in enumerate(data['dates']):
            row = {
                'Warehouse': warehouse_name,
                'Date': date,
                'Capacity_KT': data['capacity'][i] if i < len(data['capacity']) else None,
                'Predicted_Outbound_KT': data['predicted_outbound'][i] if i < len(data['predicted_outbound']) else None,
                'Predicted_Inventory_KT': data['predicted_inventory'][i] if i < len(data['predicted_inventory']) else None
            }
            all_data.append(row)
    
    df = pd.DataFrame(all_data)
    df.to_csv(output_file, index=False)
    print(f"\nData saved to {output_file}")
    return df

def extract_from_xlsx(file_path = './utils/data/Forecast.xlsx'):
    """
    Extract date headers from the Excel file.
    """
    warehouses_data = read_forecast_excel(file_path)
    df = format_for_llm(warehouses_data)
    
    return df

# Main execution
if __name__ == "__main__":
    # Replace 'Forecast.xlsx' with your file path
    file_path = './agent/query_tools/Forecast.xlsx'
    
    try:
        # Read and parse the Excel file
        warehouses_data = read_forecast_excel(file_path)
        
        # Display in LLM-friendly formats
        df = format_for_llm(warehouses_data)
        
        # Save to CSV for further analysis
        df = save_to_csv(warehouses_data)
        
        print("\n=== USAGE NOTES ===")
        print("1. The JSON format is perfect for LLM processing")
        print("2. The tabular format is easy to read and analyze")
        print("3. The CSV file can be imported into any analysis tool")
        print("4. All values are in KT (Kilotons)")
        print("5. Date format is YYYY-MM for easy sorting")
        
    except Exception as e:
        print(f"Error reading file: {e}")
        print("Make sure the file path is correct and the file format matches the expected structure.")