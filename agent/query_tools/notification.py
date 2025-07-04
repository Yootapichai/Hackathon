from forecast_loader import extract_from_xlsx

def warehouse_capacity():
    df = extract_from_xlsx()
    df = df.rename(columns={
        'Capacity (KT)': 'Capacity_KT',
        'Predicted Outbound (KT)': 'Predicted_Outbound_KT',
        'Predicted Inventory (KT)': 'Predicted_Inventory_KT'
    })
    return df

def calculate_capacity(df):
    if (df['Capacity_KT'] * 0.80 <= df['Predicted_Inventory_KT']).any(): 
        print(df[df['Capacity_KT'] * 0.80 <= df['Predicted_Inventory_KT']])
    else:
        print("All warehouses have sufficient capacity.")

if __name__ == "__main__":
    df = warehouse_capacity()
    calculate_capacity(df)
    # Uncomment the line below to save the DataFrame to a CSV file
    # df.to_csv('warehouse_capacity.csv', index=False)