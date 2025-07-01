import pandas as pd
import os

# Print the current working directory
# print("Current Directory:", os.getcwd())

# Load all the source files
material_df = pd.read_csv('utils/data/MaterialMaster.csv')
inventory_df = pd.read_csv('utils/data/Inventory.csv')
inbound_df = pd.read_csv('utils/data/Inbound.csv')
outbound_df = pd.read_csv('utils/data/Outbound.csv')
op_cost_df = pd.read_csv('utils/data/OperationCost.csv')
# forecast_df = pd.read_csv('Forecast.csv')


### Step 2: Prepare the Tricky OperationCost DataFrame

# Create a clean storage cost table
storage_cost_df = op_cost_df[op_cost_df['Operation'] == 'Inventory Storage per MT per day'].copy()
storage_cost_df = storage_cost_df.rename(columns={'Plant/Mode of Transport': 'PLANT_NAME'})
storage_cost_df = storage_cost_df[['PLANT_NAME', 'Cost', 'Currency']]
storage_cost_df.columns = ['PLANT_NAME', 'STORAGE_COST_PER_MT_DAY', 'STORAGE_COST_CURRENCY']

# Create a clean transfer cost table
transfer_cost_df = op_cost_df[op_cost_df['Operation'] == 'Transfer cost per container (24.75MT)'].copy()
transfer_cost_df = transfer_cost_df.rename(columns={'Plant/Mode of Transport': 'MODE_OF_TRANSPORT'})
transfer_cost_df = transfer_cost_df[['MODE_OF_TRANSPORT', 'Cost', 'Currency']]
transfer_cost_df.columns = ['MODE_OF_TRANSPORT', 'TRANSFER_COST_PER_CONTAINER', 'TRANSFER_COST_CURRENCY']

# Now you have two perfect lookup tables: storage_cost_df and transfer_cost_df

### Step 3: Create the transactions_master_df (The Core Join)
# Prepare inbound and outbound dataframes for concatenation
inbound_prep = inbound_df.copy()
inbound_prep['TRANSACTION_TYPE'] = 'INBOUND'
inbound_prep['TRANSACTION_DATE'] = pd.to_datetime(inbound_prep['INBOUND_DATE'])

outbound_prep = outbound_df.copy()
outbound_prep['TRANSACTION_TYPE'] = 'OUTBOUND'
outbound_prep['TRANSACTION_DATE'] = pd.to_datetime(outbound_prep['OUTBOUND_DATE'])

# Concatenate them into a single transaction log
transactions_df = pd.concat([inbound_prep, outbound_prep], ignore_index=True)

# *** THE KEY JOIN ***
# Merge the transaction log with the material master data.
# This adds POLYMER_TYPE, SHELF_LIFE, etc. to every transaction.
transactions_master_df = pd.merge(
    transactions_df,
    material_df,
    on='MATERIAL_NAME', # The common column
    how='left'         # 'left' keeps all transactions even if a material is missing from the master
)


### Step 4: Create the inventory_master_df
# *** THE SECOND KEY JOIN ***
# Enrich the inventory data with material master data
inventory_master_df = pd.merge(
    inventory_df,
    material_df,
    on='MATERIAL_NAME',
    how='left'
)

# Final dataframe: [transactions_master_df, inventory_master_df, storage_cost_df, transfer_cost_df]
