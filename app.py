import streamlit as st
import pandas as pd
from datetime import datetime # <-- NEW IMPORT FOR PHASE 2!
import io
import matplotlib.pyplot as plt # <-- NEW IMPORT for plotting!
import seaborn as sns # <-- NEW IMPORT for nicer plots (optional, but good practice)!
import numpy as np # <-- NEW IMPORT for numerical operations!
# Set the title of the Streamlit app

st.title("Retail Stock & Expiry Analyzer")
st.sidebar.header("App Controls & Assumptions")

st.write("Upload your stock data CSV file below.")
days_buffer = st.sidebar.slider(
    "1. Days Buffer for 'Expiring Soon' (days):",
    min_value=0, max_value=90, value=30, step=1,
    help="Items expiring within this many days will be marked 'Expiring Soon'." # <--- Include this 'help'
)

assumed_daily_sales_rate = st.sidebar.slider(
    "2. Override daily sales rate (for 'what-if' scenarios or products with no history):",
    min_value=0.1, max_value=10.0, value=1.0, step=0.1,
    help="This rate is used for products without historical sales data or for manual overrides." # <--- Include this 'help'
)

lead_time_days = st.sidebar.slider(
    "3. Supplier Lead Time (days - time from order to delivery):",
    min_value=1, max_value=30, value=7, step=1,
    help="How many days it takes for new stock to arrive after placing an order." # <--- Include this 'help'
)

safety_stock_units = st.sidebar.slider(
    "4. Safety Stock (units - buffer inventory to prevent stockouts):",
    min_value=0, max_value=50, value=10, step=1,
    help="Extra units to keep on hand to guard against unexpected demand or delays." # <--- Include this 'help'
)
# --- END SIDEBAR DEFINITION ---

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:

     # --- NEW DEBUGGING CODE GOES HERE ---
    st.write("### Debugging File Content (Please share this output if error persists):")
    try:
        # The uploaded_file is an in-memory file object.
        # .getvalue() reads its entire content as bytes.
        # .decode('utf-8') converts those bytes into a readable string (assuming UTF-8 encoding).
        file_content = uploaded_file.getvalue().decode("utf-8")

        st.write(f"Content length: {len(file_content)} characters")
        st.write("First 500 characters of file content:")
        st.text(file_content[:500]) # st.text displays raw text, good for code/file content

        # IMPORTANT: After reading the content with .getvalue(), the "pointer" is at the end of the file.
        # We need to reset it to the beginning (0) so that pd.read_csv() can read it from the start.
        uploaded_file.seek(0)
        st.success("File content read successfully for debugging.")

    except Exception as e:
        st.error(f"Error reading file content for debugging: {e}")
        st.info("Please ensure the file is a valid text-based CSV.")
        st.stop() # Stop the Streamlit app if we can't even debug the content
    # --- END OF NEW DEBUGGING CODE ---

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(io.StringIO(file_content))

    st.write("### Uploaded Data Preview:")
    st.dataframe(df.head())

    st.write(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns.")
    st.write("---") # Draw a separating line

    st.write("### Data Cleaning & Preparation:")

    # 1. Convert 'ExpiryDate' column to actual date format
    #    We assume the column is named 'ExpiryDate'. Adjust if yours is different.
    if 'ExpiryDate' in df.columns:
        # pd.to_datetime tries to understand and convert text dates into proper date objects
        df['ExpiryDate'] = pd.to_datetime(df['ExpiryDate'], errors='coerce')
        # 'errors='coerce'' means if a date can't be understood, it turns it into 'NaT' (Not a Time)
        st.success("✅ 'ExpiryDate' column converted to date format.")

        # Handle missing Expiry Dates: Fill with a future date or drop rows
        # For now, let's just drop rows where ExpiryDate couldn't be converted or was missing
        df.dropna(subset=['ExpiryDate'], inplace=True)
        st.info("ℹ️ Rows with missing or invalid 'ExpiryDate' have been removed.")

    else:
        st.warning("⚠️ Warning: 'ExpiryDate' column not found. Expiry analysis might be limited.")
        # If no ExpiryDate, we might still proceed with stock analysis


    # 2. Ensure 'Stock' or 'StockQuantity' is a number and handle missing values
    #    We assume a column like 'StockQuantity'. Adjust if yours is different.
    stock_col_found = False
    for col in ['Stock', 'StockQuantity', 'Quantity']: # Try common names
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce') # Convert to numbers, turn errors into NaN
            df.dropna(subset=[col], inplace=True) # Remove rows where stock isn't a valid number
            st.success(f"✅ '{col}' column cleaned and converted to numbers.")
            stock_col_found = True
            break # Stop checking after finding one
    if not stock_col_found:
        st.warning("⚠️ Warning: Common stock quantity column (e.g., 'Stock', 'StockQuantity', 'Quantity') not found or cleaned.")


    st.write("### Cleaned Data Info:")
    # This shows a summary of columns and their data types after cleaning
    st.dataframe(df.info()) # Note: df.info() returns None, so this won't display well in Streamlit.
                            # We'll adapt this. Let's just show columns and dtypes for now.
    st.write("---")
    st.write("#### Final Data Types after Cleaning:")
    st.dataframe(df.dtypes.rename('Data Type'))


    st.success("Data cleaning and preparation complete!")
    st.write("Ready for analysis and insights.")

if uploaded_file is not None:
    # --- EXISTING CODE: Read CSV and Initial Preview ---
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview:")
    st.dataframe(df.head())
    st.write(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns.")
    st.write("---")

    # --- EXISTING CODE: Data Cleaning & Preparation ---
    st.write("### Data Cleaning & Preparation:")

    if 'ExpiryDate' in df.columns:
        df['ExpiryDate'] = pd.to_datetime(df['ExpiryDate'], errors='coerce')
        st.success("✅ 'ExpiryDate' column converted to date format.")
        df.dropna(subset=['ExpiryDate'], inplace=True)
        st.info("ℹ️ Rows with missing or invalid 'ExpiryDate' have been removed.")
    else:
        st.warning("⚠️ Warning: 'ExpiryDate' column not found. Expiry analysis might be limited.")

    stock_col_found = False
    for col in ['Stock', 'StockQuantity', 'Quantity']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(subset=[col], inplace=True)
            st.success(f"✅ '{col}' column cleaned and converted to numbers.")
            stock_col_found = True
            break
    if not stock_col_found:
        st.warning("⚠️ Warning: Common stock quantity column (e.g., 'Stock', 'StockQuantity', 'Quantity') not found or cleaned.")

    st.write("### Cleaned Data Info:")
    st.write("#### Final Data Types after Cleaning:")
    st.dataframe(df.dtypes.rename('Data Type'))

    st.success("Data cleaning and preparation complete!")
    st.write("Ready for analysis and insights.")

    # --- NEW CODE FOR PHASE 2 GOES STARTING HERE! ---
    st.write("---") # Another separator

    st.write("### Phase 2: Core Analysis - Expiry Insights")

    today = datetime.now().date()
    today_pd = pd.to_datetime(today)

    if 'ExpiryDate' in df.columns:
        df['DaysToExpiry'] = (df['ExpiryDate'] - today_pd).dt.days
        st.success(f"✅ 'DaysToExpiry' column calculated based on today's date ({today.strftime('%Y-%m-%d')}).")

        def get_expiry_status(days_left):
            if days_left <= 0:
                return 'Expired'
            elif days_left <= 30:
                return 'Expiring Soon'
            elif days_left <= 90:
                return 'Expiring in 1-3 Months'
            else:
                return 'Long Shelf Life'

        df['ExpiryStatus'] = df['DaysToExpiry'].apply(get_expiry_status)
        st.success("✅ 'ExpiryStatus' categories (Expired, Expiring Soon, etc.) created.")

        st.write("#### Data with New Expiry Insights:")
        st.dataframe(df[['ProductName', 'ExpiryDate', 'DaysToExpiry', 'ExpiryStatus', 'StockQuantity']].head(10))
        st.write("---")
    else:
        st.warning("⚠️ Cannot perform expiry analysis: 'ExpiryDate' column was not found.")

    st.write("### Core Analysis Complete!")
    # --- END OF NEW CODE FOR PHASE 2 ---

    st.write("### Next Steps:")
    st.write("1. **Visualize Expiry Trends**: Create charts to see how many products are expiring soon.")
    st.write("### Phase 2: Visualizations & Summaries")

    if 'ExpiryStatus' in df.columns and 'StockQuantity' in df.columns:
        # 1. Summary of Expiry Status Counts
        st.write("#### 1. Stock Status Overview:")
        # Group by ExpiryStatus and sum the StockQuantity for each status
        # .reset_index() turns the grouped result back into a DataFrame for display
        expiry_summary = df.groupby('ExpiryStatus')['StockQuantity'].sum().reset_index()
        st.dataframe(expiry_summary)

        # 2. Visualization: Bar Chart of Stock by Expiry Status
        st.write("#### 2. Visualizing Stock by Expiry Status:")
        # Create a figure and an axes object for the plot
        fig, ax = plt.subplots()
        # Use seaborn to create a bar plot
        sns.barplot(x='ExpiryStatus', y='StockQuantity', data=expiry_summary, ax=ax, palette='viridis')
        ax.set_title('Total Stock Quantity by Expiry Status')
        ax.set_xlabel('Expiry Status')
        ax.set_ylabel('Total Stock Quantity')
        plt.xticks(rotation=45, ha='right') # Rotate labels for readability
        plt.tight_layout() # Adjust layout to prevent labels overlapping
        st.pyplot(fig) # Display the plot in Streamlit

        # 3. Critical Items Table (Expired and Expiring Soon)
        st.write("#### 3. Critical Stock Items (Expired & Expiring Soon):")
        critical_items = df[df['ExpiryStatus'].isin(['Expired', 'Expiring Soon'])].sort_values(by='DaysToExpiry')
        if not critical_items.empty:
            st.dataframe(critical_items[['ProductName', 'StockQuantity', 'ExpiryDate', 'DaysToExpiry', 'ExpiryStatus']])
            st.info(f"Found {len(critical_items)} critical items.")
        else:
            st.info("No items currently 'Expired' or 'Expiring Soon'. Great job!")

    else:
        st.warning("⚠️ Cannot generate detailed summaries: 'ExpiryStatus' or 'StockQuantity' column missing.")

        # ... (All your existing code for title, upload, data cleaning, and Phase 2 analysis/visualizations) ...

    st.write("### All Core Analysis & Visualizations Complete!")
    st.write("---") # Separator for new section

    # --- NEW CODE FOR PHASE 3 GOES HERE! ---
     # --- NEW PHASE 3 PREP: Load and Process Sales History ---
    st.write("### Preparing Historical Sales Data")
    st.info("Attempting to load `sales_history.csv` for advanced predictions...")

    try:
        # Load the sales history data directly from the file
        sales_df = pd.read_csv('sales_history.csv')

        # Convert 'Date' column to datetime objects
        sales_df['Date'] = pd.to_datetime(sales_df['Date'])

        # Calculate the number of days covered by the sales history
        # (This is a simplified approach for basic average daily sales)
        start_date = sales_df['Date'].min()
        end_date = sales_df['Date'].max()
        total_days_in_history = (end_date - start_date).days + 1 # +1 to include both start and end day

        # Calculate total units sold per product
        total_sales_per_product = sales_df.groupby('ProductID')['UnitsSold'].sum().reset_index()

        # Calculate average daily sales per product
        # Ensure total_days_in_history is not zero to prevent division by zero
        if total_days_in_history > 0:
            total_sales_per_product['AvgDailySales'] = total_sales_per_product['UnitsSold'] / total_days_in_history
        else:
            total_sales_per_product['AvgDailySales'] = 0 # Default to 0 if no days in history

        # Merge this average daily sales back into the main DataFrame (df)
        # We'll use a left merge to keep all products from our main df,
        # and add AvgDailySales where ProductIDs match.
        # If a product has no sales history, its AvgDailySales will be NaN (Not a Number)
        df = pd.merge(df, total_sales_per_product[['ProductID', 'AvgDailySales']], on='ProductID', how='left')

        # Fill NaN values (products with no sales history) with a default, e.g., 0 or a very small number
        df['AvgDailySales'] = df['AvgDailySales'].fillna(0) # or some assumed minimum sales rate

        st.success("✅ `sales_history.csv` loaded and `AvgDailySales` calculated and merged.")
        st.write("#### Sales History Summary:")
        st.dataframe(total_sales_per_product) # Show the calculated averages

    except FileNotFoundError:
        st.error("⚠️ `sales_history.csv` not found in the same directory as `app.py`. Cannot perform advanced sales predictions.")
        st.info("Please create `sales_history.csv` as instructed and place it next to your `app.py` file.")
        # If sales history is crucial, you might stop here, but for now, let app continue with 0 sales
        df['AvgDailySales'] = 0 # Ensure column exists even if file not found
    except Exception as e:
        st.error(f"❌ Error processing `sales_history.csv`: {e}")
        df['AvgDailySales'] = 0 # Ensure column exists even if error

    # --- END NEW PHASE 3 PREP ---

    st.write("### Phase 3: Smart Predictions - Expiry & Sales Forecasting")

    # 1. Allow user to input an assumed daily sales rate
    st.write("#### 1. Assume Daily Sales Rate for Risk Prediction")
    # This creates a number input slider in Streamlit
    
    
    st.info("💡 **Risk Prediction based on Historical Sales Data (where available)!**")
    st.info("For products with no sales history, the 'Override daily sales rate' above will be used as a fallback.")

    
    st.info(f"Using an assumed daily sales rate of **{assumed_daily_sales_rate} units/day** for predictions.")

    if 'DaysToExpiry' in df.columns and 'StockQuantity' in df.columns:
        # 2. Calculate "Units Expected to Sell Before Expiry"
        # This is a new calculated column
        df['EffectiveDailySalesRate'] = df['AvgDailySales'].replace(0, assumed_daily_sales_rate)
        df['ExpectedSalesBeforeExpiry'] = df['DaysToExpiry'] * df['EffectiveDailySalesRate']
        # Ensure it's not negative if DaysToExpiry is negative (already expired)
        df['ExpectedSalesBeforeExpiry'] = df['ExpectedSalesBeforeExpiry'].apply(lambda x: max(0, x))

        # 3. Predict "Risk of Expiry Before Sale"
        # If StockQuantity is greater than ExpectedSalesBeforeExpiry, it's at risk.
        # We also consider items already 'Expired' as inherently at risk.
        df['RiskOfExpiryBeforeSale'] = df.apply(
            lambda row: 'High Risk' if row['ExpiryStatus'] == 'Expired' or \
                                      row['StockQuantity'] > row['ExpectedSalesBeforeExpiry'] else 'Low Risk',
            axis=1 # Important: apply the function row by row
        )
        st.success("✅ 'ExpectedSalesBeforeExpiry' and 'RiskOfExpiryBeforeSale' calculated.")

        st.write("#### 2. Products with Risk Predictions:")
        # Display relevant columns including the new prediction
        # Filter for 'High Risk' items or show a sample of all items
        high_risk_items = df[df['RiskOfExpiryBeforeSale'] == 'High Risk'].sort_values(by='DaysToExpiry')

        if not high_risk_items.empty:
            st.warning(f"🚨 Heads up! Found **{len(high_risk_items)} products** currently at **High Risk** of Expiry Before Sale based on the assumed sales rate. Below is the expiry risk prediction for **all** your products:")
            st.dataframe(df[['ProductName', 'StockQuantity', 'ExpiryDate', 'DaysToExpiry', 'ExpiryStatus', 'ExpectedSalesBeforeExpiry', 'RiskOfExpiryBeforeSale']])
        else:
           st.info("🎉 Great news! No products currently at High Risk of Expiry Before Sale based on the assumed sales rate. Below is the expiry risk prediction for **all** your products:")



        # You could add more visualizations for risk here later!

    st.write("### All Core Analysis & Visualizations Complete!")
    
    # --- NEW PHASE 4: Reorder Point Recommendations START ---
    st.write("### Phase 4: Advanced Insights - Reorder Point Recommendations")

    st.info("💡 Calculate your ideal reorder points based on sales velocity, supplier lead times, and desired safety stock levels.")

    # User inputs for Lead Time and Safety Stock
    # Lead Time: How many days it takes for new stock to arrive after ordering
    

    # Safety Stock: A buffer to prevent stockouts due to unexpected demand or delays
    

    # Calculate Reorder Point for each product
    # Formula: (AvgDailySales * Lead Time) + Safety Stock
    # We use np.ceil to round up to the next whole unit, as you can't reorder partial units.
    # .astype(int) converts the float to an integer for cleaner display.
    df['ReorderPoint'] = (df['AvgDailySales'] * lead_time_days) + safety_stock_units
    df['ReorderPoint'] = np.ceil(df['ReorderPoint']).astype(int)

    # Determine Reorder Recommendation based on current Stock Quantity vs. Reorder Point
    df['ReorderRecommendation'] = np.where(
        df['StockQuantity'] <= df['ReorderPoint'],
        '🚨 Reorder Now!', # If current stock is at or below the reorder point
        '✅ In Stock'      # If current stock is above the reorder point
    )

    # Display Reorder Recommendations
    reorder_items = df[df['ReorderRecommendation'] == '🚨 Reorder Now!'].sort_values(by='ReorderPoint')

    if not reorder_items.empty:
        st.warning(f"📣 **Action Required!** Found {len(reorder_items)} products currently needing reorder:")
        st.dataframe(reorder_items[['ProductName', 'StockQuantity', 'AvgDailySales', 'ReorderPoint', 'ReorderRecommendation']])
    else:
        st.info("🎉 Good news! No products currently need reordering based on your current stock and settings. Keep up the good work!")

    st.write("#### All Products Reorder Status:")
    st.dataframe(df[['ProductName', 'StockQuantity', 'AvgDailySales', 'ReorderPoint', 'ReorderRecommendation']].sort_values(by='ReorderRecommendation', ascending=False))

    st.write("### Phase 4: Advanced Insights Complete!")
    st.write("---") # Separator
    # --- NEW PHASE 4: Reorder Point Recommendations END ---

else: # This 'else' belongs to the very first 'if uploaded_file is not None'
    st.info("Waiting for you to upload a CSV file to begin analysis.")
    

# Note: This code assumes the CSV has columns like 'ExpiryDate', 'Stock', etc.
# Adjust the column names based on your actual CSV structure.   
    
    