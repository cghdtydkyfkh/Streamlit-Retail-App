# 🛍️ Retail Stock & Expiry Analyzer

### A Smart, Data-Driven Inventory Management & Forecasting Tool for Small Businesses

---

## 💡 Project Overview

In the fast-paced world of retail, efficient inventory management is crucial for profitability and customer satisfaction. This Streamlit-powered application, the **Retail Stock & Expiry Analyzer**, is designed to empower small business owners and inventory managers with intelligent insights to optimize stock levels, minimize waste from expired goods, and prevent costly stockouts.

Built with a focus on ease-of-use and actionable recommendations, this tool transforms raw stock data into clear, data-driven decisions.

## ✨ Key Features

* **Intelligent Expiry Tracking:** Automatically identifies expired and soon-to-expire items, providing immediate clarity on inventory freshness.
* **AI-Powered Risk Prediction:** Goes beyond simple expiry dates. By incorporating historical sales data, it predicts which products are at "High Risk of Expiry Before Sale," allowing proactive measures to be taken. This utilizes a dynamic sales rate (historical average or user-defined override) for precise predictions.
* **Smart Reorder Point Recommendations:** Calculates optimal reorder points for each product based on its average daily sales, supplier lead time, and a user-defined safety stock buffer. This helps prevent stockouts while avoiding over-stocking.
* **Intuitive & Interactive User Interface:** A clean, responsive dashboard built with Streamlit, featuring:
    * **Sidebar Controls:** All key parameters (expiry buffer, sales override, lead time, safety stock) are neatly organized in a sidebar for easy adjustment.
    * **Expandable Sections:** Large data tables are hidden behind expanders, keeping the main view concise and focused on key insights.
    * **Flexible Data Input:** Supports CSV file uploads (and planned/implemented Excel file uploads).
    * **Open-Source & Customizable:** Developed in Python, making it accessible for further customization and integration.

## 🚀 How It Works (High-Level Data Flow)

1.  **Data Upload:** The user uploads their `stock_data.csv` (or `.xlsx`) file directly into the app.
2.  **Historical Sales Integration:** The app automatically loads `sales_history.csv` (expected to be in the same directory) to calculate `AvgDailySales` for each product.
3.  **Data Processing & Cleaning:** Raw stock data is cleaned, converted to appropriate data types, and duplicates are handled.
4.  **Core Analysis:** `DaysToExpiry` and `ExpiryStatus` are calculated, and a summary of stock health is presented.
5.  **Smart Prediction:**
    * `ExpectedSalesBeforeExpiry` is calculated using `DaysToExpiry` and the `AvgDailySales` (or a fallback override rate).
    * `RiskOfExpiryBeforeSale` is determined by comparing `StockQuantity` with `ExpectedSalesBeforeExpiry`.
6.  **Advanced Insights:**
    * User inputs for `Lead Time` and `Safety Stock` are gathered.
    * `ReorderPoint` is calculated using `AvgDailySales`, `Lead Time`, and `Safety Stock`.
    * `ReorderRecommendation` (🚨 Reorder Now! or ✅ In Stock) is provided based on `StockQuantity` vs. `ReorderPoint`.
7.  **Interactive Display:** All insights, predictions, and recommendations are presented in an intuitive dashboard format, allowing users to adjust parameters and see real-time impacts.

## 🛠️ Technical Stack

* **Python:** The core programming language.
* **Streamlit:** For building the interactive web application.
* **Pandas:** Essential for data manipulation, cleaning, and analysis.
* **NumPy:** For efficient numerical operations.
* **Matplotlib & Seaborn:** For data visualization.
* (`openpyxl`: Required if Excel (.xlsx) file upload functionality is implemented.)

## 📦 How to Run Locally

To run this application on your local machine, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/cghdtydkyfkh/Streamlit-Retail-App.git]()]
    cd Streamlit-Retail-App
    ```

2.  **Set up Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare Data Files:**
    * Ensure `stock_data.csv` (or your preferred stock data file) is in the same directory as `app.py`.
    * Ensure `sales_history.csv` (containing historical sales data, as provided in the project steps) is also in the same directory.

5.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```
    This will open the application in your default web browser.

## 🌐 Live Demo

Experience the Retail Stock & Expiry Analyzer live on Streamlit Community Cloud:

[Live Demo Link ()](https://app-retail-app-lnwyvyjrjxzkkogkryfht8.streamlit.app) 

## 🚀 Future Enhancements

* **More Advanced Sales Forecasting:** Implement time-series models (e.g., ARIMA, Prophet) for more robust sales predictions.
* **Database Integration:** Connect directly to retail databases (SQL, NoSQL) for real-time data access instead of file uploads.
* **Multi-User Support:** Implement user authentication and personalized dashboards.
* **Dynamic Safety Stock Calculation:** Incorporate variability in demand and lead time to dynamically calculate optimal safety stock levels.
* **Supplier Management:** Add features for tracking supplier details, order history, and performance.

---

## 📧 Contact

**NEHAL**


---
