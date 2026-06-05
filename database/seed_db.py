"""
database/seed_db.py
───────────────────
Seeds the SQL Server database with three sample business tables:
  • dbo.csv_employees
  • dbo.csv_departments
  • dbo.csv_sales

This provides a rich schema environment to verify indexing and retrieval layers.

Usage:
    python -m database.seed_db
"""

import sys
import pandas as pd
from database.sql_server import get_engine

def seed_database():
    print("\n=======================================================")
    print("      [INFO] Seeding SQL Server Database")
    print("=======================================================")
    
    engine = get_engine()
    
    # 1. Departments data
    departments_df = pd.DataFrame({
        "department_id": [1, 2, 3, 4],
        "department_name": ["Engineering", "Sales", "Marketing", "Human Resources"],
        "manager_name": ["Alice Johnson", "Bob Smith", "Charlie Brown", "Diana Prince"],
        "location": ["New York", "San Francisco", "Chicago", "Austin"]
    })
    
    # 2. Employees data
    employees_df = pd.DataFrame({
        "employee_id": [101, 102, 103, 104, 105],
        "employee_name": ["John Doe", "Jane Doe", "Sam Spade", "Lucy Liu", "Tony Stark"],
        "department_id": [1, 1, 2, 3, 1],
        "salary": [95000, 105000, 72000, 68000, 250000],
        "hire_date": ["2022-01-15", "2021-06-01", "2023-03-10", "2024-01-10", "2020-05-12"]
    })
    
    # 3. Sales data
    sales_df = pd.DataFrame({
        "sale_id": [1001, 1002, 1003, 1004, 1005, 1006],
        "employee_id": [103, 103, 104, 103, 104, 105],
        "sale_amount": [1200.50, 450.00, 3200.00, 150.25, 850.00, 15000.00],
        "sale_date": ["2026-05-01", "2026-05-02", "2026-05-02", "2026-05-03", "2026-05-04", "2026-05-05"],
        "product_category": ["Software", "Hardware", "Consulting", "Hardware", "Consulting", "Software"]
    })
    
    try:
        # Write to SQL Server (replace if exists)
        print("[INFO] Writing [dbo].[csv_departments]...")
        departments_df.to_sql("csv_departments", con=engine, schema="dbo", if_exists="replace", index=False)
        print("  [OK] Wrote 4 departments.")
        
        print("[INFO] Writing [dbo].[csv_employees]...")
        employees_df.to_sql("csv_employees", con=engine, schema="dbo", if_exists="replace", index=False)
        print("  [OK] Wrote 5 employees.")
        
        print("[INFO] Writing [dbo].[csv_sales]...")
        sales_df.to_sql("csv_sales", con=engine, schema="dbo", if_exists="replace", index=False)
        print("  [OK] Wrote 6 sales transactions.")
        
        print("\n=======================================================")
        print("      [OK] Database Seeding Completed Successfully")
        print("=======================================================")
        
    except Exception as e:
        print(f"\n[FAIL] Database seeding failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    seed_database()
    sys.exit(0)
