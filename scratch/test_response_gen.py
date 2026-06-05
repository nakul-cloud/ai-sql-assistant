import sys
from llm.response_generator import generate_natural_language_response

if __name__ == "__main__":
    print("\n--- Testing Response Generator ---")
    mock_result = {
        "columns": ["employee_name", "salary"],
        "rows": [
            {"employee_name": "Tony Stark", "salary": 250000},
            {"employee_name": "Sam Spade", "salary": 105000}
        ],
        "row_count": 2
    }
    try:
        res = generate_natural_language_response(
            user_query="Who has the highest salary?",
            sql_query="SELECT employee_name, salary FROM dbo.csv_employees ORDER BY salary DESC;",
            query_result=mock_result
        )
        print("Result:", res)
    except Exception as e:
        print("Exception:", e)
