import sys
from workflow.process_query import process_user_query

def main():
    query = "Show me employees in the engineering department"
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        
    print(f"Executing Query: '{query}'\n")
    res = process_user_query(query)
    
    print("\n" + "=" * 80)
    print("FINAL RESPONSE:")
    print("=" * 80)
    print(f"Intent        : {res.get('intent')}")
    print(f"Cache Hit     : {res.get('cache_hit', False)}")
    print(f"Generated SQL : {res.get('generated_sql')}")
    print(f"NL Response   : {res.get('nl_response')}")
    print("=" * 80)

if __name__ == "__main__":
    main()
