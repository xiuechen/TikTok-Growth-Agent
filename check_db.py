import duckdb

conn = duckdb.connect("./tiktok.duckdb")
print("=== dwd_agent_run_log ===")
print(conn.execute("SELECT * FROM dwd_agent_run_log").fetchall())
print("\n=== dws_agent_report ===")
print(conn.execute("SELECT * FROM dws_agent_report").fetchall())
conn.close()
