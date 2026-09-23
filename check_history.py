# check_history.py
import duckdb
import json

conn = duckdb.connect("./tiktok.duckdb")
rows = conn.execute("""
SELECT report_id, run_id, create_at, eval_score_json, length(industry_insight_markdown)
FROM dws_agent_report
ORDER BY create_at DESC
""").fetchall()

print("===== Agent历史运行汇总 =====")
for row in rows:
    rid, runid, ctime, score_str, insight_len = row
    print(f"\nreport_id:{rid}, run_id:{runid}, time:{ctime}")
    if score_str:
        s = json.loads(score_str)
        print(f"数据保真:{s.get('data_fidelity')}, 洞察价值:{s.get('insight_value')}, 策略可行性:{s.get('strategy_feasibility')}, 逻辑一致性:{s.get('logic_consistency')}")
        print(f"评语：{s.get('comment')}")
    else:
        print("无评估分数")
conn.close()

