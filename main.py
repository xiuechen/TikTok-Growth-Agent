import os
import pandas as pd
import duckdb
import json
from datetime import datetime
from dotenv import load_dotenv
from typing import Optional, Dict, List, Any
from langchain.callbacks.base import BaseCallbackHandler
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()



# ===================== 1.定义全局State =====================

class TikTokState(BaseModel):
    raw_samples: List[Dict[str, Any]] = Field(default_factory=list, description="原始视频样本列表")
    clean_samples: List[Dict[str, Any]] = Field(default_factory=list)
    parsed_contents: List[Dict[str, Any]] = Field(default_factory=list, description="LLM结构化解析结果")
    new_raw_batch: List[Dict[str, Any]] = Field(default_factory=list, description="本轮新拉取的一批原始样本（临时）")
    industry_insight: str = Field(default="", description="行业分析markdown报告")
    growth_suggestion: str = Field(default="", description="增长推广策略markdown")
    current_run_id: int = Field(default=0, description="阶段2新增：本次任务run id")
    sample_pass: bool = Field(default=True, description="样本校验是否通过") #新增
    stats_summary: str = Field(default="", description="样本量化统计摘要，用于反思校验")
    retry_count: int = Field(default=0, description="采样重试次数")
    max_retry: int = Field(default=3, description="最大重试上限，避免无限循环")
    eval_score: Optional[Dict] = None
    # 在TikTokState里面追加
    prompt_tokens_total: int = 0
    completion_tokens_total: int = 0
    token_detail: list[Dict[str, Any]] = []  # 存储每个节点的token明细


class TokenCountCallback(BaseCallbackHandler):
    def __init__(self, node_name: str):
        self.node_name = node_name
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def on_llm_end(self, response, **kwargs):
        usage = response.llm_output.get("token_usage", {})
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)


def save_run_log_node(state: TikTokState) -> Dict:
    conn = duckdb.connect("./tiktok.duckdb")
    # python侧生成run_id
    res = conn.execute("SELECT COALESCE(MAX(run_id),0)+1 FROM dwd_agent_run_log").fetchone()
    run_id = res[0]
    conn.execute("""
INSERT INTO dwd_agent_run_log(
    run_id, run_timestamp, sample_cnt, raw_samples_json, parsed_contents_json,
    llm_industry_raw, llm_growth_raw,
    prompt_tokens_total, completion_tokens_total, token_detail_json
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""", (
    run_id,
    datetime.now(),
    len(state.raw_samples),
    json.dumps(state.raw_samples, ensure_ascii=False),
    json.dumps(state.parsed_contents, ensure_ascii=False),
    state.industry_insight,
    state.growth_suggestion,
    state.prompt_tokens_total,
    state.completion_tokens_total,
    json.dumps(state.token_detail, ensure_ascii=False)
))

    print(f"\n📊 Token统计：")
    print(f"总输入token：{state.prompt_tokens_total}")
    print(f"总输出token：{state.completion_tokens_total}")
    # 简单成本估算，按gpt-3.5-turbo价格举例，你可以改成你用的模型单价
    cost = state.prompt_tokens_total / 1000 * 0.0015 + state.completion_tokens_total /1000 *0.002
    print(f"💸 本次估算成本：${cost:.4f}")

    #conn.execute("""
    #    INSERT INTO dwd_agent_run_log(
    #        run_id, sample_cnt, raw_samples_json, parsed_contents_json,
    #        llm_industry_raw, llm_growth_raw
    #    ) VALUES (?, ?, ?, ?, ?, ?)
    #""",[
    #    run_id,
    #    len(state.raw_samples),
    #    json.dumps(state.raw_samples, ensure_ascii=False),
    #    json.dumps(state.parsed_contents, ensure_ascii=False),
    #    state.industry_insight,
    #    state.growth_suggestion
    #])
    conn.close()
    return {"current_run_id": run_id}

def save_report_node(state: TikTokState) -> TikTokState:
    import json
    conn = duckdb.connect("./tiktok.duckdb")
    res = conn.execute("SELECT COALESCE(MAX(report_id),0)+1 FROM dws_agent_report").fetchone()
    report_id = res[0]

    # 序列化评估分数字典
    eval_score_json = json.dumps(state.eval_score, ensure_ascii=False) if state.eval_score else None

    conn.execute("""
        INSERT INTO dws_agent_report(
            report_id,
            run_id,
            industry_insight_markdown,
            growth_suggestion_markdown,
            eval_score_json
        )
        VALUES (?, ?, ?, ?, ?)
    """,[
        report_id,
        state.current_run_id,
        state.industry_insight,
        state.growth_suggestion,
        eval_score_json
    ])
    conn.close()
    return state


# LLM配置
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0.3
)

# ===================== 2.DuckDB 初始化，ODS层入库 =====================
def init_duckdb(csv_path: str):
    """读取csv，初始化ods_tiktok原始数据表"""
    conn = duckdb.connect("./tiktok.duckdb")
    df = pd.read_csv(csv_path)
    conn.execute("CREATE OR REPLACE TABLE ods_tiktok AS SELECT * FROM df;")
    cnt = conn.execute('select count(*) from ods_tiktok').fetchone()[0]
    print(f"DuckDB ODS层完成，共 {cnt} 条记录")
    conn.close()

def init_agent_tables():
    """阶段2新增：创建Agent运行日志、报告汇总表，复用 tiktok.duckdb"""
    conn = duckdb.connect("./tiktok.duckdb")
    # Agent运行日志
    conn.execute("""
    CREATE TABLE IF NOT EXISTS dwd_agent_run_log (
        run_id INTEGER,
        run_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        sample_cnt INTEGER,
        raw_samples_json TEXT,
        parsed_contents_json TEXT,
        llm_industry_raw TEXT,
        llm_growth_raw TEXT,
        prompt_tokens_total INTEGER,
        completion_tokens_total INTEGER,
        token_detail_json TEXT
    )
    """)
    # Agent报告表
    conn.execute("""
    CREATE TABLE IF NOT EXISTS dws_agent_report (
        report_id INTEGER,
        run_id INTEGER,
        industry_insight_markdown TEXT,
        growth_suggestion_markdown TEXT,
        eval_score_json TEXT,
        create_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    # 新增：TikTok清洗明细表 DWD层
    conn.execute("""
    CREATE TABLE IF NOT EXISTS dwd_tiktok_clean (
        video_id BIGINT,
        title TEXT,
        views INTEGER,
        likes INTEGER,
        shares INTEGER,
        comments INTEGER
    )
    """)
    conn.close()
    print("阶段2：数据表初始化完成")

def etl_ods_to_dwd():
    """清洗原始ods_tiktok，过滤脏数据写入dwd_tiktok_clean"""
    conn = duckdb.connect("./tiktok.duckdb")
    conn.execute("TRUNCATE TABLE dwd_tiktok_clean")
    conn.execute("""
    INSERT INTO dwd_tiktok_clean
    SELECT
        video_id,
        video_transcription_text as title,
        video_view_count as views,
        video_like_count as likes,
        video_share_count as shares,
        video_comment_count as comments
    FROM ods_tiktok
    WHERE video_view_count > 0
      AND video_transcription_text IS NOT NULL
      AND length(trim(video_transcription_text)) > 0
    """)
    total = conn.execute("SELECT count(*) FROM dwd_tiktok_clean").fetchone()[0]
    print(f"✅ETL完成，DWD清洗后有效样本：{total} 条")
    conn.close()


def load_sample_from_duckdb(sample_cnt:int=10) -> List[Dict[str,Any]]:
    """从ods层读取N条样本，返回字典列表给到Agent流水线"""
    conn = duckdb.connect("./tiktok.duckdb")
    df_sample = conn.execute(f"SELECT * FROM ods_tiktok LIMIT {sample_cnt}").df()
    conn.close()
    return df_sample.to_dict("records")

# ===================== 3.Agent节点定义 =====================
def sample_check_node(state: TikTokState) -> TikTokState:
    import pandas as pd
    df = pd.DataFrame(state.raw_samples)
    total_cnt = len(df)
    print(f"初始采样条数：{len(df)}")
    cond1 = ~df["video_transcription_text"].isna()
    print(f"转录文本非空：{cond1.sum()}")
    cond2 = df["video_view_count"] > 0
    print(f"播放>0：{cond2.sum()}")
    cond3 = df["video_duration_sec"] >=5
    print(f"时长>=5s：{cond3.sum()}")
    print("==== author_ban_status 唯一值 ====")
    print(df["author_ban_status"].unique())
    cond4 =  (df["author_ban_status"] == "active")
    print(f"账号未封禁：{cond4.sum()}")

    # ========== 新增多维度脏样本过滤 ==========
    df_clean = df[
        (~df["video_transcription_text"].isna()) &
        (df["video_view_count"] > 0) &
        (df["video_duration_sec"] >=5) &
        (df["author_ban_status"] == "active")
    ]
    clean_samples = df_clean.to_dict("records")
    clean_cnt = len(clean_samples)
    print(f"【样本校验】原始采样{total_cnt}条，清洗后{clean_cnt}条")

    sample_pass = False
    new_retry = state.retry_count

    if clean_cnt < 20:
        print(f"【样本校验失败】清洗后有效样本仅{clean_cnt}，不足20")
        sample_pass = False
        # 校验失败，重试计数+1
        new_retry = state.retry_count + 1
        stats_summary = ""
    else:
        sample_pass = True
        # 计算量化统计指标，传给反思节点
        views_median = df_clean["video_view_count"].median()
        avg_like_rate = (df_clean["video_like_count"] / df_clean["video_view_count"]).mean()
        stats_summary = f"""
样本统计摘要：
有效样本数：{clean_cnt}
播放量中位数：{views_median:.0f}
平均点赞率：{avg_like_rate:.2%}
视频时长均值：{df_clean["video_duration_sec"].mean():.1f}秒
认证账号占比：{(df_clean['verified_status'].notna()).mean():.2%}
"""
        print(stats_summary)

    # ✅ 重点：raw_samples保持原样，不覆盖！干净样本放到clean_samples字段
    return state.model_copy(update={
        "clean_samples": clean_samples,
        "sample_pass": sample_pass,
        "stats_summary": stats_summary,
        "retry_count": new_retry
    })

def evaluation_node(state: TikTokState) -> TikTokState:
    """评估打分节点：对洞察+增长策略做结构化自评，输出JSON指标"""
    prompt = """
你是数据评审专家。参考【样本统计摘要】【修正后的行业洞察】【TikTok增长策略】，4个维度1~5打分，只输出JSON，无多余文字。
维度：
1. data_fidelity：数据保真，有无编造数据
2. insight_value：行业洞察业务价值
3. strategy_feasibility：增长策略可落地性
4. logic_consistency：全文逻辑一致性

输出示例：
{{"data_fidelity":4,"insight_value":4,"strategy_feasibility":3,"logic_consistency":4,"comment":"一句话简短评价"}}

【样本统计摘要】
{stats_summary}
【行业洞察】
{insight}
【增长策略】
{strategy}
""".format(
        stats_summary=state.stats_summary,
        insight=state.industry_insight,
        strategy=state.growth_suggestion
    )
    token_cb = TokenCountCallback(node_name="evaluation_node")
    res = llm.invoke(prompt, config={"callbacks": [token_cb]})
    resp = res.content

    new_detail = state.token_detail.copy()
    new_detail.append({
        "node": "evaluation_node",
        "prompt": token_cb.prompt_tokens,
        "completion": token_cb.completion_tokens
    })
    try:
        score = json.loads(resp)
    except Exception:
        score = {
            "data_fidelity":0,
            "insight_value":0,
            "strategy_feasibility":0,
            "logic_consistency":0,
            "comment":"LLM返回JSON解析失败"
        }
    return state.model_copy(update={
        "prompt_tokens_total": state.prompt_tokens_total + token_cb.prompt_tokens,
        "completion_tokens_total": state.completion_tokens_total + token_cb.completion_tokens,
        "token_detail": new_detail,
        "eval_score": score})




def route_after_sample_check(state: TikTokState):
    """样本校验后的分支路由"""
    if state.sample_pass:
        return "industry_analyze"
    else:
        # 样本不足，判断是否还能重试
        if state.retry_count < state.max_retry:
            print(f"⚠️ 样本不足，准备重试。当前重试次数：{state.retry_count}/{state.max_retry}")
            # 重试：回到 parse_data 节点，重新采样
            return "fetch_sample"
        else:
            print(f"❌ 已达到最大重试{state.max_retry}次，仍然无法拿到足够有效样本，终止流程")
            return END



def reflect_node(state: TikTokState) -> TikTokState:
    prompt = f"""
你是数据评审专家。严格对照【真实样本统计摘要】复核行业分析报告。
【真实样本统计摘要】
{state.stats_summary}

【行业分析报告原文】
{state.industry_insight}

评审硬性规则：
1. 报告里所有量化数字（播放、点赞率、时长）必须和上面统计摘要保持一致，不允许编造不在摘要里的数据。
2. 区分“样本内结论”和“全网推断”：如果样本有限，必须明确标注结论仅适用于当前样本，不要泛化到全TikTok。
3. 找出逻辑矛盾、无数据支撑的猜想，直接修正。
输出修正后的完整Markdown报告，不要额外解释。
"""
    token_cb = TokenCountCallback(node_name="reflect_node")
    res = llm.invoke(prompt, config={"callbacks": [token_cb]})
    revised_insight = res.content

    new_detail = state.token_detail.copy()
    new_detail.append({
        "node": "reflect_node",
        "prompt": token_cb.prompt_tokens,
        "completion": token_cb.completion_tokens
    })
    return state.model_copy(update={
        "prompt_tokens_total": state.prompt_tokens_total + token_cb.prompt_tokens,
        "completion_tokens_total": state.completion_tokens_total + token_cb.completion_tokens,
        "token_detail": new_detail,
        "industry_insight": revised_insight})

def fetch_sample_node(state: TikTokState) -> TikTokState:
    """从DuckDB拉取一批全新样本，排除历史已经采集过的video_id"""
    import duckdb
    conn = duckdb.connect("./tiktok.duckdb")
    # 取出所有历史采集的vid
    seen_video_ids = {row["video_id"] for row in state.raw_samples}
    vid_tuple = tuple(seen_video_ids)

    sql = """
    SELECT 
        claim_status,
        video_id,
        video_duration_sec,
        video_transcription_text,
        verified_status,
        author_ban_status,
        video_view_count,
        video_like_count,
        video_share_count,
        video_download_count,
        video_comment_count
    FROM ods_tiktok
    """
    # 只有存在历史vid，才添加NOT IN条件，避免第一轮空元组报错
    if seen_video_ids:
        sql += f" WHERE video_id NOT IN {vid_tuple}"
    sql += " ORDER BY random() LIMIT 50;"

    res = conn.execute(sql)
    col_names = [col_info[0] for col_info in res.description]
    rows = res.fetchall()
    # 将查询返回tuple列表转为字典列表，适配Pydantic List[Dict]
    new_raw_batch = [dict(zip(col_names, one_row)) for one_row in rows]
    conn.close()

    print(f"\n📥 fetch_sample：本轮获取新样本 {len(new_raw_batch)} 条，已采集历史样本 {len(seen_video_ids)} 条")
    return state.model_copy(update={
        "new_raw_batch": new_raw_batch
    })




def parse_data_node(state: TikTokState) -> TikTokState:
    """
    解析节点：结构化字段直接本地提取，仅字幕文本交给LLM打标签
    增加异常捕获、批次截断、JSON解析容错
    【适配新架构：消费 state.new_raw_batch，不再读取 state.raw_samples】
    """
    # 本轮新一批原始样本，来自fetch_sample_node
    raw_samples: List[Dict[str, Any]] = state.new_raw_batch
    parsed_contents = []

    # 最多送50条字幕给LLM，防止上下文超限
    MAX_LLM_BATCH = 50
    batch_samples = raw_samples[:MAX_LLM_BATCH]
    # 预留超出部分样本：只保留结构化字段，不跑LLM分类
    rest_samples = raw_samples[MAX_LLM_BATCH:]

    # 1. 准备结构化基础数据 + 提取字幕文本
    structured_batch = []
    text_list = []
    for row in batch_samples:
        structured_batch.append({
            "video_id": row["video_id"],
            "views": row["video_view_count"],
            "likes": row["video_like_count"],
            "shares": row["video_share_count"],
            "comments": row["video_comment_count"],
            "duration_sec": row["video_duration_sec"],
            "verified_status": row.get("verified_status"),
            "author_ban_status": row.get("author_ban_status")
        })
        text = row.get("video_transcription_text", "")
        text_list.append(text)

    llm_json_result = []
    try:
        prompt = f"""
你是TikTok内容分类专家。
输入是多条视频字幕，按顺序处理，输出严格JSON数组，不要多余文字、markdown、注释。
每条输出字段：
- category: 内容赛道分类，简短（例如：冷知识 / 剧情 / 好物测评 / 情感）
- summary: 字幕内容一句话摘要，控制在30字以内
输入字幕列表：
{json.dumps(text_list, ensure_ascii=False)}
输出仅返回JSON数组，顺序必须和输入一一对应。
"""
        token_cb = TokenCountCallback(node_name="parse_data_node")
        resp = llm.invoke(prompt, config={"callbacks": [token_cb]})

        new_detail = state.token_detail.copy()
        new_detail.append({
            "node": "parse_data_node",
            "prompt": token_cb.prompt_tokens,
            "completion": token_cb.completion_tokens
        })
        raw_text = resp.content.strip()
        # 清理LLM可能输出的```json标记
        if raw_text.startswith("```json"):
            raw_text = raw_text.removeprefix("```json").removesuffix("```").strip()
        llm_json_result = json.loads(raw_text)
        if not isinstance(llm_json_result, list) or len(llm_json_result) != len(text_list):
            raise ValueError("LLM返回json数组长度和输入字幕不一致")
    except Exception as e:
        print(f"⚠️ LLM字幕分类调用失败: {str(e)}")
        # 失败兜底：全部标记为 unknown，不中断流程
        llm_json_result = [{"category": "unknown", "summary": "解析失败"} for _ in text_list]

    # 合并结构化数据 + LLM分类结果
    for idx, struct_data in enumerate(structured_batch):
        struct_data["category"] = llm_json_result[idx]["category"]
        struct_data["summary"] = llm_json_result[idx]["summary"]
        parsed_contents.append(struct_data)

    # 处理剩下的样本：只保留结构化字段，category标记为skip，不调用LLM
    for row in rest_samples:
        parsed_contents.append({
            "video_id": row["video_id"],
            "views": row["video_view_count"],
            "likes": row["video_like_count"],
            "shares": row["video_share_count"],
            "comments": row["video_comment_count"],
            "duration_sec": row["video_duration_sec"],
            "verified_status": row.get("verified_status"),
            "author_ban_status": row.get("author_ban_status"),
            "category": "skip",
            "summary": "超出LLM批次上限，未做文本解析"
        })

    # 【关键】把本轮新样本追加到全局历史
    combined_raw = state.raw_samples + raw_samples
    combined_parsed = state.parsed_contents + parsed_contents

    print(f"✅ parse_data完成：本轮新采集 {len(raw_samples)} 条，LLM解析 {len(batch_samples)}条，直接跳过文本解析 {len(rest_samples)} 条 ")

    # 更新state，清空临时new_raw_batch；不改动retry_count！
    return state.model_copy(update={
        "raw_samples": combined_raw,
        "parsed_contents": combined_parsed,
        "new_raw_batch": [],
        "prompt_tokens_total": state.prompt_tokens_total + token_cb.prompt_tokens,
        "completion_tokens_total": state.completion_tokens_total + token_cb.completion_tokens,
        "token_detail": new_detail,
        "sample_pass": False
    })



def industry_analyze_node(state: TikTokState) -> Dict:
    """节点2：行业分析Agent
    基于清洗样本统计 + LLM解析后的分类摘要做行业分析
    """
    prompt = f"""
基于下面TikTok样本的统计指标、视频结构化数据、内容分类摘要，输出简短markdown行业分析，包含3部分：
1.赛道整体特征
2.高频内容模板
3.用户偏好特点

【样本量化统计】
{state.stats_summary}

【解析后的样本（含赛道分类、字幕摘要）】
{state.parsed_contents}

【原始有效视频样本】
{state.clean_samples}
"""
    token_cb = TokenCountCallback(node_name="industry_analyze_node")
    res = llm.invoke(prompt, config={"callbacks": [token_cb]})

    new_detail = state.token_detail.copy()
    new_detail.append({
        "node": "industry_analyze_node",
        "prompt": token_cb.prompt_tokens,
        "completion": token_cb.completion_tokens
    })
    return {"industry_insight": res.content,
        "prompt_tokens_total": state.prompt_tokens_total + token_cb.prompt_tokens,
        "completion_tokens_total": state.completion_tokens_total + token_cb.completion_tokens,
        "token_detail": new_detail
            }



def growth_strategy_node(state: TikTokState) -> Dict:
    """节点3：增长策略Agent，复用用户增长经验"""
    prompt = """
你是资深用户增长专家。基于下面行业分析，输出TikTok可落地增长推广建议，markdown格式：
1）内容选题与钩子策略
2）发布节奏建议
3）达人collabs合作思路
4）转化路径建议
5）风险与避坑点

行业分析结论：
{insight}
""".format(insight=state.industry_insight)
    token_cb = TokenCountCallback(node_name="growth_strategy_node")
    res = llm.invoke(prompt, config={"callbacks": [token_cb]})

    new_detail = state.token_detail.copy()
    new_detail.append({
        "node": "growth_strategy_node",
        "prompt": token_cb.prompt_tokens,
        "completion": token_cb.completion_tokens
    })
    return {
        "prompt_tokens_total": state.prompt_tokens_total + token_cb.prompt_tokens,
        "completion_tokens_total": state.completion_tokens_total + token_cb.completion_tokens,
        "token_detail": new_detail,
            "growth_suggestion": res.content}

# ===================== 4.组装LangGraph工作流 =====================
builder = StateGraph(TikTokState)
builder.add_node("fetch_sample", fetch_sample_node)
builder.add_node("parse_data", parse_data_node)
builder.add_node("sample_check", sample_check_node) #新增
builder.add_node("industry_analyze", industry_analyze_node)
builder.add_node("reflect", reflect_node) #新增
builder.add_node("growth_strategy", growth_strategy_node)
builder.add_node("evaluation", evaluation_node)  # 新增评估节点
builder.add_node("save_run_log", save_run_log_node)
builder.add_node("save_report", save_report_node)

builder.add_edge(START, "fetch_sample")
builder.add_edge("fetch_sample", "parse_data")
builder.add_edge("parse_data", "sample_check")
builder.add_conditional_edges(
    "sample_check",
    route_after_sample_check,
    {
        "industry_analyze": "industry_analyze",
         "fetch_sample": "fetch_sample",  # 新增：跳回采样节点重试
        END: END
    }
)

builder.add_edge("industry_analyze", "reflect")
builder.add_edge("reflect", "growth_strategy")
builder.add_edge("growth_strategy", "evaluation")
builder.add_edge("evaluation", "save_run_log")
builder.add_edge("save_run_log", "save_report")
builder.add_edge("save_report", END)

graph = builder.compile()

if __name__ == "__main__":
    # 1.初始化数据库，加载csv进入ods层
    csv_file = "./data/input/tiktok_dataset.csv"
    init_duckdb(csv_file)
    init_agent_tables()
    etl_ods_to_dwd()

    # 2.从数据库读取10条样本作为输入
    #sample_list = load_sample_from_duckdb(sample_cnt=50)
    init_state = TikTokState(raw_samples=[])

    # 3.执行完整agent流水线
    result = graph.invoke(init_state)

    # 4.输出结果到控制台，同时写入output报告
    print("\n===== 行业分析报告 =====")
    print(result["industry_insight"])
    print("\n===== 增长策略建议 =====")
    print(result["growth_suggestion"])

    # 保存markdown报告
    output_md = os.path.join("./data/output/report_stage1.md")
    with open(output_md,"w",encoding="utf-8") as f:
        f.write("# TikTok行业增长分析报告(Stage1 Demo)\n\n")
        f.write("## 行业洞察\n")
        f.write(result["industry_insight"])
        f.write("\n## 增长推广策略\n")
        f.write(result["growth_suggestion"])
    print(f"\n报告已经保存至：{output_md}")
