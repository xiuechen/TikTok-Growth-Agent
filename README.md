# TikTok Growth Research Agent

LangGraph + DuckDB 构建的数据分析 Agent，自动采集 TikTok 样本、赛道分析、自检防幻觉、输出增长策略并自评成果。
面向作品集，重点展示 Agent 闭环、数据分层、LLM 幻觉校验，适合 AI / 量化 / 数据工程面试。

## ✨ Project Highlights

很多 LLM 数据分析 Demo 的通病：大模型编造数字、把小样本结论直接外推全网。
本项目设计**Reflect 事实校验节点**，形成「生成→复核修正→策略产出→自评打分」完整 Agent 闭环：

1. 自动从 DuckDB 采样，过滤无效样本；样本不足自动重试采集，直到满足最小样本量
2. 结构化指标（播放、点赞、时长等）本地计算，**仅字幕文本送入 LLM**，减少 Token 消耗
3. 生成赛道行业洞察 → Reflect 节点对照原始统计摘要校验，修正幻觉，强制区分【样本内结论】vs 全网推断
4. 基于校验后的洞察，输出 TikTok 可落地增长运营策略
5. 评估节点对整套结果做 4 维结构化打分，所有原始数据、报告、评分持久化存入 DuckDB 分层数仓
6. 配套查询脚本，回溯多次实验，观察样本质量如何影响 Agent 输出质量

## 🧩 Workflow

graph LR
A[fetch_sample 采样] --> B[sample_check 样本过滤校验]
B -->|样本充足| C[parse_data 解析：结构化本地处理，字幕送LLM]
B -->|样本不足，未达阈值| A
C --> D[industry_analyze 生成赛道行业洞察]
D --> E[reflect_node 事实校验，修正LLM幻觉]
E --> F[growth_strategy_node 输出增长策略]
F --> G[evaluation_node 4维度自评打分]
G --> H[save_run_log & save_report 入库ODS/DWD/DWS]

```
graph LR
A[fetch_sample 采样] --> B[sample_check 样本过滤校验]
B -->|样本充足| C[parse_data 解析：结构化本地处理，字幕送LLM]
B -->|样本不足，未达阈值| A
C --> D[industry_analyze 生成赛道行业洞察]
D --> E[reflect_node 事实校验，修正LLM幻觉]
E --> F[growth_strategy_node 输出增长策略]
F --> G[evaluation_node 4维度自评打分]
G --> H[save_run_log & save_report 入库ODS/DWD/DWS]
```


## 📦 Data Warehouse Layer (DuckDB)

- **ODS**：原始 TikTok 视频明细（原始导出数据）
- **DWD**：清洗后视频明细表 + Agent 单次运行原始日志
- **DWS**：报告汇总层，存储行业洞察、增长策略、Agent 自评分数

数据表：

1. `dwd_agent_run_log`：每一轮 Agent 运行原始日志，原始样本、解析内容
2. `dws_agent_report`：最终产出汇总，Markdown 报告、评估 JSON 分数
3. `dwd_tiktok_clean`：清洗过滤后的 TikTok 有效样本明细

## 📋 评估维度（evaluation_node）

1. `data_fidelity` 数据保真度：是否编造不存在的统计数字
2. `insight_value` 洞察价值：结论对业务的参考价值
3. `strategy_feasibility` 策略可行性：增长方案落地难易程度
4. `logic_consistency` 逻辑自洽性：全文结论无内部矛盾

> 
> 示例输出：`数据保真:5, 洞察价值:4, 策略可行性:4, 逻辑一致性:5`
> 评语：数据严格对应样本摘要，修正了认证账号矛盾；洞察与策略均标注样本边界，逻辑自洽，落地性强。

## 🛠️ Tech Stack

- Python 3.9+
- LangGraph：Agent 工作流编排，条件分支、循环重试
- DuckDB：嵌入式 OLAP 数仓，分层存储、快速 SQL 统计
- Pydantic：状态管理、类型校验
- LLM API：文本标签提取、报告生成、自检评审、结构化打分
- Pandas：样本过滤、指标统计

## 🚀 Quick Start

### 1. Install dependencies

```
pip install langgraph langchain duckdb pandas pydantic
```

### 2. Run Agent pipeline

```
python3 main.py
```

程序会自动初始化数据表，采样、清洗样本，执行完整 Agent 链路，结果写入`tiktok.duckdb`。

### 3. View history experiment results

```
python3 check_history.py
```

打印所有历史运行记录，查看历次报告的 4 项评估分数与评语。

## 📌 Project Structure

```
TikTok-Growth-Agent/
├── main.py               # 完整LangGraph定义、所有Agent节点、状态定义
├── check_history.py      # 历史实验结果查询脚本
├── tiktok.duckdb         # DuckDB数据库（运行后自动生成，gitignore忽略）
├── .gitignore
└── README.md
```

## ⚠️ Notes

1. 所有分析结论**仅代表当前采样样本，不代表全网 TikTok**，由 reflect 节点强制约束
2. 结构化字段本地计算，仅转录文本交给大模型，控制 LLM 调用成本
3. 评估分数为 Agent 自打分，用于实验对比，不是绝对业务指标

## 📈 Next Plan (Roadmap)

- LLM Token 用量统计，记录各节点输入输出 token，存入运行日志
- 低分自动重试分支：评估分数低于阈值，自动重新采样并分析
- 可视化脚本，绘制样本量与评估分数相关性
- 单元测试，对各个节点做独立测试
