# Featurize 金融报告 RAG 项目部署手册

本项目测试是在在 Featurize 中完成，具体以实际为准

流程：
```text
PDF 报告
  -> 按页提取正文和表格
  -> 正文分块、表格转 Markdown，并保留页码与内容类型
  -> BM25 检索
  -> Chroma 向量检索
  -> RRF 混合检索
  -> 大模型根据 Top K 证据回答
  -> 显示回答、页码和原文
```

项目不需要 GitHub，不需要单独部署数据库服务。Chroma、Embedding 模型、PDF 和代码都保存在 Featurize 的持久化目录 `/home/featurize/work`。

## 1. 项目目录

上传后的目录应为：

```text
/home/featurize/work/financial-rag-starter/
├── app.py
├── rag_core.py
├── requirements.txt
├── .env.example
├── eval_questions.csv
├── data/
│   └── report.pdf
├── models/
└── chroma_db/
```

其中：

- `data/report.pdf`：需要分析的 PDF；
- `models/`：Embedding 模型缓存；
- `chroma_db/`：持久化向量索引；
- `.env`：大模型和向量检索配置；
- `eval_questions.csv`：自己填写的检索评测集。

## 2. 上传代码和报告

将项目压缩包上传到 `/home/featurize/work`，在 Featurize 终端执行：

```bash
cd /home/featurize/work
unzip financial-rag-featurize-with-tables.zip -d financial-rag-starter
cd financial-rag-starter
mkdir -p data models chroma_db
```

如果压缩包解压后多了一层同名目录，使用以下命令检查：

```bash
pwd
ls -la
```

当前工作目录中必须直接看到 `app.py` 和 `requirements.txt`。

将报告上传到：

```text
/home/featurize/work/financial-rag-starter/data/report.pdf
```

文件名必须是 `report.pdf`。也可以不放入 `data/`，应用启动后通过网页上传 PDF。网页上传的文件只用于当前应用会话；固定放在 `data/report.pdf` 更适合重复调试。

确认文件存在：

```bash
ls -lh /home/featurize/work/financial-rag-starter/data/report.pdf
```

PDF 必须能够选择和复制文字。当前代码不包含 OCR，扫描版 PDF 可能无法提取正文和表格内容。

## 3. PyMuPDF 正文和表格提取结果

项目使用 PyMuPDF 打开整份 PDF，并逐页处理。它不会生成拆分后的新 PDF，而是在内存中生成两类结构化内容：

```python
ExtractedItem(
    page=12,
    content_type="text",
    text="该页表格区域之外的正文"
)

ExtractedItem(
    page=12,
    content_type="table",
    text="表格 1\n| 列1 | 列2 |\n| --- | --- |\n| ... | ... |"
)
```

提取流程为：

1. `page.find_tables()` 检测当前页的表格区域；
2. `page.get_text("blocks")` 获取带坐标的正文块；
3. 与表格区域重叠达到 50% 的正文块不再进入普通正文，减少表格内容重复；
4. `table.extract()` 提取表格行列；
5. 表格转换为 Markdown，并标记为 `content_type="table"`；
6. 正文和表格分别切分为 chunk；
7. 每个 chunk 保存 `chunk_id`、PDF 页码、内容类型和文字；
8. 所有 chunk 同时进入 BM25 和 Chroma。

处理完成后，程序拥有整份 PDF 的所有可提取正文与成功识别的表格，但页面问答只展示检索到的 Top K chunk。“索引检查”标签可以按编号查看全部 chunk；页面顶部的“表格片段”显示表格进入索引的数量。

表格在 Chroma 中保存的元数据为：

```python
{
    "page": 12,
    "chunk_id": 35,
    "content_type": "table"
}
```

大模型收到的表格证据类似：

```text
[来源2｜第12页｜table]
表格 1
| 项目 | 本期 | 上期 |
| --- | --- | --- |
| ... | ... | ... |
```

表格不是图片，也没有保留字体、颜色和单元格视觉样式。它保留的是 PyMuPDF 能识别出的行列文字结构。

## 4. 创建 Python 环境

进入项目目录并创建虚拟环境：

```bash
cd /home/featurize/work/financial-rag-starter
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

`sentence-transformers` 会安装 PyTorch 等依赖，因此安装时间可能较长。项目使用 CPU 运行，不要求 CUDA。

每次重新打开终端后执行：

```bash
cd /home/featurize/work/financial-rag-starter
source .venv/bin/activate
```

如果 `python -m venv` 不可用，可以使用 Featurize 已有 Conda：

```bash
conda create -n financial-rag python=3.11 -y
conda activate financial-rag
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

虚拟环境和 Conda 二选一，不要混用。

## 5. 配置向量检索和大模型

复制环境变量模板：

```bash
cd /home/featurize/work/financial-rag-starter
cp .env.example .env
nano .env
```

填写内容：

```dotenv
LLM_API_KEY=填写实际API密钥
LLM_BASE_URL=填写OpenAI兼容接口地址
LLM_MODEL=填写实际模型名称

EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
MODEL_CACHE_DIR=/home/featurize/work/financial-rag-starter/models
CHROMA_DIR=/home/featurize/work/financial-rag-starter/chroma_db
```

保存 `nano`：按 `Ctrl+O`，回车，再按 `Ctrl+X`。

### 不使用大模型时

保留空值：

```dotenv
LLM_API_KEY=
LLM_BASE_URL=
LLM_MODEL=
```

此时 BM25、向量检索和混合检索仍然可用，系统会直接展示 Top K 原文，不生成归纳回答。

### 使用 OpenAI 兼容大模型时

必须从所使用的模型服务商控制台获取以下三个值：

1. API Key；
2. OpenAI 兼容 Base URL；
3. 账号实际可用的模型名称。

不要凭示例猜测模型名称。以服务商当前文档和账号控制台为准。代码调用的是：

```python
client = OpenAI(api_key=api_key, base_url=base_url)
client.chat.completions.create(
    model=model,
    temperature=0,
    messages=[...],
)
```

因此服务商必须兼容 OpenAI Chat Completions 接口。如果服务商只支持其他协议，需要修改 `rag_core.py` 中的 `generate_answer()`。

### 密钥要求

- `.env` 只保存在 Featurize，不要放进代码压缩包；
- 不要在截图、终端记录或聊天中展示密钥；
- 公网演示结束后停止 Streamlit 或取消端口映射；
- 建议在模型服务商处设置余额和调用额度上限；
- 只向外部 API 发送允许外传的公开报告内容。

## 6. 向量检索的具体流程

代码使用：

- Embedding：`BAAI/bge-small-zh-v1.5`；
- 向量数据库：Chroma PersistentClient；
- 距离：Cosine；
- 运行设备：CPU；
- 持久化目录：`chroma_db/`。

第一次在页面中选择 `Vector` 或 `Hybrid` 时，程序执行：

1. 从模型源下载 Embedding 模型到 `models/`；
2. 使用模型将正文片段和 Markdown 表格片段编码为向量；
3. 将向量、原文、页码、内容类型和 `chunk_id` 写入 Chroma；
4. 使用 PDF 哈希、分块参数和模型名称生成独立集合名；
5. 后续启动时复用相同集合，不重复编码。

首次下载模型需要 Featurize 实例能够访问模型源。如果网络无法下载，可以在其他可联网环境下载完整模型目录，再上传到：

```text
/home/featurize/work/financial-rag-starter/models/
```

然后把 `.env` 中的 `EMBEDDING_MODEL` 改为实际本地模型目录，例如：

```dotenv
EMBEDDING_MODEL=/home/featurize/work/financial-rag-starter/models/bge-small-zh-v1.5
```

必须上传完整的模型目录，不能只上传单个权重文件。

### 三种检索模式

#### BM25

`jieba` 对问题和文本分词，BM25 按词项匹配评分。适合精确科目、专有名词和数字相关问题，不需要下载 Embedding 模型。

#### Vector

问题通过同一个 Embedding 模型编码，Chroma 查找余弦距离最近的片段。适合同义表达和语义相近问题。

#### Hybrid

分别从 BM25 和 Chroma 召回候选片段，再使用 Reciprocal Rank Fusion 合并：

```python
rrf_score = sum(1 / (60 + rank))
```

RRF 只依赖两边的排名，不直接混合不可比较的 BM25 分数和余弦相似度。相同 `chunk_id` 会被去重。

## 7. 启动应用

执行：

```bash
cd /home/featurize/work/financial-rag-starter
source .venv/bin/activate
streamlit run app.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.maxUploadSize 500
```

如果使用 Conda，将 `source .venv/bin/activate` 替换为：

```bash
conda activate financial-rag
```

启动终端必须保持运行。服务必须绑定 `0.0.0.0`，不能只绑定 `localhost` 或 `127.0.0.1`。

## 8. 暴露 Featurize 公网端口

打开第二个终端：

```bash
featurize port export 8501
```

命令会返回公网地址。使用命令实际返回的地址访问，不要直接访问 `0.0.0.0:8501`，也不要假定公网端口固定。

查看端口映射：

```bash
featurize port list
```

结束演示后取消映射：

```bash
featurize port unexport 8501
```

然后在运行 Streamlit 的终端按 `Ctrl+C` 停止服务。

## 9. 页面操作顺序

1. 首先选择 `BM25`；
2. 确认页面显示 PDF 页数和文本片段数；
3. 检查“表格片段”是否大于 0；
4. 打开“索引检查”，分别抽查 `text` 和 `table` 片段；
5. 输入正文问题，确认可以召回原文和页码；
6. 输入表格中的指标或科目问题，确认召回证据标记为 `table`；
7. 选择 `Vector`，等待首次下载模型和建立 Chroma 索引；
8. 使用相同问题比较向量检索结果；
9. 选择 `Hybrid`，检查合并后的 Top K 证据；
10. `.env` 已配置大模型时，确认回答中带有 `[来源N｜第N页]`；
11. 提问一个 PDF 中不存在答案的问题，确认模型明确表示无法确认；
12. 上传评测 CSV，分别记录三种模式的 `Hit@K`。
例`Hit@4`
每个问题取前4个来源chunk。
如果这4个来源中出现了expected_keyword，记为命中。
所有问题的命中比例，就是Hit@4。

第一次建立向量索引期间不要刷新页面或重复点击。等待左上角运行状态结束。

## 10. 大模型问答的具体流程

代码不会把整个 PDF 发送给大模型。每次问答流程为：

```text
问题
 -> 选择的检索模式召回 Top K
 -> 为每段证据附加来源编号和 PDF 页码
 -> 拼接系统提示词、问题和证据
 -> 调用 Chat Completions API
 -> 显示模型回答
 -> 在回答下方显示全部召回原文
```

系统提示词要求：

- 只能使用给定证据；
- 数字和关键结论必须引用来源；
- 证据不足必须拒绝推测；
- 使用中文简洁回答。

即使模型给出了页码引用，也必须人工核对回答下方的召回原文。Prompt 只能降低幻觉风险，不能保证事实正确。

API 调用失败不会破坏 Chroma 索引。临时清空 `.env` 中的大模型配置并重启应用，即可继续使用纯检索模式。

## 11. 检索评测 CSV

编辑 `eval_questions.csv`：

```csv
question,expected_keyword
填写实际问题,填写应当出现在召回原文中的关键词
填写实际问题,填写应当出现在召回原文中的关键词
```

要求：

- 第一行列名不能修改；
- 使用 UTF-8 或 UTF-8 with BOM 编码；
- 每个问题只填写一个主要关键词；
- 评测集中应同时包含正文问题和表格问题；
- 关键词用于检查 Top K 原文，不用于检查大模型回答；
- 使用同一份 CSV 比较 BM25、Vector 和 Hybrid。

页面显示的 `Hit@K` 表示：前 K 个检索片段中包含预期关键词的问题占比。

## 12. 更换 PDF、参数或表格提取代码

代码根据以下内容隔离 Chroma 集合：

- PDF 文件 SHA256；
- `chunk_size`；
- `overlap`；
- Embedding 模型名称。
- 索引结构版本。

更换 PDF、分块参数或索引结构版本时会建立新的集合，不会覆盖旧集合。本次加入表格提取后，代码使用 `v2-table` 集合，不会复用旧正文索引。长期调试后 `chroma_db/` 可能积累多个集合。

需要全部重建向量索引时，先停止 Streamlit，再执行：

```bash
cd /home/featurize/work/financial-rag-starter
mv chroma_db chroma_db_backup_$(date +%Y%m%d_%H%M%S)
mkdir chroma_db
```

这样旧索引仍有备份。确认新索引正常后，再自行删除备份目录。

如果只更换 `data/report.pdf`，刷新页面即可。页面上传的 PDF 优先级高于 `data/report.pdf`。

## 13. 常见错误处理

### 页面提示 PDF 未提取出可检索文本

PDF 可能是扫描件或字体编码异常。先确认能否在 PDF 阅读器中选择和复制文字。当前项目不包含 OCR。

### 表格片段数量为 0

PyMuPDF 没有在该 PDF 中识别到规则表格。先检查表格是不是扫描图片，或者是否存在复杂的跨页、合并单元格和无边框布局。此时正文仍可检索，但不能声称表格已经结构化提取。

### 表格行列错位

打开“索引检查”查看 `table` 片段，并与原 PDF 对照。复杂合并单元格、跨页表格、多层表头和无规则线框可能导致列错位。当前版本不自动修复，也不应让大模型在结构错乱的证据上推断数字归属。

### 同一表格同时出现在正文和表格片段

代码使用坐标重叠比例排除表格区域内的普通文本块，但 PDF 内部布局可能导致边界不完全一致。少量重复不会阻止检索；如果重复严重，需要根据该 PDF 调整 `_overlap_ratio` 的阈值或增加去重逻辑。

### Embedding 模型下载失败

先切换为 BM25，保证基础检索可用。检查 Featurize 网络，或手动上传完整模型目录并将 `EMBEDDING_MODEL` 改为本地路径。

### Chroma 初始化失败

确认目录可写：

```bash
mkdir -p /home/featurize/work/financial-rag-starter/chroma_db
touch /home/featurize/work/financial-rag-starter/chroma_db/write_test
```

如果错误发生在修改依赖或中断建库之后，先备份并重建 `chroma_db/`。

### 大模型返回 401 或 authentication error

API Key 无效、过期或没有正确读取。修改 `.env` 后必须停止并重新启动 Streamlit。

### 大模型返回 404 或 model not found

检查 Base URL 和模型名称。两者必须来自同一家服务商，并且模型必须对当前账号开放。

### 大模型返回 429

通常是余额不足、频率限制或并发限制。等待后重试并检查服务商控制台。不要在页面中连续点击问答按钮。
### 大模型返回 Using SOCKS proxy, but the 'socksio' package is not installed. Make sure to install httpx using pip install httpx[socks].

云服务器原因，python -m pip install socksio即可

### 网页无法打开

依次确认：

```bash
featurize port list
```

- Streamlit 终端仍在运行；
- 启动参数包含 `--server.address 0.0.0.0`；
- 导出的是本地端口 `8501`；
- 浏览器使用的是 `featurize port export` 返回的公网地址。

### 修改 `.env` 后没有生效

在 Streamlit 终端按 `Ctrl+C`，重新执行启动命令。应用启动时读取 `.env`，运行中修改不会自动更新所有缓存对象。

## 14. 文件持久化和关机

需要持久化的文件必须保存在：

```text
/home/featurize/work/financial-rag-starter/
```

包括：

- 项目代码；
- `data/report.pdf`；
- `.env`；
- `models/`；
- `chroma_db/`；
- 评测 CSV。

结束使用时：

```bash
featurize port unexport 8501
```

在 Streamlit 终端按 `Ctrl+C`，然后通过 Featurize 控制台关机或归还实例，避免持续计费。

## 15. 关键代码位置

- PDF 正文与表格提取：`rag_core.py` 中的 `extract_document()`；
- 表格转 Markdown：`_table_to_markdown()`；
- 正文和表格分块：`split_items()`；
- BM25：`BM25Index`；
- Chroma 建库和查询：`ChromaIndex`；
- 混合检索：`reciprocal_rank_fusion()`；
- 大模型调用：`generate_answer()`；
- 检索评测：`evaluate_retrieval()`；
- 页面流程和配置：`app.py`。

当前版本的明确边界：只处理单份文本型 PDF；可提取 PyMuPDF 能识别的规则表格，但不包含 OCR、图片图表理解、复杂合并单元格还原、跨页表格拼接、用户登录、多报告元数据筛选、Reranker 和大规模并发服务。
