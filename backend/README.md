<p align="center">
  <img src="assets/logo.png" width="120" alt="FIRE Logo"/>
</p>

# 🔥 FIRE: Fact-checking with Iterative Retrieval and Verification

[FIRE](https://github.com/mbzuai-nlp/fire) is a novel agent-based framework for **fact-checking atomic claims**, designed to integrate **evidence retrieval and claim verification** in an **iterative and cost-effective manner**. Unlike traditional systems that fix the number of web queries before verifying, FIRE dynamically decides whether to stop or continue querying based on confidence.

<p align="center">
  <img src="assets/arch_fire.png" alt="FIRE vs SAFE vs FACTOOL architecture" width="100%"/>
</p>

## 🔍 Why FIRE?

Compared to previous systems like **FACTCHECKGPT**, **FACTOOL**, and **SAFE**, FIRE:

- Integrates reasoning and retrieval instead of separating them
- Dynamically controls the retrieval depth
- Reduces **LLM cost by 7.6×** and **search cost by 16.5×**
- Performs comparably or better on public datasets like FacTool-QA, FELM-WK, BingCheck

## 📌 Features

- **Iterative agent-based reasoning**
- **Unified decision function for retrieval or finalization**
- **Optimized for low-cost verification**
- **Supports both proprietary and open-source LLMs**
- **Extensive evaluations and ablations available**

## 🧠 How It Works

```
Input Claim
   │
   ▼
[FIRE Decision Module]
   ├── confident → Output Label (True / False)
   └── uncertain → Generate Search Query
                      │
                      ▼
          Web Search (via SerperAPI)
                      │
                      ▼
            Update Evidence Set
                      │
                      └── Loop until confident or max steps
```

## 📊 Performance Snapshot

### 🔍 FIRE vs. Baseline Systems

FIRE is compared against state-of-the-art frameworks including **FactCheckGPT**, **FACTOOL**, and **SAFE**.

### 🔧 Performance Across Datasets

<p align="center">
  <img src="assets/performance.png" width="100%" alt="FIRE Performance Table"/>
</p>

---

### 💰 Cost and Time Efficiency

<p align="center">
  <img src="assets/cost.png" width="400" alt="FIRE Cost Table"/>
</p>

## 🚀 Quickstart

### Installation

```bash
# Clone the repository
git clone https://github.com/thanhbinhbent/fire
cd fire/backend

# Create a virtual environment (recommended)
python -m venv .venv

# Activate virtual environment
# On Windows:
.\.venv\Scripts\Activate.ps1
# On Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### ⚡ Performance Modes

FIRE now supports **two verification modes**:

| Mode         | Latency | Accuracy | Use Case                      |
| ------------ | ------- | -------- | ----------------------------- |
| **Fast**     | 1-5s    | ~75%     | User-facing apps, basic facts |
| **Accurate** | 20-30s  | ~80%     | Research, complex claims      |

**Choose based on your priorities:**

- **Speed priority** → Use `mode="fast"` (default)
- **Accuracy priority** → Use `mode="accurate"`

### Configuration

1. **Copy the example environment file:**

```bash
cp .env.example .env
```

2. **Edit `.env` file and add your API keys:**

```bash
# OpenAI API Key - Get from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your-openai-api-key-here

# Anthropic API Key - Get from: https://console.anthropic.com/settings/keys
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Serper API Key (for web search) - Get from: https://serper.dev/api-key
SERPER_API_KEY=your-serper-api-key-here

# Local Ollama URL (for running models locally)
LOCAL_OLLAMA_URL=http://localhost:8001

# Random Seed
RANDOM_SEED=1
```

**Important:** The `.env` file is already added to `.gitignore` to prevent accidentally committing your API keys.

### 🖥️ Local LLM Setup (Optional)

Run models locally using Ollama with Docker/Podman for **free, private inference**:

```bash
# Navigate to LLM directory
cd ../llm

# Start Ollama server with Docker Compose
docker-compose up -d

# Verify server is running
curl http://localhost:8001/api/tags

# Pull a model (e.g., gemma3:4b)
docker exec -it ollama ollama pull gemma3:4b
```

**Available local models:**
- `gemma3:4b` - Fast, 4B parameters (~2.5GB)
- `llama3.2:3b` - Efficient, 3B parameters (~2GB)
- `qwen2.5:7b` - High quality, 7B parameters (~4GB)
- `phi3.5:latest` - Balanced performance (~2GB)

See [llm/README.md](../llm/README.md) for detailed setup instructions.

### Running FIRE

#### 🌐 API Mode (Recommended for Production)

Start the FastAPI server:

```bash
uvicorn api:app --reload
```

**API Endpoints:**

```bash
# Check status
curl http://localhost:8000/

# Fast mode (1-5s, default)
curl -X POST http://localhost:8000/api/check \
  -H "Content-Type: application/json" \
  -d '{"claim": "Thành phố Hồ Chí Minh là thủ đô của Việt Nam", "mode": "fast"}'

# Accurate mode (20-30s)
curl -X POST http://localhost:8000/api/check \
  -H "Content-Type: application/json" \
  -d '{"claim": "Chủ tịch quốc hội Việt Nam hiện nay là Trần Thanh Mẫn", "mode": "accurate"}'
```

**Response Format:**

```json
{
  "verdict": "Sai (Chắc chắn)",
  "explanation": "Thủ đô của Việt Nam là Hà Nội, không phải Thành phố Hồ Chí Minh.",
  "confidence": 0.95,
  "sources": [...],
  "metadata": {
    "mode": "fast",
    "latency": "2.58s",
    "searches": 0,
    "fast_path": true
  }
}
```

#### 🔬 Command-Line Mode (For Research/Testing)

Basic usage:

```bash
python run_fire.py --model <model-name> --dataset <dataset-name>
```

**Command Arguments:**

- `--model`: Model to use (required)
  - OpenAI: `gpt-4o-mini`, `gpt-4o`, `o1-preview`, `o1-mini`
  - Anthropic: `claude-3-5-sonnet-20240620`, `claude-3-opus`, `claude-3-haiku`
  - Groq: `groq/llama-3.1-70b-versatile` (ultra-fast)
  - Google: `gemini/gemini-1.5-flash` (fast & free)
  - Local: `ollama/gemma3:4b`, `ollama/llama3.2:3b`, `ollama/qwen2.5:7b` (free, private)
- `--provider`: Provider to use (optional, default: auto-detect)
  - Options: `openai`, `anthropic`, `groq`, `gemini`, `local`
  - Example: `--provider local --model gemma3:4b`
- `--dataset`: Dataset to evaluate (required)
  - `factcheck_bench`: Factcheck-Bench dataset (631 claims)
  - `bingcheck`: BingCheck dataset (142 claims)
  - `factool_qa`: FacTool-QA dataset (233 claims)
  - `felm_wk`: FELM-WK dataset (184 claims)
  - `vifactcheck`: Vietnamese ViFactCheck dataset (20 claims)
- `--framework`: Framework to use (default: `fire`)
  - Options: `fire`, `safe`
- `--limit`: Limit number of claims to process (optional)
  - Example: `--limit 10` processes only first 10 claims
- `--output-dir`: Output directory for results (default: `results`)

**Examples:**

```bash
# Run FIRE with GPT-4o-mini on Factcheck-Bench (all claims)
python run_fire.py --model gpt-4o-mini --dataset factcheck_bench

# Run FIRE with GPT-4o-mini, process only 1 claim for testing
python run_fire.py --model gpt-4o-mini --dataset factcheck_bench --limit 1

# Run FIRE with Claude-3.5 Sonnet on BingCheck
python run_fire.py --model claude-3-5-sonnet-20240620 --dataset bingcheck

# Run with Groq (ultra-fast)
python run_fire.py --model groq/llama-3.1-70b-versatile --dataset factcheck_bench --limit 5

# Run with local Ollama model (free, private)
python run_fire.py --provider local --model gemma3:4b --dataset vifactcheck
python run_fire.py --provider local --model llama3.2:3b --dataset factcheck_bench --limit 5

# Run SAFE framework instead of FIRE
python run_fire.py --model gpt-4o-mini --dataset factcheck_bench --framework safe

# Specify custom output directory
python run_fire.py --model gpt-4o-mini --dataset factcheck_bench --output-dir my_results
```

#### 📦 Download ViFactCheck Dataset

Download and prepare the Vietnamese fact-checking dataset:

```bash
# Download test split
python common/vifactcheck_loader.py --split test

# Download all splits with statistics
python common/vifactcheck_loader.py --split all --stats --samples 5

# Include NEI (Not Enough Info) samples
python common/vifactcheck_loader.py --split test --include-nei
```

#### ⚡ Performance Testing

Test fast verifier performance:

```bash
# Test fast mode only
python test_fast_verifier.py --fast-only

# Compare original vs fast
python test_fast_verifier.py --compare
```

**Output:**

Results are saved as JSONL files in the output directory:

```
results/fire_factcheckbench_gpt-4o-mini.jsonl
```

Each line contains:

- `claim`: The original claim text
- `label`: Ground truth label (True/False)
- `result`: Verification result with confidence and reasoning
- `searches`: List of search queries performed

## 📄 Citation

```bibtex
@inproceedings{xie-etal-2025-fire,
 address = {Albuquerque, New Mexico},
 author = {Xie, Zhuohan  and
Xing, Rui  and
Wang, Yuxia  and
Geng, Jiahui  and
Iqbal, Hasan  and
Sahnan, Dhruv  and
Gurevych, Iryna  and
Nakov, Preslav},
 booktitle = {Findings of the Association for Computational Linguistics: NAACL 2025},
 isbn = {979-8-89176-195-7},
 pages = {2901--2914},
 publisher = {Association for Computational Linguistics},
 title = {{FIRE}: Fact-checking with Iterative Retrieval and Verification},
 url = {https://aclanthology.org/2025.findings-naacl.158/},
 year = {2025}
}
```

## 👥 Authors

Developed by **Zhuohan Xie**, Rui Xing, Yuxia Wang, Jiahui Geng, Hasan Iqbal, Dhruv Sahnan, Iryna Gurevych, and Preslav Nakov  
**Affiliations**: MBZUAI, The University of Melbourne

For questions or collaborations, contact:  
📬 **zhuohan.xie@mbzuai.ac.ae**

---

_“Fact-checking, now with FIREpower.”_
