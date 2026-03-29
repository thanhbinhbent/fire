# FIRE - Vietnamese Fact-Checking System

AI-powered fact-checking for Vietnamese claims using LLM and web search.

## 🚀 Quick Start

### Step 1: Get API Keys

#### **OpenAI Key** (Required)

1. Visit https://platform.openai.com/api-keys
2. Sign up / Log in
3. Click "Create new secret key" → Copy the key
4. Add credits to your account (minimum $5)

#### **Serper Key** (Required)

1. Visit https://serper.dev/
2. Sign up with Google
3. Copy API key from dashboard
4. **Free tier**: 2,500 searches/month

### Step 2: Run Backend

### Step 2: Run Backend

```bash
# 1. Navigate to backend folder
cd backend

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# 4. Install packages
pip install -r requirements.txt

# 5. Create .env file and add API keys
# Windows:
copy .env.example .env
# Mac/Linux:
cp .env.example .env
```

**Edit `.env` file:**

```env
OPENAI_API_KEY=sk-proj-xxxxx  # Your OpenAI key
SERPER_API_KEY=xxxxx           # Your Serper key
DEFAULT_MODEL_NAME=openai/gpt-4o-mini
```

**Run server:**

```bash
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

✅ Backend running at: http://localhost:8000

### Step 3: Run Frontend

**Open new terminal:**

```bash
# 1. Navigate to frontend folder
cd frontend

# 2. Install packages
npm install

# 3. Run frontend
npm run dev
```

✅ Frontend running at: http://localhost:5173

### Step 4: Use the System

1. Open browser: http://localhost:5173
2. Enter a claim to verify, e.g., "The current General Secretary of Vietnam is Tô Lâm"
3. Wait for results (10-30 seconds)

## Experiments and Re-running

All experiments assume the backend environment from Quick Start is active and API keys are set.

If you want to evaluate ViFactCheck, generate the FIRE-format dataset first:

```bash
cd backend
python datasets/vifactcheck/vifactcheck_loader.py --split test
```

### 1) Baseline CLI evaluation (FIRE/SAFE)

```bash
cd backend
python run_fire.py --model gpt-4o-mini --dataset factcheck_bench --framework fire --output-dir results
# SAFE baseline
python run_fire.py --model gpt-4o-mini --dataset factcheck_bench --framework safe --output-dir results
```

Output: `backend/results/<framework>_<dataset>_<model>.jsonl`

### 2) Decontextualized ViFactCheck (dataset + CLI eval)

```bash
cd backend
python datasets/vifactcheck/decontextualize.py --split test --output datasets/vifactcheck/vifactcheck_decontextualized.jsonl
python run_fire.py --model gpt-4o-mini --dataset vifactcheck_decontextualized --framework fire --output-dir results
```

### 3) API-mode evaluation (fast/accurate)

Start the API server:

```bash
cd backend
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Then evaluate:

```bash
cd backend
python eval_api.py --api-url http://localhost:8000/api/check \
  --dataset datasets/vifactcheck/vifactcheck_decontextualized.jsonl \
  --mode accurate \
  --results-jsonl results/eval_api_accurate_mode.jsonl \
  --output results/eval_api_accurate_mode_summary.json
```

### 4) Translation variant (VI -> EN before verification)

```bash
cd backend
python run_fire_with_translation.py --model gpt-4o-mini --dataset vifactcheck --use-translate --output-dir results
```

### 5) Reasoning vs no reasoning (subproject)

Note: copy `backend/.env` to `reasoning_vs_no_reasoning/backend/.env` (or export the same env vars) before running.

```bash
cd reasoning_vs_no_reasoning/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Benchmark thinking vs no-thinking
python benchmark.py --model ollama/qwen3:1.7b --dataset vifactcheck --limit 100 --output-dir results

# Batch or continuous processing
python run_fire_batch.py --model ollama/qwen3:1.7b --dataset vifactcheck --batch-size 50 --output-dir results
python run_fire_continuous.py --model ollama/qwen3:1.7b --dataset vifactcheck --batch-size 10 --output-dir results
```
