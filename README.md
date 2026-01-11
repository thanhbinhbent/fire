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
