# Pizza Hut Lucknow — Voice Ordering Agent
 
A voice AI agent for Pizza Hut Lucknow (Hazratganj) built with **Pipecat**, **LangGraph**, **Groq**, and **Sarvam AI**. Customers call in, speak naturally in Hindi or English, and can place orders, track deliveries, request refunds, or ask FAQs — all hands-free.
 
---
 
## Architecture
 
```
Caller (WebRTC)
    │
    ▼
Pipecat Pipeline
    ├── Sarvam STT  (speech → text, supports Hindi + English)
    ├── LangGraphProcessor  (routes to specialist agent)
    │       ├── Triage node     → classifies intent
    │             ├── Order Collect   → gathers order details, writes to Supabase
    │             ├── Order Status    → reads latest order from Supabase
    │             ├── Refund          → checks eligibility, confirms refund
    │             └── FAQ             → searches knowledge base
    └── Sarvam TTS  (text → speech)
```
 
State is held in-memory per call. Conversation turns are logged to **Upstash Redis**. All orders, users, and FAQs live in **Supabase (PostgreSQL)**.
 
---
 
## Requirements
 
- Python 3.11+
- A [Supabase](https://supabase.com) project
- A [Groq](https://console.groq.com) API key
- A [Sarvam AI](https://www.sarvam.ai) API key
- An [Upstash Redis](https://upstash.com) database (optional — for call logging)
---
 
## Installation
 
### 1. Clone the repo
 
```bash
git clone https://github.com/your-org/pizza-hut-agent.git
cd pizza-hut-agent
```
 
### 2. Create and activate a virtual environment
 
```bash
python3.11 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```
 
### 3. Install dependencies
 
```bash
pip install -r requirements.txt
```
 
### 4. Set up environment variables
 
Copy the example file and fill in your keys:
 
```bash
cp .env.example .env
```
 
Edit `.env`:
 
```env
SUPABASE_URL=https://zxabttgnzrgumeuogkfr.supabase.co
SUPABASE_KEY=sb_publishable_-PjCQ1t5zPSEguRg_Dmksw_VBr-a7tD
GROQ_API_KEY=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
DEEPGRAM_API_KEY=5b828a59fe001e1b47897112e9ff02df2e6c2749
ELEVENLABS_API_KEY=sk_d2b4cdbec59cf1b0a7ab358d3bec2d161e9edea876e844ce
UPSTASH_REDIS_REST_URL="https://super-kodiak-116676.upstash.io"
UPSTASH_REDIS_REST_TOKEN="gQAAAAAAAcfEAAIgcDIwMWJjMjVkNjU0YzI0OTA3YTNjY2M4MTgxZTg5MTRhNg"
CARTESIA_API_KEY=sk_car_mMZtxuEuGvzFdnnCwEsu5M
SARVAM_API_KEY=sk_4rcdf36d_ZVXbtTTmym1RWYqSB07qVz0X

```

