# Sentinella: Enterprise AI Gateway & Evaluation Platform

> **A centralized AI Gateway that provides unified access to multiple LLM providers with intelligent routing, automatic fallback, response caching, multi-tenancy, streaming, and comprehensive observability.**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<div align="center">
  <img src="architecture-diagram.png" alt="Sentinella Architecture"/>
  <p><em>System Architecture Overview</em></p>
  
  <img src="system-overview.png" alt="Sentinella System Overview"/>
  <p><em>Complete System Components</em></p>
</div>

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Core Features](#core-features)
- [Request Flow](#request-flow)
- [Supported Models](#supported-models)
- [Installation](#installation)
- [Usage](#usage)
- [SDKs](#sdks)
- [Evaluation Framework](#evaluation-framework)
- [Infrastructure](#infrastructure)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [API Reference](#api-reference)
- [License](#license)

## 🎯 Overview

Sentinella is an enterprise-grade AI Gateway that acts as a unified interface between applications and multiple LLM providers (OpenAI, Anthropic, Google Gemini, Grok, DeepSeek, and more). Instead of integrating directly with each provider, applications communicate through Sentinella, which handles intelligent routing, caching, fallback strategies, multi-tenancy, streaming, and observability.

### What Sentinella Does

- **Unified API**: Single endpoint for all LLM providers (25+ models across 8 providers)
- **Intelligent Routing**: ML-based model selection considering complexity, cost, and latency
- **Smart Fallback**: Automatic retry and fallback to alternative models on failure
- **Response Caching**: Redis-based caching with semantic similarity search
- **Multi-Tenancy**: Complete tenant isolation with per-tenant rate limits and cost controls
- **Streaming Support**: Real-time streaming responses via Server-Sent Events
- **Conversational Memory**: Session-based memory management for context-aware conversations
- **Prompt Templates**: Versioned prompt templates with variable substitution
- **Function Calling**: Native support for tool/function calling
- **Fine-Tuning**: Integration with fine-tuned models
- **Webhooks**: Event-driven webhooks for completions, errors, and cost thresholds
- **Full Observability**: LangFuse integration for request tracing and metrics
- **Evaluation Pipeline**: Automated quality assessment and model comparison

## 🏗️ Architecture

### System Architecture

```mermaid
graph TB
    subgraph "Client Applications"
        A1[Application 1]
        A2[Application 2]
        A3[Application 3]
        SDK[Python/JS SDKs]
    end
    
    subgraph "Sentinella Gateway"
        G[FastAPI Gateway<br/>Port 8000]
        AUTH[Auth & Rate Limiter]
        TM[Tenant Manager]
        R[Smart Router<br/>ML-Based]
        SC[Semantic Cache]
        C[Redis Cache]
        F[Fallback Chain]
        T[LangFuse Tracer]
        MEM[Memory Manager]
        WH[Webhook Manager]
    end
    
    subgraph "LLM Providers"
        O[OpenAI<br/>GPT-4o, GPT-4, GPT-3.5]
        AN[Anthropic<br/>Claude 3.5, Opus]
        GO[Google<br/>Gemini Pro, 1.5]
        GR[Grok xAI<br/>Grok-2]
        DS[DeepSeek<br/>Chat, Coder]
        CO[Cohere<br/>Command]
        MS[Mistral<br/>Large, Medium]
        LL[Meta Llama<br/>3-70b, 3-8b]
    end
    
    subgraph "Observability"
        L[LangFuse<br/>Dashboard Port 3000]
        DB[(PostgreSQL)]
        DASH[Admin Dashboard]
    end
    
    A1 --> G
    A2 --> G
    A3 --> G
    SDK --> G
    
    G --> AUTH
    AUTH --> TM
    TM --> R
    R --> SC
    SC -->|Cache Miss| C
    C -->|Cache Miss| F
    F --> O
    F --> AN
    F --> GO
    F --> GR
    F --> DS
    F --> CO
    F --> MS
    F --> LL
    
    G --> MEM
    G --> T
    T --> L
    L --> DB
    G --> WH
    G --> DASH
    
    style G fill:#4CAF50
    style R fill:#2196F3
    style SC fill:#FF9800
    style C fill:#FF9800
    style F fill:#9C27B0
    style T fill:#00BCD4
    style MEM fill:#E91E63
    style WH fill:#9C27B0
```

### Complete Request Flow

```mermaid
sequenceDiagram
    participant Client
    participant Gateway
    participant Auth as Auth & Rate Limiter
    participant Tenant as Tenant Manager
    participant Cache as Semantic Cache
    participant Router as Smart Router
    participant Memory as Memory Manager
    participant Fallback as Fallback Chain
    participant LLM as LLM Provider
    participant Tracer as LangFuse Tracer
    participant Webhook as Webhook Manager
    
    Client->>Gateway: POST /v1/chat/completions
    Gateway->>Auth: Verify API Key & Rate Limit
    Auth->>Tenant: Get Tenant Config
    Tenant-->>Auth: Tenant Config
    
    alt Rate Limit Exceeded
        Auth-->>Client: 429 Too Many Requests
    else Cost Limit Exceeded
        Auth-->>Client: 429 Cost Limit Exceeded
    else Authenticated
        Auth-->>Gateway: Authenticated
        
        Gateway->>Memory: Get Conversation History
        Memory-->>Gateway: Messages
        
        Gateway->>Cache: Check Semantic Cache
        Cache-->>Gateway: Cache Miss
        
        Gateway->>Router: Select Model (ML-Based)
        Router->>Router: Analyze Complexity
        Router->>Router: Score Models
        Router-->>Gateway: Model Selected
        
        Gateway->>Tracer: Start Trace
        Gateway->>Fallback: Execute Request
        
        Fallback->>LLM: Call Primary Model
        LLM-->>Fallback: Success Response
        
        Fallback->>Gateway: Return Response
        Gateway->>Cache: Store in Cache
        Gateway->>Memory: Save Message
        Gateway->>Tracer: Complete Trace
        Gateway->>Webhook: Send Completion Event
        Gateway->>Tenant: Track Usage
        Gateway->>Client: Return Response
    end
```

## 🔧 Core Features

### 1. Intelligent ML-Based Routing

The Smart Router uses machine learning algorithms to analyze prompts and select optimal models:

```mermaid
flowchart TD
    START[Incoming Request] --> ANALYZE[Analyze Prompt<br/>Complexity, Length, Keywords]
    ANALYZE --> ML_SCORE[ML Scoring Algorithm]
    ML_SCORE --> FACTORS[Consider Factors:<br/>Cost, Latency, Capability Match,<br/>Historical Performance]
    FACTORS --> FILTER[Filter by Constraints:<br/>Max Latency, Max Cost, Provider]
    FILTER --> SCORE[Score All Candidates]
    SCORE --> SELECT[Select Highest Score]
    SELECT --> TRACK[Track Selection & Performance]
    
    style ML_SCORE fill:#2196F3
    style FACTORS fill:#4CAF50
    style SELECT fill:#FF9800
```

**Routing Factors:**
- **Input Complexity**: ML-based complexity scoring (0.0-1.0)
- **Task Type Detection**: Code, reasoning, translation, general
- **Cost Efficiency**: Routes to cheaper models when quality threshold is met
- **Latency Constraints**: Respects maximum latency requirements
- **Historical Performance**: Tracks success rates and latency per model
- **Provider Preferences**: Supports tenant-specific provider preferences

### 2. Multi-Tenancy & Rate Limiting

Complete tenant isolation with configurable limits:

```mermaid
graph LR
    REQ[Request] --> AUTH[Authenticate API Key]
    AUTH --> TENANT[Get Tenant]
    TENANT --> RATE{Rate Limit<br/>Check}
    RATE -->|Per Minute| MIN[Minute Limit]
    RATE -->|Per Hour| HOUR[Hour Limit]
    RATE -->|Per Day| DAY[Day Limit]
    MIN --> COST{Cost Limit<br/>Check}
    HOUR --> COST
    DAY --> COST
    COST -->|Within Limits| ALLOW[Allow Request]
    COST -->|Exceeded| DENY[429 Error]
    
    style ALLOW fill:#4CAF50
    style DENY fill:#F44336
```

**Features:**
- Per-tenant API keys
- Configurable rate limits (per minute, hour, day)
- Daily and monthly cost limits
- Usage tracking per tenant
- Tenant-specific routing preferences

### 3. Semantic Caching

Advanced caching using embeddings for similarity-based cache hits:

```mermaid
flowchart LR
    REQ[Request] --> EMBED[Generate Embedding<br/>Sentence Transformers]
    EMBED --> FAISS[FAISS Index<br/>Similarity Search]
    FAISS --> SIMILAR{Similarity<br/>> 0.85?}
    SIMILAR -->|Yes| MATCH[Cache Hit<br/>Return Similar]
    SIMILAR -->|No| MISS[Cache Miss<br/>Call LLM]
    MISS --> STORE[Store Embedding<br/>& Response]
    STORE --> FAISS
    
    style MATCH fill:#4CAF50
    style MISS fill:#FF9800
```

**Benefits:**
- Finds similar cached responses even if prompts aren't identical
- Reduces costs for semantically similar queries
- Configurable similarity threshold (default: 0.85)
- Uses FAISS for fast similarity search

### 4. Streaming Responses

Real-time streaming via Server-Sent Events:

```mermaid
sequenceDiagram
    participant Client
    participant Gateway
    participant LLM as LLM Provider
    
    Client->>Gateway: POST /v1/chat/completions<br/>stream: true
    Gateway->>LLM: Start Stream
    LLM-->>Gateway: Chunk 1
    Gateway-->>Client: data: {"content": "Hello"}
    LLM-->>Gateway: Chunk 2
    Gateway-->>Client: data: {"content": " world"}
    LLM-->>Gateway: Chunk 3
    Gateway-->>Client: data: {"content": "!"}
    LLM-->>Gateway: Done
    Gateway-->>Client: data: [DONE]
```

### 5. Conversational Memory

Session-based memory management for context-aware conversations:

```mermaid
flowchart TD
    REQ[Request with session_id] --> CHECK{Session<br/>Exists?}
    CHECK -->|No| CREATE[Create Session]
    CHECK -->|Yes| LOAD[Load Messages]
    CREATE --> LOAD
    LOAD --> COUNT{Messages<br/>> 50?}
    COUNT -->|Yes| SUMMARIZE[Summarize Old<br/>Keep Last 10]
    COUNT -->|No| APPEND[Append New Message]
    SUMMARIZE --> APPEND
    APPEND --> LLM[Send to LLM<br/>with Context]
    LLM --> SAVE[Save Conversation]
    
    style SUMMARIZE fill:#FF9800
    style APPEND fill:#4CAF50
```

### 6. Smart Fallback Chain

Automatic retry and fallback with provider diversity:

```mermaid
sequenceDiagram
    participant App as Application
    participant GW as Gateway
    participant FB as Fallback Chain
    participant M1 as Primary Model<br/>OpenAI GPT-4
    participant M2 as Fallback 1<br/>Claude 3.5
    participant M3 as Fallback 2<br/>GPT-3.5
    
    App->>GW: Request
    GW->>FB: Execute Request
    FB->>M1: Try Primary Model
    M1-->>FB: Timeout Error
    FB->>M2: Fallback (Different Provider)
    M2-->>FB: Success
    FB->>GW: Response
    GW->>App: Return Response
    
    Note over FB: Provider diversity ensures<br/>resilience to provider outages
```

### 7. Prompt Templates & Versioning

Versioned prompt templates with variable substitution:

```mermaid
flowchart LR
    CREATE["Create Template<br/>Hello name!"] --> VERSION["Version 1"]
    VERSION --> UPDATE["Update Template"]
    UPDATE --> VERSION2["Version 2<br/>Hello name, welcome!"]
    VERSION2 --> RENDER["Render with Variables"]
    RENDER --> RESULT["Hello Alice, welcome!"]
    
    VERSION --> HISTORY["Version History<br/>Last 100 versions"]
    
    style VERSION fill:#4CAF50
    style VERSION2 fill:#2196F3
```

### 8. Function Calling / Tool Use

Native support for LLM function calling:

```mermaid
graph LR
    REQ[Request with Tools] --> LLM[LLM Selects Tool]
    LLM --> CALL[Function Call Request]
    CALL --> EXEC[Execute Tool]
    EXEC --> RESULT[Tool Result]
    RESULT --> LLM2[LLM Processes Result]
    LLM2 --> RESPONSE[Final Response]
    
    style CALL fill:#9C27B0
    style EXEC fill:#4CAF50
```

## 📊 Supported Models

Sentinella supports **25+ models** across **8 providers**:

### OpenAI (7 models)
- `gpt-4o` - Latest GPT-4 optimized model
- `gpt-4o-mini` - Cost-effective GPT-4 variant
- `gpt-4-turbo` - High-performance GPT-4
- `gpt-4` - Standard GPT-4
- `gpt-3.5-turbo` - Fast and cost-effective
- `gpt-4-1106-preview` - Preview model
- `gpt-4-0125-preview` - Preview model

### Anthropic Claude (4 models)
- `claude-3-5-sonnet-20241022` - Latest Claude 3.5
- `claude-3-opus-20240229` - Most capable Claude
- `claude-3-sonnet-20240229` - Balanced Claude
- `claude-3-haiku-20240307` - Fastest Claude

### Google Gemini (5 models)
- `gemini-pro` - Standard Gemini
- `gemini-pro-vision` - Multimodal Gemini
- `gemini-1.5-pro` - Advanced with 2M context window
- `gemini-1.5-flash` - Fast with 1M context window
- `gemini-ultra` - Most capable Gemini

### Grok / xAI (2 models)
- `grok-beta` - Beta Grok model
- `grok-2` - Latest Grok with 131K context

### DeepSeek (2 models)
- `deepseek-chat` - General purpose
- `deepseek-coder` - Code-optimized

### Cohere (2 models)
- `command` - Standard Cohere model
- `command-light` - Lightweight variant

### Mistral (3 models)
- `mistral-large` - Most capable Mistral
- `mistral-medium` - Balanced Mistral
- `mistral-small` - Cost-effective Mistral

### Meta Llama (2 models)
- `llama-3-70b` - Large Llama model
- `llama-3-8b` - Efficient Llama model

## 🚀 Installation

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- API keys for LLM providers

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/sentinella.git
cd sentinella
```

2. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys:
# - OPENAI_API_KEY
# - ANTHROPIC_API_KEY
# - GOOGLE_API_KEY
# - XAI_API_KEY (for Grok)
# - SENTINELLA_API_KEY
# - LANGFUSE_SECRET_KEY
# - LANGFUSE_PUBLIC_KEY
```

3. **Start services with Docker Compose**
```bash
docker-compose up -d
```

This starts:
- **Sentinella Gateway** on port 8000
- **Redis** on port 6379
- **LangFuse Dashboard** on port 3000
- **PostgreSQL** on port 5432 (for LangFuse)

4. **Verify installation**
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "cache": "connected",
  "version": "0.1.0"
}
```

## 💻 Usage

### Basic Request

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-sentinella-api-key" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "Hello, world!"}
    ],
    "temperature": 0.7
  }'
```

### Automatic Model Selection

If you don't specify a model, Sentinella will automatically select the best model:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-sentinella-api-key" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is 2+2?"}
    ]
  }'
```

### Streaming Responses

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-sentinella-api-key" \
  -d '{
    "messages": [
      {"role": "user", "content": "Tell me a story"}
    ],
    "stream": true
  }'
```

### Conversational Memory

```bash
# First message
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-sentinella-api-key" \
  -d '{
    "messages": [
      {"role": "user", "content": "My name is Alice"}
    ],
    "session_id": "session-123"
  }'

# Follow-up (remembers context)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-sentinella-api-key" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is my name?"}
    ],
    "session_id": "session-123"
  }'
```

### Using Prompt Templates

```bash
# Create template
curl -X POST http://localhost:8000/v1/templates \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-sentinella-api-key" \
  -d '{
    "name": "greeting",
    "template": "Hello {name}, welcome to {company}!",
    "variables": ["name", "company"]
  }'

# Render template
curl -X POST http://localhost:8000/v1/templates/greeting/render \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-sentinella-api-key" \
  -d '{
    "variables": {
      "name": "Alice",
      "company": "Acme Corp"
    }
  }'
```

### Function Calling

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-sentinella-api-key" \
  -d '{
    "model": "gpt-4",
    "messages": [
      {"role": "user", "content": "What is the weather in San Francisco?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get weather for a location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string"}
            }
          }
        }
      }
    ]
  }'
```

### List Available Models

```bash
curl http://localhost:8000/v1/models \
  -H "X-API-Key: your-sentinella-api-key"
```

### Get Metrics

```bash
curl http://localhost:8000/metrics \
  -H "X-API-Key: your-sentinella-api-key"
```

## 📦 SDKs

### Python SDK

```python
import asyncio
from sentinella import SentinellaClient

async def main():
    async with SentinellaClient(
        api_key="your-api-key",
        base_url="http://localhost:8000"
    ) as client:
        # Simple chat
        response = await client.chat(
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print(response["choices"][0]["message"]["content"])
        
        # Streaming
        async for chunk in client.chat_stream(
            messages=[{"role": "user", "content": "Tell me a story"}]
        ):
            print(chunk, end="", flush=True)
        
        # With conversational memory
        await client.chat(
            messages=[{"role": "user", "content": "My name is Alice"}],
            session_id="session-123"
        )
        
        response = await client.chat(
            messages=[{"role": "user", "content": "What's my name?"}],
            session_id="session-123"
        )
        print(response["choices"][0]["message"]["content"])  # "Alice"

asyncio.run(main())
```

**Installation:**
```bash
cd sdks/python
pip install -e .
```

### JavaScript/TypeScript SDK

```typescript
import SentinellaClient from '@sentinella/sdk';

const client = new SentinellaClient(
  'your-api-key',
  'http://localhost:8000'
);

// Simple chat
const response = await client.chat([
  { role: 'user', content: 'Hello!' }
]);
console.log(response.choices[0].message.content);

// Streaming
for await (const chunk of client.chatStream([
  { role: 'user', content: 'Tell me a story' }
])) {
  process.stdout.write(chunk);
}
```

**Installation:**
```bash
cd sdks/javascript
npm install
npm run build
```

## 🧪 Evaluation Framework

### Running Evaluations

Evaluate a single model against the golden dataset:

```bash
python -m src.evaluator.evaluator \
  --model gpt-3.5-turbo \
  --dataset src/evaluator/datasets/golden_dataset.json \
  --output results.json
```

Compare multiple models:

```bash
python -m src.evaluator.evaluator \
  --compare gpt-4o-mini gpt-3.5-turbo claude-3-haiku \
  --dataset src/evaluator/datasets/golden_dataset.json
```

### Evaluation Metrics

The evaluation framework tracks:

- **Quality Score**: Semantic similarity to expected answers
- **Latency**: Response time in milliseconds
- **Cost**: Estimated cost per request
- **Token Usage**: Input/output token counts
- **Success Rate**: Percentage of successful requests

### Using Jupyter Notebooks

```bash
cd notebooks/evaluation
jupyter notebook evaluation_demo.ipynb
```

## 🏗️ Infrastructure

### Local Development (Docker Compose)

```mermaid
graph TB
    subgraph "Docker Network"
        GW[Gateway:8000]
        RD[Redis:6379<br/>Multiple DBs]
        LF[LangFuse:3000]
        PG[(PostgreSQL:5432)]
    end
    
    GW --> RD
    GW --> LF
    LF --> PG
    
    style GW fill:#4CAF50
    style RD fill:#FF9800
    style LF fill:#00BCD4
```

### Production Deployment (AWS EKS)

```mermaid
graph TB
    subgraph "AWS VPC"
        subgraph "Public Subnet"
            LB[Load Balancer]
        end
        subgraph "Private Subnet"
            EKS[EKS Cluster]
            subgraph "Pods"
                GW1[Gateway Pod 1]
                GW2[Gateway Pod 2]
                GW3[Gateway Pod 3]
            end
        end
        subgraph "Data Layer"
            RD[(ElastiCache Redis)]
            PG[(RDS PostgreSQL)]
        end
    end
    
    LB --> EKS
    EKS --> GW1
    EKS --> GW2
    EKS --> GW3
    GW1 --> RD
    GW2 --> RD
    GW3 --> RD
    GW1 --> PG
    GW2 --> PG
    GW3 --> PG
    
    style LB fill:#2196F3
    style EKS fill:#4CAF50
    style RD fill:#FF9800
    style PG fill:#9C27B0
```

### Terraform Infrastructure

The project includes Terraform configurations for:

- VPC with public/private subnets
- ElastiCache Redis cluster
- RDS PostgreSQL instance
- EKS cluster (basic configuration)
- Security groups and IAM roles

### Kubernetes Deployment

Helm charts are provided for:

- Gateway deployment with auto-scaling
- Service configuration
- Horizontal Pod Autoscaler (HPA)
- ConfigMaps and Secrets management

## 📁 Project Structure

```
sentinella/
├── src/
│   ├── gateway/              # FastAPI application
│   │   ├── __init__.py
│   │   ├── main.py          # API entry point
│   │   └── models.py        # Pydantic schemas
│   ├── router/              # Intelligent routing
│   │   ├── __init__.py
│   │   ├── router.py        # ML-based model selection
│   │   └── fallback.py      # Fallback strategies
│   ├── cache/               # Redis caching
│   │   ├── __init__.py
│   │   ├── redis_client.py   # Regular cache
│   │   └── semantic_cache.py # Semantic similarity cache
│   ├── tenancy/             # Multi-tenancy
│   │   ├── __init__.py
│   │   └── tenant_manager.py
│   ├── rate_limiter/        # Rate limiting
│   │   ├── __init__.py
│   │   └── limiter.py
│   ├── streaming/           # Streaming support
│   │   ├── __init__.py
│   │   └── stream_handler.py
│   ├── memory/              # Conversational memory
│   │   ├── __init__.py
│   │   └── memory_manager.py
│   ├── prompts/            # Prompt templates
│   │   ├── __init__.py
│   │   ├── template_manager.py
│   │   └── optimizer.py
│   ├── tools/              # Function calling
│   │   ├── __init__.py
│   │   └── tool_manager.py
│   ├── webhooks/           # Webhook system
│   │   ├── __init__.py
│   │   └── webhook_manager.py
│   ├── finetuning/        # Fine-tuning support
│   │   ├── __init__.py
│   │   └── finetune_manager.py
│   ├── connection_pool/   # Connection pooling
│   │   ├── __init__.py
│   │   └── pool_manager.py
│   ├── async_processing/  # Background jobs
│   │   ├── __init__.py
│   │   └── queue_manager.py
│   ├── dashboard/          # Admin dashboard
│   │   ├── __init__.py
│   │   └── routes.py
│   ├── observability/      # LangFuse integration
│   │   ├── __init__.py
│   │   └── tracer.py
│   ├── evaluator/         # Evaluation pipeline
│   │   ├── __init__.py
│   │   ├── evaluator.py
│   │   └── datasets/
│   │       └── golden_dataset.json
│   └── models/            # Data models
│       ├── __init__.py
│       ├── tenant.py
│       ├── prompt_template.py
│       ├── conversation.py
│       └── webhook.py
├── sdks/
│   ├── python/            # Python SDK
│   │   ├── sentinella/
│   │   │   ├── __init__.py
│   │   │   └── client.py
│   │   └── setup.py
│   └── javascript/        # JavaScript/TypeScript SDK
│       ├── src/
│       │   └── index.ts
│       ├── package.json
│       └── tsconfig.json
├── infra/
│   ├── terraform/         # AWS infrastructure
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   └── kubernetes/        # K8s deployment
│       └── helm/
│           └── sentinella-chart/
├── tests/                 # Test suite
│   ├── test_router.py
│   └── test_cache.py
├── notebooks/             # Jupyter notebooks
│   └── evaluation/
│       └── evaluation_demo.ipynb
├── docker-compose.yml     # Local development
├── Dockerfile            # Container image
├── requirements.txt      # Python dependencies
├── pyproject.toml       # Project configuration
└── README.md            # This file
```

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **API Framework** | FastAPI | High-performance async API |
| **LLM Abstraction** | LiteLLM | Unified interface for LLM providers |
| **Caching** | Redis | Response caching layer |
| **Semantic Cache** | FAISS + Sentence Transformers | Similarity-based caching |
| **Observability** | LangFuse | LLM-specific tracing and metrics |
| **Database** | PostgreSQL | LangFuse data storage |
| **Rate Limiting** | Redis + SlowAPI | Distributed rate limiting |
| **Connection Pooling** | httpx | HTTP connection reuse |
| **Async Processing** | Redis Queues | Background job processing |
| **Infrastructure** | Terraform | Infrastructure as Code |
| **Orchestration** | Kubernetes/Helm | Container orchestration |
| **Cloud Provider** | AWS | EKS, ElastiCache, RDS |

## 🔐 Security

- **API Key Authentication**: All requests require valid API keys via `X-API-Key` header
- **Multi-Tenancy**: Complete tenant isolation
- **Rate Limiting**: Per-tenant rate limits and cost controls
- **Environment Variables**: Sensitive credentials stored in environment variables
- **Secrets Management**: Kubernetes secrets for production deployments
- **Network Isolation**: VPC configuration with private subnets
- **Webhook Signatures**: HMAC-SHA256 signature verification for webhooks

## 📝 API Reference

### POST /v1/chat/completions

Main endpoint for chat completions with support for streaming, memory, and function calling.

**Request:**
```json
{
  "model": "gpt-3.5-turbo",  // Optional, auto-selected if omitted
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,        // Optional
  "max_tokens": 100,         // Optional
  "stream": false,           // Optional, enable streaming
  "session_id": "session-123", // Optional, for conversational memory
  "tools": []                // Optional, for function calling
}
```

**Response:**
```json
{
  "id": "chatcmpl-123",
  "model": "gpt-3.5-turbo",
  "choices": [...],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  },
  "latency_ms": 1250.5,
  "cached": false
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "cache": "connected",
  "version": "0.1.0"
}
```

### GET /v1/models

List all available models (25+ models across 8 providers).

### GET /metrics

Get gateway metrics (cache stats, router stats, tenant usage).

### POST /v1/templates

Create a prompt template.

### POST /v1/templates/{id}/render

Render a template with variables.

### GET /api/dashboard/stats

Get overall gateway statistics (requires admin key).

## 🧪 Testing

Run tests:

```bash
# Install test dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LiteLLM](https://github.com/BerriAI/litellm) for provider abstraction
- [LangFuse](https://github.com/langfuse/langfuse) for observability
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [FAISS](https://github.com/facebookresearch/faiss) for similarity search
- [Sentence Transformers](https://www.sbert.net/) for embeddings
