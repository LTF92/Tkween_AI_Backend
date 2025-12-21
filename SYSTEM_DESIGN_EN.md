# System Design - Tkween Arabic Poetry AI Platform

---

## 1. Overview

### 1.1 System Description
An intelligent system for generating Arabic poetry using Generative AI technologies. The system enables users to create eloquent, metered, and rhymed Arabic poems based on a given title and details.

### 1.2 Objectives
- Generate metered and rhymed Arabic poems
- Support specifying the desired poetic meter
- Ensure correct Classical Arabic language
- Simple and fast user experience

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CLIENT LAYER                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────────────────┐      │
│  │                     Web Application                                │      │
│  │              (Vanilla HTML / CSS / JavaScript)                     │      │
│  │                   Hosted on Cloudflare Pages                       │      │
│  └───────────────────────────────┬───────────────────────────────────┘      │
│                                  │                                           │
└──────────────────────────────────┼───────────────────────────────────────────┘
                                   │ HTTPS
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GATEWAY LAYER                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      Cloudflare CDN                                  │    │
│  │              (SSL/TLS, DDoS Protection, Caching)                     │    │
│  └──────────────────────────────┬──────────────────────────────────────┘    │
│                                 │                                            │
│  ┌──────────────────────────────▼──────────────────────────────────────┐    │
│  │                         Nginx                                        │    │
│  │              (Reverse Proxy, Load Balancing)                         │    │
│  └──────────────────────────────┬──────────────────────────────────────┘    │
└─────────────────────────────────┼────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       APPLICATION LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      FastAPI Server                                  │    │
│  │                    (Python 3.12, Uvicorn)                           │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │                                                                      │    │
│  │  ┌──────────────────────────┐  ┌──────────────────────────┐         │    │
│  │  │      Poem Generator     │  │      Stats Handler       │         │    │
│  │  │    (POST /poems)        │  │      (GET /stats)        │         │    │
│  │  └────────────┬────────────┘  └────────────┬─────────────┘         │    │
│  │               │                            │                        │    │
│  │  ┌──────▼─────────────────▼─────────────────▼───────┐               │    │
│  │  │              CORS Middleware                      │               │    │
│  │  │         (Cross-Origin Resource Sharing)          │               │    │
│  │  └──────────────────────────────────────────────────┘               │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ API Request
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AI LAYER                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                           OpenAI API                                   │  │
│  ├───────────────────────────────────────────────────────────────────────┤  │
│  │                                                                        │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │                     ChatGPT-5.2                                  │  │  │
│  │  │  ─────────────────────────────────────────────────────────────  │  │  │
│  │  │  • Generate eloquent Arabic poems                               │  │  │
│  │  │  • Maintain meter and rhyme                                     │  │  │
│  │  │  • Temperature: 0.6 for accuracy and consistency                │  │  │
│  │  │  • Response Format: JSON                                         │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. System Components

### 3.1 Client Layer

| Component | Technology | Description |
|-----------|------------|-------------|
| Web App | Vanilla HTML/CSS/JS | Graphical user interface |
| Hosting | Cloudflare Pages | Fast cloud hosting |
| Design | Modern CSS | Responsive modern design |

### 3.2 Gateway Layer

| Component | Function |
|-----------|----------|
| Cloudflare | DDoS protection, SSL certificates, caching |
| Nginx | Reverse Proxy, load balancing, redirection |

### 3.3 Application Layer

| Component | Technology | Function |
|-----------|------------|----------|
| FastAPI | Python 3.12 | Main framework |
| Uvicorn | ASGI Server | High-performance async server |
| Pydantic | Data Validation | Input validation |

### 3.4 AI Layer

| Model | Function | Specifications |
|-------|----------|----------------|
| ChatGPT-5.2 | Poem generation | Advanced large language model |
| - | Temperature: 0.6 | For accurate and consistent results |

### 3.5 Data Layer

| Component | Type | Description |
|-----------|------|-------------|
| No database | - | System relies directly on OpenAI API |
| Environment Variables | .env | API keys storage |

---

## 4. Data Flow

### 4.1 Creating a New Poem

```
┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│              │  POST   │              │  API    │              │
│    Client    │────────▶│   FastAPI    │────────▶│   OpenAI     │
│   (Browser)  │         │   Server     │         │ ChatGPT-5.2  │
│              │         │              │         │              │
└──────────────┘         └──────────────┘         └──────┬───────┘
                                                         │
       ┌─────────────────────────────────────────────────┘
       │ Generated Poem (JSON)
       ▼
┌──────────────┐         ┌──────────────┐
│   Format     │         │   Display    │
│   Response   │────────▶│   to User    │
│              │         │              │
└──────────────┘         └──────────────┘
```

### 4.2 Request and Response Details

```
Request:                              Response:
{                                     {
  "choice": "إنشاء قصيدة",              "title": "Poem Title",
  "title": "Title",                     "verses": [
  "title_details": "Details",             "First verse...",
  "verses_count": 4,                      "Second verse..."
  "meter": "الطويل"                     ],
}                                       "meter": "الطويل"
                                      }
```

---

## 5. API Design

### 5.1 Endpoints

```yaml
Base URL: https://api.tkween.co

Endpoints:
  POST /poems:
    description: Create a new Arabic poem
    request_body:
      choice: "إنشاء قصيدة"
      title: string (required) - Poem title
      title_details: string (optional) - Additional details
      verses_count: integer (required) - Number of verses
      meter: string (optional) - Poetic meter
    responses:
      200: CreatePoemResponse
      400: ValidationError
      500: InternalError

  GET /stats:
    description: System information
    responses:
      200: StatsResponse

  GET /:
    description: API information
    responses:
      200: InfoResponse
```

### 5.2 Data Models

```python
# Poem Request
class PoemRequest:
    choice: Literal["إنشاء قصيدة"]
    title: str                    # Poem title (required)
    title_details: Optional[str]  # Additional details (optional)
    verses_count: int             # Number of verses (required)
    meter: Optional[str]          # Poetic meter (optional)

# Create Poem Response
class CreatePoemResponse:
    title: str          # Poem title
    verses: List[str]   # List of verses
    meter: str          # Poetic meter used
```

---

## 6. Infrastructure

### 6.1 System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Production Environment                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐         ┌─────────────────┐           │
│  │  Cloudflare     │         │  VPS Server     │           │
│  │  Pages          │         │  (Linode)       │           │
│  │  ─────────────  │         │  ─────────────  │           │
│  │  Frontend       │◀───────▶│  Backend API    │           │
│  │  tkween.co      │  HTTPS  │  api.tkween.co  │           │
│  └─────────────────┘         └────────┬────────┘           │
│                                       │                     │
│                                       │ API Call            │
│                                       ▼                     │
│                              ┌─────────────────┐           │
│                              │   OpenAI API    │           │
│                              │  (ChatGPT-5.2)  │           │
│                              └─────────────────┘           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Server Details

| Server | Function | Technology |
|--------|----------|------------|
| Frontend | User Interface | Cloudflare Pages |
| Backend | API Server | VPS + FastAPI + Uvicorn |
| AI | Poem Generation | OpenAI ChatGPT-5.2 |

### 6.3 Software Environment

```bash
# Server Environment
Python: 3.12+
FastAPI: 0.115.6
Uvicorn: 0.34.0
OpenAI SDK: 1.58.1
```

---

## 7. Poem Generation Algorithm

### 7.1 Algorithm Diagram

```
Algorithm: GeneratePoem(title, title_details, verses_count, meter)

Input:
  - title: Poem title
  - title_details: Additional details (optional)
  - verses_count: Number of verses
  - meter: Poetic meter (optional)

Output:
  - poem: Metered and rhymed poem

Procedure:
  1. prompt ← BuildPrompt(title, title_details, verses_count, meter)
  
  2. response ← OpenAI.ChatCompletion(
       model="gpt-5.2",
       messages=[system_prompt, user_prompt],
       temperature=0.6,
       response_format="json"
     )
  
  3. poem ← ParseJSON(response)
  
  4. RETURN {
       title: poem.title,
       verses: poem.verses,
       meter: poem.meter
     }
```

### 7.2 Prompt Engineering

```python
SYSTEM_PROMPT = """
You are an eloquent Arabic poet of the highest caliber.
You master prosody and Arabic poetic meters.
You maintain meter and rhyme in every poem.
You always respond in JSON format only.
"""

USER_PROMPT = """
You are an eloquent Arabic poet, master of prosody and rhyme.

═══════════════════════════════════════
Required: Create an eloquent Arabic poem
═══════════════════════════════════════

Poem Title: {title}
{title_details}
Number of Verses: {verses_count}
{meter_instruction}

═══════════════════════════════════════
Meter and Rhyme Rules:
═══════════════════════════════════════
1. Each verse consists of two hemistichs (صدر and عجز)
2. Prosodic meter is identical in all verses
3. Unified rhyme at the end of each verse
4. Rhyme letter (روي) remains constant

═══════════════════════════════════════
Language Rules:
═══════════════════════════════════════
1. Correct Classical Arabic grammar and morphology
2. Avoid obscure or unclear words
3. Clear and coherent meaning between verses
4. Eloquent expressions appropriate to the topic

Response in JSON format:
{
  "title": "...",
  "verses": ["Verse 1", "Verse 2", ...],
  "meter": "Meter name",
  "rhyme": "Rhyme letter"
}
"""
```

### 7.3 Model Parameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| model | gpt-5.2 | Best quality for Arabic poetry |
| temperature | 0.6 | Balance between creativity and accuracy |
| response_format | json_object | Ensure structured response |

---

## 8. System Requirements

### 8.1 Hardware Requirements (Backend Server)

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 1 GB | 2 GB |
| CPU | 1 core | 2 cores |
| Storage | 5 GB SSD | 10 GB SSD |
| Network | 50 Mbps | 100 Mbps |

### 8.2 Software Requirements

```yaml
Backend:
  Runtime:
    - Python: 3.12+
  
  Dependencies:
    - fastapi: 0.115.6
    - uvicorn: 0.34.0
    - openai: 1.58.1
    - pydantic: 2.10.4
    - python-dotenv: 1.0.1

Frontend:
  - HTML5
  - CSS3
  - Vanilla JavaScript (ES6+)

External Services:
  - OpenAI API (ChatGPT-5.2)
  - Cloudflare Pages (Frontend Hosting)
  - Cloudflare DNS
```

---

## 9. Security

### 9.1 Protection Layers

```
┌─────────────────────────────────────────────┐
│            Security Layers                   │
├─────────────────────────────────────────────┤
│ 1. Cloudflare WAF (Web Application Firewall)│
│ 2. DDoS Protection                          │
│ 3. SSL/TLS Encryption (HTTPS)               │
│ 4. CORS Policy                              │
│ 5. Input Validation (Pydantic)              │
│ 6. API Key Protection (.env)                │
│ 7. Rate Limiting (Future)                   │
└─────────────────────────────────────────────┘
```

### 9.2 CORS Configuration

```python
CORS_CONFIG = {
    "allow_origins": [
        "https://tkween.co",
        "https://www.tkween.co",
        "https://tkween.pages.dev"
    ],
    "allow_methods": ["*"],
    "allow_headers": ["*"],
    "allow_credentials": True
}
```

---

## 10. Scalability

### 10.1 Horizontal Scaling

```
                    ┌─────────────┐
                    │ Load        │
                    │ Balancer    │
                    └──────┬──────┘
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ Server 1 │    │ Server 2 │    │ Server 3 │
    │ FastAPI  │    │ FastAPI  │    │ FastAPI  │
    └────┬─────┘    └────┬─────┘    └────┬─────┘
         │               │               │
         └───────────────┼───────────────┘
                         │
                         ▼
              ┌──────────────────┐
              │    OpenAI API    │
              │  (ChatGPT-5.2)   │
              └──────────────────┘
```

### 10.2 Optimization Strategies

| Strategy | Description |
|----------|-------------|
| Response Caching | Temporarily store repeated poems |
| Connection Pooling | Reuse OpenAI connections |
| Async Processing | Asynchronous request handling |
| CDN | Serve frontend from Cloudflare Edge |

---

## 11. Monitoring & Logging

### 11.1 Monitored Metrics

```yaml
Performance Metrics:
  - Response Time (p50, p95, p99)
  - Request Rate (req/sec)
  - Error Rate (%)
  - CPU Usage (%)
  - Memory Usage (%)

Business Metrics:
  - Poems Generated/day
  - Unique Users/day
  - Popular Topics
  - Popular Meters
```

### 11.2 Log Format

```json
{
  "timestamp": "2025-12-20T10:30:00Z",
  "level": "INFO",
  "service": "tkween-api",
  "endpoint": "/poems",
  "method": "POST",
  "status": 200,
  "duration_ms": 3500,
  "request": {
    "choice": "إنشاء قصيدة",
    "title": "Love",
    "verses_count": 4
  }
}
```

---

## 12. Future Development Roadmap

### 12.1 Next Phase (v2.0)

- [ ] Improve meter and rhyme accuracy
- [ ] Add automatic diacritization for poems
- [ ] Support free verse and blank verse
- [ ] Add citation feature for classical poems
- [ ] Mobile app (iOS/Android)

### 12.2 Technical Improvements

- [ ] Implement Rate Limiting
- [ ] Add Redis Cache
- [ ] Improve response time
- [ ] WebSocket support for real-time response
- [ ] Add user authentication

---

## 13. References

1. OpenAI API Documentation - https://platform.openai.com/docs
2. OpenAI Prompt Engineering Guide - https://platform.openai.com/docs/guides/prompt-engineering
3. FastAPI Documentation - https://fastapi.tiangolo.com
4. Cloudflare Pages Documentation - https://developers.cloudflare.com/pages

---

## 14. Appendices

### 14.1 Supported Arabic Poetic Meters

| Meter (Arabic) | Meter (Transliterated) | Metrical Pattern |
|----------------|------------------------|------------------|
| الطويل | Al-Taweel | فعولن مفاعيلن فعولن مفاعلن |
| البسيط | Al-Baseet | مستفعلن فاعلن مستفعلن فعلن |
| الكامل | Al-Kamil | متفاعلن متفاعلن متفاعلن |
| الوافر | Al-Wafir | مفاعلتن مفاعلتن فعولن |
| الخفيف | Al-Khafif | فاعلاتن مستفعلن فاعلاتن |
| الرمل | Al-Ramal | فاعلاتن فاعلاتن فاعلاتن |
| السريع | Al-Saree' | مستفعلن مستفعلن مفعولات |
| المنسرح | Al-Munsarih | مستفعلن مفعولات مستفعلن |
| المتقارب | Al-Mutaqarib | فعولن فعولن فعولن فعولن |
| المتدارك | Al-Mutadarik | فاعلن فاعلن فاعلن فاعلن |

---

**Prepared by:** Tkween AI Team  
**Date:** December 20, 2025  
**Version:** 2.0


