import os
import json
import random
import numpy as np
from enum import Enum
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import chromadb
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================
# Application Setup
# ============================================
app = FastAPI(
    title="Arabic Poems API",
    description="API for creating Arabic poems or quoting poems from the database",
    version="1.0.0"
)

# CORS setup to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://tkween.pages.dev",
        "https://tkween.co",
        "https://www.tkween.co",
        "http://localhost:8080",
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI Setup
# Make sure to set OPENAI_API_KEY in .env file
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set OPENAI_API_KEY in .env file")
openai_client = OpenAI(api_key=api_key)

# ChromaDB Setup
chroma_client = chromadb.PersistentClient(path="./arabic_poems_db")
collection = chroma_client.get_or_create_collection(name="arabic_poems")


# ============================================
# Models
# ============================================
class PoemChoice(str, Enum):
    CREATE = "إنشاء قصيدة"
    QUOTE = "استشهاد بقصيدة"


class PoemRequest(BaseModel):
    choice: PoemChoice
    title: str                          # Poem title
    title_details: Optional[str] = None # Additional details about the title (optional)
    verses_count: int                   # Number of verses
    meter: Optional[str] = None         # Poetic meter
    poet: Optional[str] = None          # Poet name (for quoting only)


class CreatePoemResponse(BaseModel):
    title: str
    verses: List[str]
    meter: str


class QuotePoemResponse(BaseModel):
    title: str
    verses: List[str]
    meter: str
    poet: str
    poet_era: Optional[str] = None
    theme: Optional[str] = None


# ============================================
# Helper Functions
# ============================================
def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """Calculate embedding for a given text using OpenAI API"""
    text = text.replace("\n", " ")
    response = openai_client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


def create_poem_with_openai(title: str, title_details: Optional[str], verses_count: int, meter: Optional[str] = None) -> dict:
    """
    Create an eloquent Arabic poem using OpenAI ChatGPT-5.2
    Optimized according to OpenAI prompt engineering guidelines
    """
    
    # Build inputs in structured format
    meter_text = meter if meter else "اختر الأنسب (الطويل/البسيط/الكامل/الوافر/الخفيف)"
    details_text = f"\nالتفاصيل: {title_details}" if title_details else ""
    
    # Concise and clear system message (OpenAI Best Practice)
    system_message = """أنت شاعر عربي فصيح متخصص في الشعر العمودي الموزون.
التزم دائماً بـ: الوزن العروضي الصحيح، القافية الموحدة، اللغة الفصيحة السليمة.
أجب بـ JSON فقط."""

    # Structured user prompt
    prompt = f"""ألّف قصيدة عربية:

العنوان: {title}{details_text}
الأبيات: {verses_count}
البحر: {meter_text}

المتطلبات:
• كل بيت = صدر + "    " + عجز
• قافية موحدة (حرف روي ثابت)
• لغة فصيحة واضحة المعنى
• لا تستخدم أي ألفاظ غريبة أو مبهمة
• لا تضع حركات في الأبيات إلا المهم منها لوضوح المعنى

مثال للشكل:
"قِفا نَبك من ذكرى حبيبٍ ومنزلِ    بِسِقطِ اللِوى بين الدَخولِ فَحَومَلِ"

{{
  "title": "{title}",
  "verses": ["البيت الأول...", "البيت الثاني..."],
  "meter": "اسم البحر"
}}"""

    response = openai_client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        temperature=0.6,  # Lower = more accurate (OpenAI recommendation)
        response_format={"type": "json_object"}
    )
    
    result = json.loads(response.choices[0].message.content)
    
    return {
        "title": result.get("title", title),
        "verses": result.get("verses", []),
        "meter": result.get("meter", "غير محدد")
    }


def get_poet_poems_cached(poet_name: str, limit: int = 200) -> dict:
    """Fetch poems by a specific poet from the database"""
    # Search database in batches to find the poet
    batch_size = 10000
    offset = 0
    poet_poems = {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
    
    while offset < collection.count() and len(poet_poems["ids"]) < limit:
        data = collection.get(
            include=["metadatas", "documents", "embeddings"],
            limit=batch_size,
            offset=offset
        )
        
        for i, meta in enumerate(data["metadatas"]):
            if poet_name in meta.get("poet_name", ""):
                poet_poems["ids"].append(data["ids"][i])
                poet_poems["documents"].append(data["documents"][i])
                poet_poems["metadatas"].append(meta)
                if data["embeddings"] is not None and len(data["embeddings"]) > i:
                    poet_poems["embeddings"].append(data["embeddings"][i])
                
                if len(poet_poems["ids"]) >= limit:
                    break
        
        offset += batch_size
        
        # If we found enough poems, stop
        if len(poet_poems["ids"]) >= limit:
            break
    
    return poet_poems


def search_poems_in_chromadb(title: str, verses_count: int, meter: Optional[str] = None, poet: Optional[str] = None) -> dict:
    """Search for poems in ChromaDB using semantic search"""
    
    # If a poet is specified, search their poems and rank by topic
    if poet:
        poet_results = get_poet_poems_cached(poet, limit=200)
        
        if poet_results["documents"]:
            # Calculate embedding for the title
            query_embedding = get_embedding(title)
            
            # Calculate similarity manually and sort results
            similarities = []
            for i, emb in enumerate(poet_results["embeddings"]):
                if emb is not None and len(emb) > 0:
                    sim = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
                    similarities.append((i, sim))
            
            if similarities:
                # Sort by similarity
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                for idx, sim in similarities:
                    metadata = poet_results["metadatas"][idx]
                    
                    # Check meter if specified
                    if meter and meter not in metadata.get("poem_meter", ""):
                        continue
                    
                    document = poet_results["documents"][idx]
                    all_verses = [v.strip() for v in document.split("\n") if v.strip()]
                    return {
                        "title": metadata.get("poem_title", title),
                        "verses": all_verses[:verses_count],
                        "meter": metadata.get("poem_meter", "غير محدد"),
                        "poet": metadata.get("poet_name", "غير معروف"),
                        "poet_era": metadata.get("poet_era", ""),
                        "theme": metadata.get("poem_theme", "")
                    }
            
            # If no match with meter filter, return any poem by the poet
            if poet_results["documents"]:
                idx = random.randint(0, len(poet_results["documents"]) - 1)
                metadata = poet_results["metadatas"][idx]
                document = poet_results["documents"][idx]
                all_verses = [v.strip() for v in document.split("\n") if v.strip()]
                return {
                    "title": metadata.get("poem_title", title),
                    "verses": all_verses[:verses_count],
                    "meter": metadata.get("poem_meter", "غير محدد"),
                    "poet": metadata.get("poet_name", "غير معروف"),
                    "poet_era": metadata.get("poet_era", ""),
                    "theme": metadata.get("poem_theme", "")
                }
    
    # Regular search using embedding
    query_embedding = get_embedding(title)
    
    # Fetch many results for filtering
    n_results = 500 if meter else 100
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    if not results["documents"] or not results["documents"][0]:
        raise HTTPException(status_code=404, detail="No matching poems found")
    
    # Collect all matching results
    matching_results = []
    
    for i, metadata in enumerate(results["metadatas"][0]):
        poem_meter = metadata.get("poem_meter", "")
        
        # Check meter if specified
        if meter:
            if meter not in poem_meter:
                continue
        
        # Matching result - add to list
        matching_results.append(i)
    
    # Choose a random result from matching results
    if matching_results:
        chosen_idx = random.choice(matching_results)
        document = results["documents"][0][chosen_idx]
        metadata = results["metadatas"][0][chosen_idx]
        all_verses = [v.strip() for v in document.split("\n") if v.strip()]
        return {
            "title": metadata.get("poem_title", title),
            "verses": all_verses[:verses_count],
            "meter": metadata.get("poem_meter", "غير محدد"),
            "poet": metadata.get("poet_name", "غير معروف"),
            "poet_era": metadata.get("poet_era", ""),
            "theme": metadata.get("poem_theme", "")
        }
    
    # No match found, return random result from first 10
    chosen_idx = random.randint(0, min(9, len(results["documents"][0]) - 1))
    document = results["documents"][0][chosen_idx]
    metadata = results["metadatas"][0][chosen_idx]
    all_verses = [v.strip() for v in document.split("\n") if v.strip()]
    
    return {
        "title": metadata.get("poem_title", title),
        "verses": all_verses[:verses_count],
        "meter": metadata.get("poem_meter", "غير محدد"),
        "poet": metadata.get("poet_name", "غير معروف"),
        "poet_era": metadata.get("poet_era", ""),
        "theme": metadata.get("poem_theme", "")
    }


# ============================================
# Endpoints
# ============================================
@app.get("/")
async def root():
    """Home page"""
    return {
        "message": "مرحباً بك في API القصائد العربية",
        "endpoints": {
            "POST /poems": "إنشاء قصيدة أو استشهاد بقصيدة"
        }
    }


@app.post("/poems")
async def handle_poem_request(request: PoemRequest):
    """
    Handle poem requests
    
    - Create poem: Uses OpenAI to generate a new poem
    - Quote poem: Searches the database for similar poems
    
    Required fields:
    - title: Poem title
    - title_details: Additional details (optional)
    - verses_count: Number of verses
    - meter: Poetic meter (optional)
    - poet: Poet name (for quoting only)
    """
    
    if request.choice == PoemChoice.CREATE:
        # Create a new poem using OpenAI
        try:
            result = create_poem_with_openai(
                title=request.title,
                title_details=request.title_details,
                verses_count=request.verses_count,
                meter=request.meter
            )
            
            return CreatePoemResponse(
                title=result["title"],
                verses=result["verses"],
                meter=result["meter"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error creating poem: {str(e)}")
    
    elif request.choice == PoemChoice.QUOTE:
        # Search for a poem in ChromaDB
        try:
            result = search_poems_in_chromadb(
                title=request.title,
                verses_count=request.verses_count,
                meter=request.meter,
                poet=request.poet
            )
            return QuotePoemResponse(
                title=result["title"],
                verses=result["verses"],
                meter=result["meter"],
                poet=result["poet"],
                poet_era=result.get("poet_era"),
                theme=result.get("theme")
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error searching: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Database statistics"""
    total = collection.count()
    return {
        "total_poems": total,
        "database_path": "./arabic_poems_db",
        "dataset_source": "arbml/ashaar (Hugging Face)",
        "features": ["Semantic search", "Filter by poet", "Filter by meter"]
    }


# ============================================
# Run Server
# ============================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
