import os
import json
import random
import numpy as np
from enum import Enum
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import chromadb
from dotenv import load_dotenv

# تحميل متغيرات البيئة من ملف .env
load_dotenv()

# ============================================
# إعداد التطبيق
# ============================================
app = FastAPI(
    title="Arabic Poems API",
    description="API لإنشاء قصائد عربية أو الاستشهاد بقصائد من قاعدة البيانات",
    version="1.0.0"
)

# إعداد OpenAI
# تأكد من تعيين متغير البيئة OPENAI_API_KEY في ملف .env
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("يرجى تعيين OPENAI_API_KEY في ملف .env")
openai_client = OpenAI(api_key=api_key)

# إعداد ChromaDB
chroma_client = chromadb.PersistentClient(path="./arabic_poems_db")
collection = chroma_client.get_or_create_collection(name="arabic_poems")


# ============================================
# النماذج (Models)
# ============================================
class PoemChoice(str, Enum):
    CREATE = "إنشاء قصيدة"
    QUOTE = "استشهاد بقصيدة"


class PoemRequest(BaseModel):
    choice: PoemChoice
    topic: str
    verses_count: int
    meter: Optional[str] = None
    poet: Optional[str] = None


class VerificationResult(BaseModel):
    is_valid: bool
    meter_check: str
    rhyme_check: str
    language_check: str
    issues: List[str] = []
    attempts: int = 1


class CreatePoemResponse(BaseModel):
    verses: List[str]
    meter: str
    topic: str
    verification: Optional[VerificationResult] = None


class QuotePoemResponse(BaseModel):
    verses: List[str]
    meter: str
    topic: str
    poet: str
    poet_era: Optional[str] = None
    title: Optional[str] = None


# ============================================
# دوال مساعدة
# ============================================
def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """حساب embedding لنص معين باستخدام OpenAI API"""
    text = text.replace("\n", " ")
    response = openai_client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


def verify_poem(verses: List[str], meter: str) -> dict:
    """
    التحقق من صحة القصيدة (الوزن، القافية، اللغة)
    يستخدم OpenAI للمراجعة
    """
    verses_text = "\n".join(verses)
    
    prompt = f"""أنت خبير في علم العروض والقافية العربية.
راجع القصيدة التالية وتحقق من:
1. صحة الوزن على بحر {meter}
2. اتساق القافية في جميع الأبيات
3. سلامة اللغة العربية الفصحى

القصيدة:
{verses_text}

أرجع التقييم بصيغة JSON فقط:
{{
    "is_valid": true/false,
    "meter_check": "✅ موزونة على {meter}" أو "❌ كسر في الوزن",
    "rhyme_check": "✅ القافية موحدة" أو "❌ اختلاف في القافية",
    "language_check": "✅ لغة فصيحة سليمة" أو "❌ أخطاء لغوية",
    "issues": ["قائمة بالمشاكل المحددة إن وجدت"],
    "suggestions": ["اقتراحات للتصحيح إن وجدت"]
}}"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "أنت خبير في علم العروض والقافية. ترد بصيغة JSON فقط."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)


def fix_poem(verses: List[str], meter: str, issues: List[str]) -> dict:
    """
    تصحيح القصيدة بناءً على المشاكل المحددة
    """
    verses_text = "\n".join(verses)
    issues_text = "\n".join(f"- {issue}" for issue in issues)
    
    prompt = f"""أنت شاعر عربي فصيح متمكن من العروض والقافية.
صحح القصيدة التالية مع الحفاظ على المعنى قدر الإمكان.

القصيدة الأصلية:
{verses_text}

البحر المطلوب: {meter}

المشاكل المطلوب تصحيحها:
{issues_text}

أرجع القصيدة المصححة بصيغة JSON:
{{
    "verses": ["البيت الأول المصحح", "البيت الثاني المصحح", ...],
    "meter": "{meter}",
    "changes_made": ["قائمة بالتغييرات التي تمت"]
}}"""

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "أنت شاعر عربي فصيح متخصص في تصحيح الشعر. ترد بصيغة JSON فقط."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)


def generate_poem_raw(topic: str, verses_count: int, meter: Optional[str] = None) -> dict:
    """إنشاء قصيدة خام باستخدام OpenAI (بدون مراجعة)"""
    
    meter_instruction = f"- البحر: {meter}" if meter else "- البحر: اختر البحر المناسب للموضوع"
    
    prompt = f"""أنت شاعر عربي فصيح متمكن من الشعر العربي الفصيح وأوزانه وبحوره.
أنشئ قصيدة عربية فصيحة بالمواصفات التالية:
- الموضوع: {topic}
- عدد الأبيات: {verses_count}
{meter_instruction}

يجب أن تكون القصيدة:
1. موزونة على البحر المحدد
2. ذات قافية موحدة
3. بلغة عربية فصيحة جميلة

أرجع الرد بصيغة JSON فقط بالشكل التالي:
{{
    "verses": ["البيت الأول", "البيت الثاني", ...],
    "meter": "اسم البحر",
    "topic": "موضوع القصيدة"
}}

لا تضف أي نص آخر خارج JSON."""

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "أنت شاعر عربي فصيح. ترد دائماً بصيغة JSON فقط."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.9,
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)


def create_poem_with_openai(topic: str, verses_count: int, meter: Optional[str] = None, max_attempts: int = 3) -> dict:
    """
    إنشاء قصيدة جديدة مع مراجعة وتصحيح
    
    - يولّد القصيدة
    - يراجعها للتحقق من الوزن والقافية
    - يصححها إذا لزم الأمر (حتى max_attempts محاولات)
    """
    
    attempt = 1
    verification_result = None
    
    while attempt <= max_attempts:
        # توليد القصيدة
        if attempt == 1:
            result = generate_poem_raw(topic, verses_count, meter)
        else:
            # في المحاولات اللاحقة، نحاول تصحيح القصيدة
            if verification_result and verification_result.get("issues"):
                fixed = fix_poem(
                    verses=result["verses"],
                    meter=result["meter"],
                    issues=verification_result.get("issues", [])
                )
                result["verses"] = fixed.get("verses", result["verses"])
            else:
                # إعادة التوليد من جديد
                result = generate_poem_raw(topic, verses_count, meter)
        
        # مراجعة القصيدة
        verification_result = verify_poem(result["verses"], result["meter"])
        
        # إذا كانت صحيحة، نرجعها
        if verification_result.get("is_valid", False):
            result["verification"] = {
                "is_valid": True,
                "meter_check": verification_result.get("meter_check", "✅ موزونة"),
                "rhyme_check": verification_result.get("rhyme_check", "✅ القافية موحدة"),
                "language_check": verification_result.get("language_check", "✅ لغة سليمة"),
                "issues": [],
                "attempts": attempt
            }
            return result
        
        attempt += 1
    
    # إذا فشلت كل المحاولات، نرجع آخر نتيجة مع معلومات المراجعة
    result["verification"] = {
        "is_valid": False,
        "meter_check": verification_result.get("meter_check", "⚠️ قد يكون هناك كسر"),
        "rhyme_check": verification_result.get("rhyme_check", "⚠️ تحقق من القافية"),
        "language_check": verification_result.get("language_check", "✅ لغة مقبولة"),
        "issues": verification_result.get("issues", []),
        "attempts": max_attempts
    }
    
    return result


def get_poet_poems_cached(poet_name: str, limit: int = 200) -> dict:
    """جلب قصائد شاعر محدد من قاعدة البيانات"""
    # البحث في قاعدة البيانات على دفعات للعثور على الشاعر
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
        
        # إذا وجدنا قصائد كافية، نتوقف
        if len(poet_poems["ids"]) >= limit:
            break
    
    return poet_poems


def search_poems_in_chromadb(topic: str, verses_count: int, meter: Optional[str] = None, poet: Optional[str] = None) -> dict:
    """البحث عن قصائد في ChromaDB"""
    
    # إذا تم تحديد شاعر، نبحث في قصائده ثم نرتبها حسب الموضوع
    if poet:
        poet_results = get_poet_poems_cached(poet, limit=200)
        
        if poet_results["documents"]:
            # حساب الـ embedding للموضوع
            query_embedding = get_embedding(topic)
            
            # حساب التشابه يدوياً وترتيب النتائج
            similarities = []
            for i, emb in enumerate(poet_results["embeddings"]):
                if emb is not None and len(emb) > 0:
                    sim = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
                    similarities.append((i, sim))
            
            if similarities:
                # ترتيب حسب التشابه
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                for idx, sim in similarities:
                    metadata = poet_results["metadatas"][idx]
                    
                    # تحقق من البحر إذا محدد
                    if meter and meter not in metadata.get("poem_meter", ""):
                        continue
                    
                    document = poet_results["documents"][idx]
                    all_verses = [v.strip() for v in document.split("\n") if v.strip()]
                    return {
                        "verses": all_verses[:verses_count],
                        "meter": metadata.get("poem_meter", "غير محدد"),
                        "topic": metadata.get("poem_theme", topic),
                        "poet": metadata.get("poet_name", "غير معروف"),
                        "poet_era": metadata.get("poet_era", ""),
                        "title": metadata.get("poem_title", "")
                    }
            
            # إذا لم نجد مع فلتر البحر، نرجع أي قصيدة للشاعر
            if poet_results["documents"]:
                idx = random.randint(0, len(poet_results["documents"]) - 1)
                metadata = poet_results["metadatas"][idx]
                document = poet_results["documents"][idx]
                all_verses = [v.strip() for v in document.split("\n") if v.strip()]
                return {
                    "verses": all_verses[:verses_count],
                    "meter": metadata.get("poem_meter", "غير محدد"),
                    "topic": metadata.get("poem_theme", topic),
                    "poet": metadata.get("poet_name", "غير معروف"),
                    "poet_era": metadata.get("poet_era", ""),
                    "title": metadata.get("poem_title", "")
                }
    
    # البحث العادي بالـ embedding
    query_embedding = get_embedding(topic)
    
    # جلب نتائج كثيرة للفلترة
    n_results = 500 if meter else 100
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    if not results["documents"] or not results["documents"][0]:
        raise HTTPException(status_code=404, detail="لم يتم العثور على قصائد مطابقة")
    
    # جمع كل النتائج المطابقة
    matching_results = []
    
    for i, metadata in enumerate(results["metadatas"][0]):
        poem_meter = metadata.get("poem_meter", "")
        
        # تحقق من البحر إذا محدد
        if meter:
            if meter not in poem_meter:
                continue
        
        # نتيجة مطابقة - نضيفها للقائمة
        matching_results.append(i)
    
    # اختيار نتيجة عشوائية من النتائج المطابقة
    if matching_results:
        chosen_idx = random.choice(matching_results)
        document = results["documents"][0][chosen_idx]
        metadata = results["metadatas"][0][chosen_idx]
        all_verses = [v.strip() for v in document.split("\n") if v.strip()]
        return {
            "verses": all_verses[:verses_count],
            "meter": metadata.get("poem_meter", "غير محدد"),
            "topic": metadata.get("poem_theme", topic),
            "poet": metadata.get("poet_name", "غير معروف"),
            "poet_era": metadata.get("poet_era", ""),
            "title": metadata.get("poem_title", "")
        }
    
    # ما لقينا، نرجع نتيجة عشوائية من أول 10
    chosen_idx = random.randint(0, min(9, len(results["documents"][0]) - 1))
    document = results["documents"][0][chosen_idx]
    metadata = results["metadatas"][0][chosen_idx]
    all_verses = [v.strip() for v in document.split("\n") if v.strip()]
    
    return {
        "verses": all_verses[:verses_count],
        "meter": metadata.get("poem_meter", "غير محدد"),
        "topic": metadata.get("poem_theme", topic),
        "poet": metadata.get("poet_name", "غير معروف"),
        "poet_era": metadata.get("poet_era", ""),
        "title": metadata.get("poem_title", "")
    }


# ============================================
# الـ Endpoints
# ============================================
@app.get("/")
async def root():
    """الصفحة الرئيسية"""
    return {
        "message": "مرحباً بك في API القصائد العربية",
        "endpoints": {
            "POST /poems": "إنشاء قصيدة أو استشهاد بقصيدة"
        }
    }


@app.post("/poems")
async def handle_poem_request(request: PoemRequest):
    """
    معالجة طلبات القصائد
    
    - إنشاء قصيدة: يستخدم OpenAI لتوليد قصيدة جديدة
    - استشهاد بقصيدة: يبحث في قاعدة البيانات عن قصائد مشابهة
    """
    
    if request.choice == PoemChoice.CREATE:
        # إنشاء قصيدة جديدة باستخدام OpenAI مع المراجعة
        try:
            result = create_poem_with_openai(
                topic=request.topic,
                verses_count=request.verses_count,
                meter=request.meter,
                max_attempts=3
            )
            
            # تحويل معلومات التحقق للنموذج
            verification = None
            if result.get("verification"):
                v = result["verification"]
                verification = VerificationResult(
                    is_valid=v.get("is_valid", False),
                    meter_check=v.get("meter_check", ""),
                    rhyme_check=v.get("rhyme_check", ""),
                    language_check=v.get("language_check", ""),
                    issues=v.get("issues", []),
                    attempts=v.get("attempts", 1)
                )
            
            return CreatePoemResponse(
                verses=result["verses"],
                meter=result["meter"],
                topic=result["topic"],
                verification=verification
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"خطأ في إنشاء القصيدة: {str(e)}")
    
    elif request.choice == PoemChoice.QUOTE:
        # البحث عن قصيدة في ChromaDB
        try:
            result = search_poems_in_chromadb(
                topic=request.topic,
                verses_count=request.verses_count,
                meter=request.meter,
                poet=request.poet
            )
            return QuotePoemResponse(
                verses=result["verses"],
                meter=result["meter"],
                topic=result["topic"],
                poet=result["poet"],
                poet_era=result.get("poet_era"),
                title=result.get("title")
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"خطأ في البحث: {str(e)}")


@app.get("/stats")
async def get_stats():
    """إحصائيات قاعدة البيانات"""
    total = collection.count()
    return {
        "total_poems": total,
        "database_path": "./arabic_poems_db",
        "dataset_source": "arbml/ashaar (Hugging Face)",
        "features": ["البحث الدلالي", "فلترة حسب الشاعر", "فلترة حسب البحر"]
    }


# ============================================
# تشغيل السيرفر
# ============================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

