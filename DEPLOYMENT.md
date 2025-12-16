# 🚀 دليل نشر Arabic Poems API

## الملفات المطلوبة للنشر

```
├── main.py              # الكود الرئيسي
├── requirements.txt     # المتطلبات
├── Dockerfile          # لـ Docker
├── docker-compose.yml  # لـ Docker Compose
├── .env                # متغيرات البيئة (OPENAI_API_KEY)
└── arabic_poems_db/    # قاعدة البيانات (مهم جداً!)
```

---

## 🌐 الخيار 1: Railway (مجاني للمشاريع الصغيرة)

### الخطوات:

```bash
# 1. تثبيت Railway CLI
npm install -g @railway/cli

# 2. تسجيل الدخول
railway login

# 3. الانتقال لمجلد المشروع
cd /Users/abdullateefalbahlal/Documents/AI/Dataset

# 4. إنشاء مشروع
railway init

# 5. إضافة متغير البيئة
railway variables set OPENAI_API_KEY=sk-xxx

# 6. رفع المشروع
railway up

# 7. الحصول على الرابط
railway domain
```

**الرابط سيكون:** `https://your-project.up.railway.app`

---

## 🐳 الخيار 2: Docker على VPS

### المتطلبات:
- سيرفر VPS (DigitalOcean, AWS, Linode)
- Docker مثبت

### الخطوات:

```bash
# 1. نسخ الملفات للسيرفر
scp -r /Users/abdullateefalbahlal/Documents/AI/Dataset user@your-server:/home/user/

# 2. الاتصال بالسيرفر
ssh user@your-server

# 3. الانتقال للمجلد
cd /home/user/Dataset

# 4. إنشاء ملف .env
echo "OPENAI_API_KEY=sk-xxx" > .env

# 5. بناء وتشغيل الـ Container
docker-compose up -d --build

# 6. التحقق من التشغيل
docker-compose logs -f
```

### أوامر مفيدة:

```bash
# إيقاف السيرفر
docker-compose down

# إعادة التشغيل
docker-compose restart

# عرض اللوجات
docker-compose logs -f

# تحديث الكود
git pull && docker-compose up -d --build
```

---

## ☁️ الخيار 3: Render.com

### الخطوات:

1. سجّل في [render.com](https://render.com)
2. اربط حساب GitHub
3. أنشئ "New Web Service"
4. اختر المستودع
5. أضف الإعدادات:

```
Build Command: pip install -r requirements.txt
Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT
```

6. أضف Environment Variable:
   - `OPENAI_API_KEY` = مفتاحك

---

## 🔧 الخيار 4: DigitalOcean App Platform

### الخطوات:

1. سجّل في [DigitalOcean](https://digitalocean.com)
2. أنشئ "App"
3. اربط GitHub
4. اختر:
   - Type: Web Service
   - Build Command: `pip install -r requirements.txt`
   - Run Command: `uvicorn main:app --host 0.0.0.0 --port 8080`

5. أضف Environment:
   - `OPENAI_API_KEY`

---

## ⚠️ ملاحظات مهمة

### 1. قاعدة البيانات (arabic_poems_db)

قاعدة البيانات كبيرة (~2GB). خيارات التعامل معها:

**الخيار أ:** رفعها مع الكود (بطيء)
```bash
# ضغطها أولاً
tar -czvf arabic_poems_db.tar.gz arabic_poems_db/
```

**الخيار ب:** استخدام Volume في Docker
```yaml
volumes:
  - ./arabic_poems_db:/app/arabic_poems_db
```

**الخيار ج:** رفعها لـ S3/Cloud Storage

### 2. OPENAI_API_KEY

⚠️ **لا ترفع ملف `.env` لـ GitHub!**

أضف `.env` لـ `.gitignore`:
```bash
echo ".env" >> .gitignore
```

### 3. الـ Port

بعض المنصات تستخدم متغير `PORT`:
```python
import os
port = int(os.getenv("PORT", 8000))
uvicorn.run(app, host="0.0.0.0", port=port)
```

---

## 🔒 إعداد HTTPS (SSL)

### مع Nginx:

```nginx
server {
    listen 80;
    server_name api.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl;
    server_name api.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### تثبيت SSL مع Let's Encrypt:

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d api.yourdomain.com
```

---

## 📊 مراقبة الأداء

### عرض استخدام الموارد:

```bash
# Docker stats
docker stats

# الذاكرة والـ CPU
htop
```

### اللوجات:

```bash
# Docker logs
docker-compose logs -f arabic-poems-api

# أو مباشرة
docker logs -f container_name
```

---

## 🔄 تحديث الكود

```bash
# 1. سحب التحديثات
git pull origin main

# 2. إعادة بناء الـ Container
docker-compose up -d --build

# أو مع Railway
railway up
```

---

## ✅ التحقق من النشر

```bash
# فحص الـ API
curl https://your-domain.com/

# فحص الإحصائيات
curl https://your-domain.com/stats

# اختبار إنشاء قصيدة
curl -X POST "https://your-domain.com/poems" \
  -H "Content-Type: application/json" \
  -d '{"choice": "إنشاء قصيدة", "topic": "الحب", "verses_count": 2}'
```

---

📅 آخر تحديث: ديسمبر 2024

