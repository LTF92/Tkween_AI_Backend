# استخدام Python 3.12
FROM python:3.12-slim

# تعيين مجلد العمل
WORKDIR /app

# نسخ ملف المتطلبات
COPY requirements.txt .

# تثبيت المتطلبات
RUN pip install --no-cache-dir -r requirements.txt

# نسخ الكود
COPY main.py .
COPY .env .

# نسخ قاعدة البيانات
COPY arabic_poems_db/ ./arabic_poems_db/

# فتح البورت
EXPOSE 8000

# تشغيل السيرفر
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

