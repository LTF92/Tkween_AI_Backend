from datasets import load_dataset

# تحميل مجموعة بيانات القصائد العربية من Hugging Face
ds = load_dataset("alwalid54321/Arabic_Poems")

# عرض معلومات عن مجموعة البيانات
print(ds)

