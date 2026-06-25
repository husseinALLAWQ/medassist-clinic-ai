# MedAssist Clinic AI Ultra v3

نسخة متعددة المراحل كما طلبت:

1. Intake:
   - القصة
   - الأعراض
   - الإحساس
   - الأدوية
   - الحساسية
   - vitals

2. AI Questions:
   - أسئلة للمريض
   - red flags
   - أمان دوائي أولي

3. Clinical Exam:
   - نصائح للفحص السريري
   - خانة لإدخال ماذا وجد الطبيب

4. Preliminary Dx & Workup:
   - تحليل بعد الأسئلة والفحص
   - differential diagnosis
   - فحوصات وصور مطلوبة حسب الحالة

5. Results:
   - إدخال labs يدويًا
   - إدخال MRI/X-ray/CT reports
   - رفع PDF/images/Excel/CSV/Word

6. Final Support:
   - تحليل بعد النتائج
   - updated differential
   - هل يحتاج فحوصات/صور إضافية
   - medication support مع موانع وتحذيرات وليس وصفة نهائية

7. Report:
   - Download report
   - JSON/Markdown

## Streamlit Secrets

ضع المفتاح في Streamlit secrets:

```toml
OPENAI_API_KEY = "sk-..."
```

## Main file path

`app.py`
