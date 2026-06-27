# MedAssist Level-5 AutoEvidence GPT-5 Strong v5.1

نسخة GPT-5 القوية من Level-5 AutoEvidence.

## الافتراضي
- Clinical analysis model: `gpt-5.5`
- Evidence search model: `gpt-5.4-mini`

## Model Power Preset
في الـ sidebar يوجد:
- GPT-5.5 strongest
- GPT-5.4 strong
- GPT-5.4 mini balanced
- Manual

إذا حسابك لا يملك وصولًا إلى النموذج المختار، التطبيق يحاول fallback تلقائيًا:
1. selected model
2. gpt-5.5
3. gpt-5.4
4. gpt-5.4-mini
5. gpt-4o
6. gpt-4o-mini

## ملاحظة مهمة
قد تختلف أسماء/إتاحة الموديلات حسب حساب OpenAI API لديك. إذا ظهر خطأ model not found أو access denied، اختر Manual واكتب اسم الموديل المتاح لديك.

## وظيفة البرنامج
- محرك طبي عام وليس خاصًا بحالة واحدة.
- يبحث أوتوماتيكيًا في المصادر الطبية المفتوحة.
- يعطي:
  - Safety Gate
  - Evidence Verification
  - Clinical Reasoning Summary
  - Questions
  - Detailed Clinical Exam
  - Differential Diagnosis
  - Recommended Workup
  - Comprehensive Diagnostic Map
  - Imaging Decisions
  - Action Pathway
  - Medication Safety بدون جرعات

## Streamlit Secrets
```toml
OPENAI_API_KEY = "sk-..."
```

## Main file
app.py
