# MedAssist Neuro-General AutoEvidence Guard v4.7

هذه النسخة تضيف بحثًا أوتوماتيكيًا في المصادر الطبية المفتوحة قبل إعطاء النتيجة.

## الجديد
- Automatic Evidence Web Search
- بحث تلقائي قبل التحليل عبر OpenAI Responses API web_search tool
- اختيار نطاق المصادر:
  - Authoritative medical domains only
  - Broad web search
- Evidence search model مستقل، افتراضيًا: gpt-4.1-mini
- Evidence Verification يميز بين:
  - automatic_web_search / verified_from_web_search
  - uploaded_guideline_or_reference / verified_from_uploaded_material
  - built_in_guideline_framework / framework_based_not_live_checked
  - clinical_reasoning_only
  - not_verified_live
  - needs_specialist_review

## مهم
- لا يستطيع فتح UpToDate أو مصادر مدفوعة إلا إذا كان لديك وصول مرخص/API خاص.
- البحث يعتمد على المصادر الطبية المفتوحة والمتاحة على الويب.
- يجب على الطبيب مراجعة المصادر والحكم النهائي.

## التسلسل
1. Intake
2. Neuro Screen
3. General Medicine Screen
4. Evidence / Guidelines
5. Questions Before Exam
6. Exam Protocol: كيف أفحص؟
7. Enter Exam Findings
8. Questions After Exam
9. Dx & Workup
10. Results / Imaging
11. Full Review / Medication
12. Report / Search

## Streamlit Secrets

```toml
OPENAI_API_KEY = "sk-..."
```

## Main file
app.py
