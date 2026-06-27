# MedAssist Neuro-General AutoEvidence Guard v4.7.1

هذه نسخة إصلاحية من v4.7.

## سبب الإصلاح
في بعض موديلات OpenAI مثل `gpt-4.1-mini` يظهر خطأ:
`Parameter 'filters' not supported with model...`

لذلك أزلنا `filters` من أداة `web_search`.
الآن اختيار "Authoritative medical domains only" يتم عبر تعليمات داخل نص البحث، وليس عبر API filters.

## الجديد في v4.7.1
- Automatic Evidence Web Search بدون filters غير مدعومة
- Fallback تلقائي إذا فشل البحث:
  1. نفس الموديل بدون `tool_choice=required`
  2. fallback إلى `gpt-4o-mini`
- Evidence Verification يوضح هل النتيجة:
  - verified_from_web_search
  - framework_based_not_live_checked
  - not_live_verified
  - needs_manual_reference_check

## مهم
- البحث في المصادر المفتوحة المتاحة على الويب.
- لا يفتح UpToDate أو مصادر مدفوعة إلا إذا كان لديك وصول مرخص/API.
- القرار النهائي للطبيب.

## Streamlit Secrets

```toml
OPENAI_API_KEY = "sk-..."
```

## Main file
app.py
