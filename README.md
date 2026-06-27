# MedAssist Neuro-General Evidence Guard v4.6

هذه النسخة تضيف طبقة Evidence Verification قبل إعطاء النتائج.

## الجديد
- Evidence / Guidelines tab
- Reference notes / guideline excerpts
- Evidence Verification section داخل كل تحليل
- يوضح هل التوصية:
  - verified_from_uploaded_material
  - framework_based_not_live_checked
  - not_live_verified
  - insufficient_evidence
  - needs_manual_reference_check
- لا يدّعي أنه بحث live في UpToDate/NICE/AAN/AHA/ESC/IDSA إذا لم يكن هناك Web API أو مرجع مرفوع.
- يفصل بين guideline-based و clinical reasoning.
- يحافظ على أسئلة قبل وبعد كل مرحلة من v4.5.

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

## مهم
هذه النسخة لا تقوم ببحث مباشر على الإنترنت.
للحصول على تدقيق أقوى، ضع مقتطفات guideline في صفحة Evidence / Guidelines أو ارفع PDF مرجعي في صفحة Results / Imaging.

## Streamlit Secrets

```toml
OPENAI_API_KEY = "sk-..."
```

## Main file
app.py
