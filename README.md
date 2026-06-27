# MedAssist Neuro-General AutoEvidence Guard v4.7.3

Patch بعد اختبار v4.7.2.

## ماذا تم إصلاحه؟
1. Guardrails برمجية بعد جواب AI:
   - إذا يوجد palpitations + near-syncope/presyncope:
     - يضيف Cardiology تلقائيًا إذا نسيه النموذج.
     - يضيف Endocrinology/Metabolic و Toxicology/Medication Safety عند اللزوم.
     - يزيل تكرار Neurology أو ECG checklist.
2. تصحيح triage:
   - near-syncope مع tachycardia وحالة مستقرة = same_day assessment.
   - Emergency فقط إذا: complete syncope, exertional syncope, chest pain, severe dyspnea, abnormal ECG, shock, persistent tachyarrhythmia, SpO2 drop, seizure, focal neurologic deficit, severe new headache.
3. TIA guardrail:
   - إذا لا توجد focal neurologic signs والأعراض posture-related مع palpitations:
     - TIA تصبح low.
     - لا تكون medium/high.
4. أسئلة قبل الفحص أصبحت أقوى وإجبارية في palpitations + near-syncope:
   - complete syncope
   - exertional/supine symptoms
   - chest pain/severe dyspnea
   - sudden onset/offset palpitations
   - family history sudden cardiac death
   - structural heart disease
   - stimulants/caffeine/decongestants/QT-risk drugs
   - dehydration/vomiting/diarrhea
   - bleeding/anemia symptoms
   - glucose/diabetes symptoms
5. Workup guardrails:
   - ECG
   - orthostatic BP/HR
   - capillary glucose
   - BMP/electrolytes/renal
   - CBC conditional
   - TSH conditional
   - Holter/event monitor conditional
6. Evidence guardrail:
   - إذا TIA evidence غير مناسب أو citation mismatch، يصبح needs_manual_reference_check.
7. إصلاح خطأ Streamlit frontend في Follow-up questions:
   - استبدلنا HTML unsafe داخل render_questions بعرض Streamlit بسيط ومستقر.

## مهم
- البحث في المصادر الطبية المفتوحة فقط.
- لا يفتح UpToDate أو مصادر مدفوعة إلا بترخيص/API.
- القرار النهائي للطبيب.

## Streamlit Secrets

```toml
OPENAI_API_KEY = "sk-..."
```

## Main file
app.py
