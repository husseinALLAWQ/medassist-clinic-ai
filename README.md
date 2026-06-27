# MedAssist Neuro-General AutoEvidence Guard v4.7.2

هذه نسخة Patch بعد اختبار v4.7.1.

## ما الذي تم إصلاحه؟
1. Cardiology إجباري إذا يوجد palpitations + near-syncope/presyncope.
2. TIA لا يجب أن تكون medium/high إذا لا توجد علامات عصبية بؤرية.
3. تقوية أسئلة ما قبل الفحص في حالات الدوخة والخفقان:
   - complete syncope
   - exertional syncope
   - chest pain
   - severe dyspnea
   - sudden onset/offset palpitations
   - family history sudden cardiac death
   - structural heart disease
   - stimulant/caffeine/decongestant use
   - dehydration/vomiting/diarrhea
   - bleeding/anemia symptoms
   - glucose/diabetes symptoms
4. تقوية Workup:
   - ECG
   - orthostatic BP/HR
   - capillary glucose
   - BMP/electrolytes/renal function
   - CBC when possible anemia/bleeding/fatigue
   - TSH conditional
   - Holter/event monitor conditional
   - Troponin conditional
   - PE workup conditional
5. Evidence Verification أصبح يطلب:
   - source title
   - organization
   - year/date
   - URL/citation
   - exact evidence point
   - caution/limitation
6. ER threshold أقوى:
   - complete syncope
   - exertional syncope
   - chest pain
   - severe dyspnea
   - abnormal ECG
   - hypotension/shock
   - persistent tachyarrhythmia
   - focal neurologic deficit
   - severe new headache
   - seizure
   - SpO2 drop

## مهم
- البحث في المصادر الطبية المفتوحة.
- لا يفتح UpToDate أو مصادر مدفوعة إلا إذا توفر وصول مرخص/API.
- القرار النهائي للطبيب.

## Streamlit Secrets

```toml
OPENAI_API_KEY = "sk-..."
```

## Main file
app.py
