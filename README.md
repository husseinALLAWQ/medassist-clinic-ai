# MedAssist Neuro-General AutoEvidence Guard v4.7.6

تحسين كبير بعد اختبار v4.7.5.

## أهم التحسينات
1. تنظيف التكرار:
   - يدمج Resting 12-lead ECG و 12-lead ECG.
   - يدمج Orthostatic BP measurements و Orthostatic BP/HR.
   - يدمج Ambulatory ECG Monitoring و Holter/event monitor.
   - يدمج BMP/electrolytes/renal/Mg/Ca.
   - يحذف الأسئلة المتشابهة بشكل أذكى.

2. Differential أقوى:
   - في palpitations + near-syncope يبقى Cardiac arrhythmia = high / same_day.
   - Orthostatic/postural presyncope = high إذا الأعراض worse standing/improve sitting.
   - Vasovagal/neurocardiogenic يبقى medium ولا يسبق arrhythmia.

3. Action Pathway جديد:
   - ماذا تفعل الآن.
   - ماذا إذا ECG abnormal.
   - ماذا إذا orthostatic positive.
   - ماذا إذا ECG/labs normal لكن النوبات تتكرر.
   - ماذا إذا ظهرت red flags neurologic/cardiac.

4. Diagnostic Map مرتبة:
   - must_do_now
   - same_day_if_available
   - conditional_next_step
   - specialist_level
   - not_indicated_now

5. Evidence guardrail:
   - يحذر إذا المصدر عام جدًا وليس guideline section واضح.
   - يطلب manual reference check إذا المصدر غير محدد بدقة.

6. يحافظ على:
   - AutoEvidence Web Search.
   - Cardiology forced in palpitations + near-syncope.
   - TIA low/cannot_rank without focal signs.
   - No routine neuroimaging unless neurologic red flags.
   - Clinical exam protocol مفصل.

## Main file
app.py
