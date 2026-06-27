# MedAssist Neuro-General AutoEvidence Guard v4.7.5

Patch بعد اختبار v4.7.4.

## ما الجديد؟
1. Diagnostic Map مضمون أكثر:
   - إذا لم يرجع النموذج الخريطة، تظهر رسالة تحذير.
   - في palpitations + near-syncope، الكود يضيف core diagnostic map تلقائيًا إذا كانت فارغة.

2. Recommended Workup أقوى:
   - يضيف BMP/electrolytes/renal function including Mg/Ca إذا كانت الحالة palpitations + near-syncope.

3. أسئلة ما قبل الفحص:
   - إزالة التكرار الذكي للأسئلة المتشابهة مثل complete syncope/exertional syncope/chest pain/dehydration/glucose.

4. Differential cleanup:
   - Neurocardiogenic/Vasovagal syncope يصبح General Medicine / Cardiology بدل Neurology.
   - لا يرفع vasovagal فوق cardiac arrhythmia في حالة palpitations + near-syncope.

5. يحافظ على:
   - Cardiology guardrail.
   - TIA low/cannot_rank بدون focal signs.
   - No routine brain imaging without neuro red flags.
   - AutoEvidence web search.
   - Comprehensive Diagnostic Map:
     must_do_now / same_day_if_available / conditional_next_step / specialist_level / not_indicated_now.

## Main file
app.py
