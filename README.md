# MedAssist Neuro-General AutoEvidence Guard v4.7.4

نسخة قوية جدًا بعد اختبار v4.7.3.

## لماذا أضفنا هذه النسخة؟
النسخ السابقة كانت تمنع overtesting، لذلك في حالة الدوخة مع الخفقان لم تكن تعرض كل الصور والفحوصات الممكنة.
الآن أضفنا خريطة كاملة: ليس فقط ماذا تطلب الآن، بل ماذا يمكن أن تطلب لاحقًا ومتى يصبح مبررًا.

## الجديد
1. Comprehensive Diagnostic Map:
   - must_do_now
   - same_day_if_available
   - conditional_next_step
   - specialist_level
   - not_indicated_now

2. تعرض كل الفحوصات والصور الممكنة مع شروطها، مثل:
   - ECG
   - Orthostatic BP/HR
   - Capillary glucose
   - BMP/electrolytes/renal/Mg/Ca
   - CBC
   - TSH
   - Troponin/ACS pathway
   - D-dimer/PE workup
   - Holter/event monitor
   - Echocardiography
   - CT brain
   - MRI brain ± MRA/MRV
   - CTA/MRA head-neck
   - EEG

3. فحص سريري أقوى:
   - لا يقول فقط visually
   - يطلب palpate radial pulse
   - auscultate heart
   - orthostatic BP/HR
   - focused neuro screen
   - signs of dehydration/anemia/respiratory distress

4. Evidence أقوى:
   - لا يقبل رابط عام كمصدر قوي
   - إذا المصدر عام أو غير محدد يضع needs_manual_reference_check

## مهم
هذه خريطة فحوصات وليست أمرًا بطلب كل شيء. القرار النهائي للطبيب.

## Streamlit Secrets
```toml
OPENAI_API_KEY = "sk-..."
```

## Main file
app.py
