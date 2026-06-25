import base64
import json
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

import streamlit as st
from openai import OpenAI
from pydantic import BaseModel, Field


# =========================
# Basic setup
# =========================

st.set_page_config(page_title="MedAssist Clinic AI", layout="wide")

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

MODEL_DEFAULT = "gpt-5.5"

DISCLAIMER = """
Clinical decision support only. This app does not provide a final diagnosis and does not replace physician judgment.
Use de-identified patient data whenever possible. Do not enter names, phone numbers, national IDs, or addresses unless your clinic policy permits it.
"""


# =========================
# Structured output schema
# =========================

class QuestionItem(BaseModel):
    question: str
    why_needed: str
    specialty: str
    urgency: Literal["routine", "important", "urgent"]


class WorkupItem(BaseModel):
    test_or_action: str
    why_needed: str
    priority: Literal["routine", "important", "urgent"]
    category: Literal["lab", "imaging", "bedside", "exam", "referral", "other"]


class RedFlagItem(BaseModel):
    flag: str
    reason: str
    recommended_action: str
    urgency: Literal["urgent", "same_day", "soon", "routine"]


class DifferentialItem(BaseModel):
    diagnosis_or_category: str
    specialty: str
    supporting_data: List[str] = Field(default_factory=list)
    against_data: List[str] = Field(default_factory=list)
    missing_data: List[str] = Field(default_factory=list)
    priority: Literal["high", "medium", "low"]
    urgency: Literal["emergency", "soon", "routine"]


class FindingItem(BaseModel):
    finding: str
    source: str
    interpretation: str
    clinical_relevance: str
    needs_confirmation: bool


class MedicationSafetyItem(BaseModel):
    issue: str
    reason: str
    action: str
    severity: Literal["low", "moderate", "high"]


class ClinicalAnalysis(BaseModel):
    mode: Literal["visit_1_questions_and_workup", "visit_2_results_and_diagnostic_support"]
    case_summary: str
    key_missing_information: List[str] = Field(default_factory=list)
    questions_for_patient: List[QuestionItem] = Field(default_factory=list)
    red_flags: List[RedFlagItem] = Field(default_factory=list)
    activated_specialties: List[str] = Field(default_factory=list)
    initial_or_next_workup: List[WorkupItem] = Field(default_factory=list)
    extracted_or_interpreted_findings: List[FindingItem] = Field(default_factory=list)
    differential_diagnosis: List[DifferentialItem] = Field(default_factory=list)
    medication_safety: List[MedicationSafetyItem] = Field(default_factory=list)
    referral_urgency: str
    explanation: str
    limitations: List[str] = Field(default_factory=list)


# =========================
# Helper functions
# =========================

def get_api_key() -> Optional[str]:
    # Works on Streamlit Cloud secrets or local environment variable
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    return None


def build_system_prompt() -> str:
    return """
You are MedAssist Clinic AI, a clinical decision support assistant for a licensed clinician.

Core rules:
- Do not claim to make a final diagnosis.
- Do not replace physician judgment.
- Analyze only the data provided.
- Give clinically useful missing questions and workup.
- Separate emergency red flags from routine possibilities.
- Be careful with medication safety, allergies, anticoagulants, renal function, pregnancy, pediatrics, and elderly patients.
- If the data is insufficient, say what is missing.
- Do not provide dangerous medication dosing.
- Prefer differential diagnosis categories and next diagnostic steps.
- For images/reports/labs: extract visible findings and explain clinical relevance, but do not pretend certainty if the image/report is unclear.
- Output in the required structured schema.
"""


def make_case_text(patient_id, age, sex, chief_complaint, history, symptoms, exam, vitals, meds, allergies, past_history, notes):
    return f"""
Patient ID: {patient_id}
Age: {age}
Sex: {sex}

Chief complaint:
{chief_complaint}

History of present illness:
{history}

Symptoms:
{symptoms}

Physical exam:
{exam}

Vitals:
{vitals}

Past medical history:
{past_history}

Medications:
{meds}

Allergies:
{allergies}

Additional notes:
{notes}
"""


def uploaded_file_to_content_item(uploaded_file):
    """
    Converts uploaded Streamlit files into OpenAI Responses content items.
    PDFs/spreadsheets/text-like files are sent as input_file.
    Images are sent as input_image.
    """
    file_bytes = uploaded_file.getvalue()
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    name = uploaded_file.name
    mime = uploaded_file.type or "application/octet-stream"
    suffix = name.lower().split(".")[-1] if "." in name else ""

    if suffix in ["png", "jpg", "jpeg", "webp"]:
        if suffix == "jpg":
            suffix = "jpeg"
        return {
            "type": "input_image",
            "image_url": f"data:image/{suffix};base64,{b64}"
        }

    # For PDFs and documents, use input_file.
    # OpenAI file inputs can process PDFs and document/spreadsheet text.
    return {
        "type": "input_file",
        "filename": name,
        "file_data": f"data:{mime};base64,{b64}"
    }


def run_ai(mode: str, case_text: str, uploaded_files: list, model: str) -> ClinicalAnalysis:
    api_key = get_api_key()
    if not api_key:
        st.error("OPENAI_API_KEY غير موجود. ضع المفتاح في Streamlit Secrets.")
        st.stop()

    client = OpenAI(api_key=api_key)

    if mode == "visit_1_questions_and_workup":
        task_text = """
TASK: Visit 1.
The clinician entered the first presentation before labs/imaging are available.
Return:
- missing patient questions
- red flags
- general medicine differential
- activated specialty modules
- initial labs/imaging/exam/referral workup
- medication safety alerts if relevant
Do not provide final diagnosis.
"""
    else:
        task_text = """
TASK: Visit 2.
The clinician is now adding lab results, PDFs, images, imaging reports, ECG reports, or other files.
Return:
- interpret uploaded findings
- update differential diagnosis
- explain what increased/decreased in probability qualitatively
- recommend next step
- list missing tests/questions
- flag urgent findings
Do not provide final diagnosis.
"""

    content = [
        {"type": "input_text", "text": task_text + "\n\nCASE DATA:\n" + case_text}
    ]

    for f in uploaded_files:
        content.append(uploaded_file_to_content_item(f))

    response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": content},
        ],
        text_format=ClinicalAnalysis,
    )
    return response.output_parsed


def show_analysis(analysis: ClinicalAnalysis):
    st.subheader("ملخص الحالة")
    st.write(analysis.case_summary)

    st.subheader("الاختصاصات المفعّلة")
    st.write(", ".join(analysis.activated_specialties) if analysis.activated_specialties else "—")

    st.subheader("أسئلة ناقصة للمريض")
    if analysis.questions_for_patient:
        for q in analysis.questions_for_patient:
            with st.expander(f"{q.urgency.upper()} — {q.question}"):
                st.write("**لماذا؟**", q.why_needed)
                st.write("**اختصاص:**", q.specialty)
    else:
        st.info("لا توجد أسئلة محددة في المخرجات.")

    st.subheader("Red Flags")
    if analysis.red_flags:
        for r in analysis.red_flags:
            st.error(f"{r.urgency}: {r.flag}")
            st.write(r.reason)
            st.write("**Action:**", r.recommended_action)
    else:
        st.success("لا توجد Red flags واضحة من البيانات المدخلة. هذا لا ينفي المرض الخطير.")

    st.subheader("الفحوصات / الخطوات المقترحة")
    for w in analysis.initial_or_next_workup:
        st.write(f"- **{w.priority} / {w.category}:** {w.test_or_action}")
        st.caption(w.why_needed)

    st.subheader("النتائج المستخرجة/المفسّرة")
    if analysis.extracted_or_interpreted_findings:
        for f in analysis.extracted_or_interpreted_findings:
            with st.expander(f.finding):
                st.write("**Source:**", f.source)
                st.write("**Interpretation:**", f.interpretation)
                st.write("**Clinical relevance:**", f.clinical_relevance)
                st.write("**Needs confirmation:**", f.needs_confirmation)
    else:
        st.info("لا توجد نتائج مرفوعة أو قابلة للتفسير في هذه المرحلة.")

    st.subheader("Differential diagnosis / احتمالات")
    for d in analysis.differential_diagnosis:
        with st.expander(f"{d.priority.upper()} — {d.diagnosis_or_category} ({d.specialty})"):
            st.write("**Supporting data:**")
            st.write(d.supporting_data or ["—"])
            st.write("**Against data:**")
            st.write(d.against_data or ["—"])
            st.write("**Missing data:**")
            st.write(d.missing_data or ["—"])
            st.write("**Urgency:**", d.urgency)

    st.subheader("Medication safety")
    if analysis.medication_safety:
        for m in analysis.medication_safety:
            st.warning(f"{m.severity}: {m.issue}")
            st.write(m.reason)
            st.write("**Action:**", m.action)
    else:
        st.success("لا توجد تنبيهات دوائية واضحة من البيانات المدخلة.")

    st.subheader("Referral urgency")
    st.write(analysis.referral_urgency)

    st.subheader("شرح مختصر")
    st.write(analysis.explanation)

    st.subheader("Limitations")
    for item in analysis.limitations:
        st.caption(f"- {item}")


def save_case(case_text: str, analysis: ClinicalAnalysis):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "timestamp": now,
        "case_text": case_text,
        "analysis": json.loads(analysis.model_dump_json())
    }
    path = DATA_DIR / f"case_{now}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# =========================
# UI
# =========================

st.title("MedAssist Clinic AI")
st.caption("مرحلتان: قبل الفحوصات → أسئلة وفحوصات. بعد الفحوصات/الصور → دعم تشخيصي وتضييق الاحتمالات.")
st.info(DISCLAIMER)

with st.sidebar:
    st.header("Settings")
    model = st.text_input("OpenAI model", value=MODEL_DEFAULT)
    st.caption("يمكن تغيير النموذج حسب حسابك وتوفّر النماذج لديك.")
    mode = st.radio(
        "Workflow",
        [
            "visit_1_questions_and_workup",
            "visit_2_results_and_diagnostic_support"
        ],
        format_func=lambda x: "1) أسئلة + فحوصات أولية" if x.startswith("visit_1") else "2) تحليل نتائج + دعم تشخيصي"
    )

st.header("بيانات الحالة")

col1, col2 = st.columns(2)
with col1:
    patient_id = st.text_input("Patient ID / كود المريض", value="Patient-001")
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    sex = st.selectbox("Sex", ["Male", "Female", "Other / Unknown"])
    chief_complaint = st.text_area("Chief complaint / الشكوى الأساسية", height=100)
    history = st.text_area("History of present illness / القصة المرضية", height=160)
    symptoms = st.text_area("Symptoms / الأعراض", height=140)

with col2:
    vitals = st.text_area("Vitals / العلامات الحيوية", height=90)
    exam = st.text_area("Physical exam / الفحص السريري", height=140)
    past_history = st.text_area("Past medical history / الأمراض السابقة", height=100)
    meds = st.text_area("Medications / الأدوية", height=100)
    allergies = st.text_area("Allergies / الحساسية", height=80)
    notes = st.text_area("Additional notes / ملاحظات إضافية", height=80)

st.header("رفع الفحوصات والصور والتقارير")
uploaded_files = st.file_uploader(
    "ارفع PDF / صورة تقرير / صورة تحليل / CSV / Excel / Word",
    type=["pdf", "png", "jpg", "jpeg", "webp", "txt", "csv", "xlsx", "docx"],
    accept_multiple_files=True
)

case_text = make_case_text(patient_id, age, sex, chief_complaint, history, symptoms, exam, vitals, meds, allergies, past_history, notes)

if st.button("Analyze with AI", type="primary"):
    with st.spinner("AI يحلل الحالة..."):
        try:
            analysis = run_ai(mode, case_text, uploaded_files or [], model)
            show_analysis(analysis)
            saved_path = save_case(case_text, analysis)
            st.success(f"تم حفظ الحالة محليًا داخل التطبيق: {saved_path.name}")
        except Exception as e:
            st.error("حدث خطأ أثناء التحليل.")
            st.exception(e)

with st.expander("عرض النص الكامل المرسل للنموذج"):
    st.text(case_text)
