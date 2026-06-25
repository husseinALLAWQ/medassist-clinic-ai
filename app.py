import base64
import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

import streamlit as st
from openai import OpenAI
from pydantic import BaseModel, Field


# =========================================================
# MedAssist Clinic AI Ultra v3
# Multi-stage clinical workflow:
# 1) Intake + meds + symptoms
# 2) AI questions
# 3) Clinical exam advice
# 4) Exam findings entered by doctor
# 5) Preliminary diagnostic analysis + workup
# 6) Labs/imaging upload
# 7) Results analysis + medication support + next step
# =========================================================

st.set_page_config(
    page_title="MedAssist Clinic AI Ultra",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
UPLOAD_DIR = APP_DIR / "uploads"
DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

DEFAULT_MODEL = "gpt-4o-mini"


# =========================================================
# UI style
# =========================================================

st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 2rem;}
.stage-card {
    border: 1px solid rgba(120,120,120,.25);
    border-radius: 16px;
    padding: 16px;
    background: rgba(120,120,120,.06);
    margin-bottom: 12px;
}
.big-title {font-size: 1.25rem; font-weight: 850; margin-bottom: 6px;}
.small-note {font-size: .88rem; color: #777;}
.red-box {
    border-left: 5px solid #ff4b4b;
    padding: 11px 14px;
    border-radius: 9px;
    background: rgba(255,75,75,.11);
    margin-bottom: 8px;
}
.blue-box {
    border-left: 5px solid #2e86de;
    padding: 11px 14px;
    border-radius: 9px;
    background: rgba(46,134,222,.10);
    margin-bottom: 8px;
}
.purple-box {
    border-left: 5px solid #7d5fff;
    padding: 11px 14px;
    border-radius: 9px;
    background: rgba(125,95,255,.10);
    margin-bottom: 8px;
}
.green-box {
    border-left: 5px solid #00b894;
    padding: 11px 14px;
    border-radius: 9px;
    background: rgba(0,184,148,.10);
    margin-bottom: 8px;
}
.orange-box {
    border-left: 5px solid #fd9644;
    padding: 11px 14px;
    border-radius: 9px;
    background: rgba(253,150,68,.10);
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# Schemas
# =========================================================

class PatientQuestion(BaseModel):
    question: str
    why_ask: str
    expected_impact_on_diagnosis: str
    specialty: str
    priority: Literal["routine", "important", "urgent"]


class MedicationSafety(BaseModel):
    medication_or_issue: str
    concern: str
    why_it_matters: str
    what_to_check_before_treatment: List[str] = Field(default_factory=list)
    safe_clinical_note: str
    severity: Literal["low", "moderate", "high"]


class RedFlag(BaseModel):
    red_flag: str
    why_it_matters: str
    immediate_action: str
    urgency: Literal["emergency", "same_day", "soon", "routine"]


class ExamAdvice(BaseModel):
    exam_item: str
    how_to_check: str
    abnormal_findings_to_look_for: List[str] = Field(default_factory=list)
    why_it_matters: str
    related_specialty: str
    priority: Literal["routine", "important", "urgent"]


class DifferentialDx(BaseModel):
    diagnosis_or_category: str
    specialty: str
    probability_level: Literal["high", "medium", "low", "cannot_rank"]
    urgency: Literal["emergency", "same_day", "soon", "routine"]
    supporting_data: List[str] = Field(default_factory=list)
    against_data: List[str] = Field(default_factory=list)
    missing_data_needed: List[str] = Field(default_factory=list)
    how_to_confirm_or_exclude: str


class WorkupRecommendation(BaseModel):
    test_or_imaging: str
    type: Literal["blood_test", "urine_test", "stool_test", "imaging", "ECG", "bedside_test", "procedure", "referral", "other"]
    priority: Literal["urgent", "important", "routine"]
    why_needed: str
    what_result_would_change: str


class ImagingRequest(BaseModel):
    imaging_type: str
    body_region: str
    indication: str
    urgency: Literal["urgent", "same_day", "soon", "routine"]
    contrast_or_special_notes: str


class ResultFinding(BaseModel):
    finding: str
    source: str
    value_or_description: str
    interpretation: str
    effect_on_differential: str
    confidence: Literal["high", "medium", "low"]


class DrugSupport(BaseModel):
    treatment_goal: str
    possible_medication_class_or_option: str
    when_it_may_be_considered: str
    contraindications_or_cautions: List[str] = Field(default_factory=list)
    required_checks_before_use: List[str] = Field(default_factory=list)
    monitoring_or_followup: str
    note: str


class SpecialtyRoute(BaseModel):
    specialty: str
    why_activated: str
    urgency: Literal["emergency", "same_day", "soon", "routine"]


class SOAP(BaseModel):
    subjective: str
    objective: str
    assessment: str
    plan: str


class StageAnalysis(BaseModel):
    stage: str
    one_line_summary: str
    problem_representation: str
    triage_level: Literal["emergency", "same_day", "soon", "routine", "unclear"]
    triage_reason: str

    patient_questions: List[PatientQuestion] = Field(default_factory=list)
    medication_safety: List[MedicationSafety] = Field(default_factory=list)
    red_flags: List[RedFlag] = Field(default_factory=list)
    exam_advice: List[ExamAdvice] = Field(default_factory=list)
    preliminary_differential: List[DifferentialDx] = Field(default_factory=list)
    suggested_workup: List[WorkupRecommendation] = Field(default_factory=list)
    suggested_imaging: List[ImagingRequest] = Field(default_factory=list)
    interpreted_results: List[ResultFinding] = Field(default_factory=list)
    updated_differential: List[DifferentialDx] = Field(default_factory=list)
    drug_support_after_results: List[DrugSupport] = Field(default_factory=list)
    specialty_routes: List[SpecialtyRoute] = Field(default_factory=list)

    what_to_do_now: str
    what_to_enter_next: str
    doctor_caution: str
    patient_friendly_explanation_arabic: str
    soap_note: SOAP
    limitations: List[str] = Field(default_factory=list)


# =========================================================
# Helpers
# =========================================================

def get_api_key():
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    return None


def system_prompt():
    return """
You are MedAssist Clinic AI Ultra, a clinical decision support assistant for a licensed clinician.

The app is multi-stage:
1. Intake: understand symptoms, medications, allergies, history.
2. Questions: generate focused patient questions.
3. Exam guidance: advise what physical exam to perform.
4. Exam interpretation: analyze entered exam findings.
5. Preliminary diagnosis/workup: propose differential and tests/imaging.
6. Results review: interpret labs, MRI, X-ray, CT, ECG, reports, and images.
7. Medication support: provide safe medication considerations after results.

Strict clinical safety rules:
- You are not the final decision maker.
- Do not provide a final diagnosis.
- Do not provide dangerous medication dosing.
- Medication section must be "treatment considerations/safety support", not an order.
- Always mention contraindications/cautions and what to check before medication.
- Always check medications, allergies, anticoagulants/antiplatelets, renal function, pregnancy possibility, age, and red flags.
- If a case may be urgent, say emergency/same-day and explain.
- If data is insufficient, ask for missing data.
- If uploaded images/reports are unclear, say so.
- Separate: questions, exam advice, preliminary differential, workup, imaging, result interpretation, medication support.
- Use Arabic with accepted medical English terms.
- Return exactly according to the schema.
"""


def stage_instruction(stage, specialty):
    instructions = {
        "1_questions": """
STAGE 1: Generate patient questions after intake.
Focus:
- questions to ask the patient
- red flags
- medication safety from current meds/allergies
- what information is missing
Do not jump to final diagnosis.
""",
        "2_exam_advice": """
STAGE 2: Clinical exam advice.
Focus:
- what physical exam should the doctor perform next
- how to check it briefly
- abnormal findings to look for
- why each exam item matters
- red flags that would change urgency
""",
        "3_after_exam_analysis": """
STAGE 3: Analysis after doctor enters exam findings.
Focus:
- interpret history + symptoms + exam findings
- preliminary differential
- decide which labs/imaging are needed
- list exact tests/imaging and why
- medication safety cautions but no dosing
""",
        "4_results_analysis": """
STAGE 4: Analysis after labs/imaging/reports are uploaded.
Focus:
- interpret labs/MRI/Xray/CT/ECG/reports/images
- update differential
- say what became more likely/less likely
- decide if more imaging or tests are needed
- give medication support/considerations with contraindications and checks
- follow-up plan
""",
        "5_full_review": """
STAGE 5: Full clinical review.
Use all entered data and uploaded files to produce the most complete clinical decision-support report:
- questions still missing
- exam advice if missing
- red flags
- differential
- workup
- imaging
- results interpretation
- medication support
- referral urgency
- SOAP note
"""
    }
    return instructions.get(stage, instructions["5_full_review"]) + f"\nTarget specialty focus: {specialty}\n"


def make_context(data):
    return f"""
PATIENT BASICS
Patient ID: {data.get("patient_id")}
Age: {data.get("age")}
Sex: {data.get("sex")}
Pregnancy status: {data.get("pregnancy")}
Setting: {data.get("setting")}
Visit type: {data.get("visit_type")}

INTAKE / HISTORY
Chief complaint: {data.get("chief")}
Main symptoms: {data.get("symptoms")}
HPI: {data.get("hpi")}
Pain description / sensation / إحساس المريض: {data.get("sensation")}
Associated symptoms / ROS: {data.get("ros")}
Past medical/surgical history: {data.get("pmh")}
Family/social history: {data.get("family_social")}

MEDICATION SAFETY INPUT
Current medications: {data.get("meds")}
Allergies: {data.get("allergies")}
Pregnancy/renal/liver/bleeding risks if known: {data.get("safety")}

VITALS
{data.get("vitals")}

EXAM PLAN / EXAM FINDINGS
Doctor performed exam findings:
{data.get("exam_findings")}

MANUAL RESULTS
Labs written manually:
{data.get("labs")}
Imaging/report text:
{data.get("imaging_text")}
ECG/Echo/Other report text:
{data.get("other_results")}

DOCTOR QUESTION
{data.get("doctor_question")}
"""


def file_to_item(uploaded_file):
    data = uploaded_file.getvalue()
    b64 = base64.b64encode(data).decode("utf-8")
    name = uploaded_file.name
    mime = uploaded_file.type or "application/octet-stream"
    ext = name.lower().split(".")[-1] if "." in name else ""
    if ext in ["png", "jpg", "jpeg", "webp"]:
        image_ext = "jpeg" if ext == "jpg" else ext
        return {"type": "input_image", "image_url": f"data:image/{image_ext};base64,{b64}"}
    return {
        "type": "input_file",
        "filename": name,
        "file_data": f"data:{mime};base64,{b64}"
    }


def run_ai(stage, specialty, context, uploaded_files, model):
    key = get_api_key()
    if not key:
        st.error("OPENAI_API_KEY غير موجود. ضعه في Streamlit Secrets ثم Reboot app.")
        st.stop()

    client = OpenAI(api_key=key)
    content = [{"type": "input_text", "text": stage_instruction(stage, specialty) + "\n\nCLINICAL CONTEXT:\n" + context}]
    for f in uploaded_files or []:
        content.append(file_to_item(f))

    response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": content},
        ],
        text_format=StageAnalysis,
    )
    return response.output_parsed


def save_report(context, analysis):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path_json = DATA_DIR / f"case_{stamp}.json"
    path_md = DATA_DIR / f"case_{stamp}.md"
    path_json.write_text(json.dumps({
        "created": stamp,
        "context": context,
        "analysis": json.loads(analysis.model_dump_json())
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    path_md.write_text(report_markdown(analysis), encoding="utf-8")
    return path_json, path_md


def report_markdown(a: StageAnalysis):
    lines = []
    lines.append("# MedAssist Clinic AI Ultra Report\n")
    lines.append(f"## Stage\n{a.stage}\n")
    lines.append(f"## Summary\n{a.one_line_summary}\n")
    lines.append(f"## Problem representation\n{a.problem_representation}\n")
    lines.append(f"## Triage\n**{a.triage_level}** — {a.triage_reason}\n")

    def add_section(title, items, formatter):
        lines.append(f"\n## {title}")
        if not items:
            lines.append("—")
        for item in items:
            lines.append(formatter(item))

    add_section("Patient Questions", a.patient_questions, lambda q: f"- **{q.priority} / {q.specialty}:** {q.question} — {q.why_ask}. Impact: {q.expected_impact_on_diagnosis}")
    add_section("Medication Safety", a.medication_safety, lambda m: f"- **{m.severity}: {m.medication_or_issue}** — {m.concern}. {m.why_it_matters}. Check: {', '.join(m.what_to_check_before_treatment)}. Note: {m.safe_clinical_note}")
    add_section("Red Flags", a.red_flags, lambda r: f"- **{r.urgency}: {r.red_flag}** — {r.why_it_matters}. Action: {r.immediate_action}")
    add_section("Exam Advice", a.exam_advice, lambda e: f"- **{e.priority} / {e.related_specialty}: {e.exam_item}** — How: {e.how_to_check}. Look for: {', '.join(e.abnormal_findings_to_look_for)}. Why: {e.why_it_matters}")
    add_section("Preliminary Differential", a.preliminary_differential, lambda d: f"- **{d.probability_level} / {d.urgency}: {d.diagnosis_or_category} ({d.specialty})** — Support: {', '.join(d.supporting_data)}. Against: {', '.join(d.against_data)}. Missing: {', '.join(d.missing_data_needed)}. Confirm/exclude: {d.how_to_confirm_or_exclude}")
    add_section("Suggested Workup", a.suggested_workup, lambda w: f"- **{w.priority} / {w.type}: {w.test_or_imaging}** — {w.why_needed}. Changes: {w.what_result_would_change}")
    add_section("Suggested Imaging", a.suggested_imaging, lambda im: f"- **{im.urgency}: {im.imaging_type} {im.body_region}** — Indication: {im.indication}. Notes: {im.contrast_or_special_notes}")
    add_section("Interpreted Results", a.interpreted_results, lambda f: f"- **{f.finding}** from {f.source}: {f.value_or_description}. Interpretation: {f.interpretation}. Effect: {f.effect_on_differential}. Confidence: {f.confidence}")
    add_section("Updated Differential", a.updated_differential, lambda d: f"- **{d.probability_level} / {d.urgency}: {d.diagnosis_or_category} ({d.specialty})** — Confirm/exclude: {d.how_to_confirm_or_exclude}")
    add_section("Drug Support After Results", a.drug_support_after_results, lambda ds: f"- **Goal: {ds.treatment_goal}** — Option/class: {ds.possible_medication_class_or_option}. Consider when: {ds.when_it_may_be_considered}. Cautions: {', '.join(ds.contraindications_or_cautions)}. Checks: {', '.join(ds.required_checks_before_use)}. Monitoring: {ds.monitoring_or_followup}. Note: {ds.note}")
    add_section("Specialty Routes", a.specialty_routes, lambda s: f"- **{s.urgency}: {s.specialty}** — {s.why_activated}")

    lines.append(f"\n## What to do now\n{a.what_to_do_now}")
    lines.append(f"\n## What to enter next\n{a.what_to_enter_next}")
    lines.append(f"\n## Doctor caution\n{a.doctor_caution}")
    lines.append(f"\n## Patient-friendly Arabic Explanation\n{a.patient_friendly_explanation_arabic}")
    lines.append("\n## SOAP")
    lines.append(f"**S:** {a.soap_note.subjective}")
    lines.append(f"**O:** {a.soap_note.objective}")
    lines.append(f"**A:** {a.soap_note.assessment}")
    lines.append(f"**P:** {a.soap_note.plan}")
    lines.append("\n## Limitations")
    for l in a.limitations:
        lines.append(f"- {l}")
    return "\n".join(lines)


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:
    st.title("🩺 MedAssist Ultra v3")
    st.caption("Multi-stage clinical workflow")

    model = st.text_input("OpenAI model", value=DEFAULT_MODEL)
    st.caption("للتجربة والتكلفة الأقل: gpt-4o-mini")

    specialty = st.selectbox(
        "Specialty focus",
        [
            "General Medicine",
            "Emergency Medicine",
            "Neurology",
            "Cardiology",
            "Rheumatology",
            "Endocrinology",
            "Pulmonology",
            "Gastroenterology",
            "Infectious Disease",
            "Hematology",
            "Nephrology",
            "ENT",
            "Psychiatry",
            "Pediatrics",
            "Dermatology",
            "Orthopedics",
            "Urology",
            "Gynecology"
        ]
    )

    st.divider()
    st.warning("Clinical decision support only. القرار النهائي للطبيب.")
    st.caption("لا تدخل اسم المريض إذا ليس ضروريًا. استعمل Patient ID.")


# =========================================================
# Main layout
# =========================================================

st.title("MedAssist Clinic AI Ultra v3")
st.caption("نسخة متعددة المراحل: أسئلة → فحص سريري → تحليل → فحوصات/صور → تحليل نتائج → دعم دوائي آمن")

progress_cols = st.columns(6)
progress_cols[0].metric("1", "Intake")
progress_cols[1].metric("2", "Questions")
progress_cols[2].metric("3", "Exam")
progress_cols[3].metric("4", "Pre-Dx")
progress_cols[4].metric("5", "Results")
progress_cols[5].metric("6", "Medication")

tabs = st.tabs([
    "① Intake",
    "② AI Questions",
    "③ Clinical Exam",
    "④ Preliminary Dx & Workup",
    "⑤ Labs / MRI / X-ray / Results",
    "⑥ Final Support & Medication",
    "⑦ Report / History"
])


# =========================================================
# Tab 1: Intake
# =========================================================

with tabs[0]:
    st.header("① Intake: القصة، الأدوية، الأعراض، الإحساس")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        patient_id = st.text_input("Patient ID", value="Patient-001")
    with c2:
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
    with c3:
        sex = st.selectbox("Sex", ["Male", "Female", "Other/Unknown"])
    with c4:
        pregnancy = st.selectbox("Pregnancy status", ["Not relevant/Unknown", "Not pregnant", "Pregnant", "Possible"])

    c5, c6 = st.columns(2)
    with c5:
        setting = st.selectbox("Setting", ["Clinic", "ER", "Ward", "Telemedicine", "Other"])
    with c6:
        visit_type = st.selectbox("Visit type", ["First visit", "Follow-up", "Emergency/urgent", "Second opinion"])

    chief = st.text_area("Chief complaint / الشكوى الأساسية", height=90)
    symptoms = st.text_area("Main symptoms / الأعراض الأساسية", height=130)
    hpi = st.text_area("HPI / تفاصيل القصة المرضية", height=160)
    sensation = st.text_area("Sensation / كيف يصف المريض الإحساس؟", height=90, placeholder="حرق، تنميل، ضغط، طعن، دوخة، خفقان، خوف، ألم يتحسن بالحركة...")
    ros = st.text_area("Associated symptoms / ROS / أعراض مرافقة", height=110)

    c7, c8 = st.columns(2)
    with c7:
        pmh = st.text_area("Past medical/surgical history", height=120)
        family_social = st.text_area("Family/social history", height=90)
    with c8:
        meds = st.text_area("Current medications / الأدوية الحالية", height=120)
        allergies = st.text_area("Allergies / الحساسية", height=90)

    safety = st.text_area("Safety notes: renal/liver/bleeding/pregnancy risks", height=80)
    vitals = st.text_area("Vitals / العلامات الحيوية إن وجدت", height=90)
    doctor_question = st.text_area("Doctor question / ماذا تريد من النظام؟", height=80)


# Build context function after all fields
def current_context():
    data = {
        "patient_id": patient_id,
        "age": age,
        "sex": sex,
        "pregnancy": pregnancy,
        "setting": setting,
        "visit_type": visit_type,
        "chief": chief,
        "symptoms": symptoms,
        "hpi": hpi,
        "sensation": sensation,
        "ros": ros,
        "pmh": pmh,
        "family_social": family_social,
        "meds": meds,
        "allergies": allergies,
        "safety": safety,
        "vitals": vitals,
        "exam_findings": st.session_state.get("exam_findings", ""),
        "labs": st.session_state.get("manual_labs", ""),
        "imaging_text": st.session_state.get("imaging_text", ""),
        "other_results": st.session_state.get("other_results", ""),
        "doctor_question": doctor_question,
    }
    return make_context(data)


# =========================================================
# Render output function
# =========================================================

def render_analysis(a: StageAnalysis):
    st.subheader("Dashboard")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Triage", a.triage_level)
    d2.metric("Questions", len(a.patient_questions))
    d3.metric("Red flags", len(a.red_flags))
    d4.metric("Differentials", len(a.preliminary_differential) + len(a.updated_differential))

    st.markdown(f"""
    <div class="stage-card">
    <div class="big-title">🧾 Summary</div>
    <b>{a.one_line_summary}</b><br><br>
    {a.problem_representation}
    </div>
    """, unsafe_allow_html=True)

    if a.triage_level in ["emergency", "same_day"]:
        st.error(f"{a.triage_level}: {a.triage_reason}")
    elif a.triage_level == "soon":
        st.warning(f"{a.triage_level}: {a.triage_reason}")
    else:
        st.info(f"{a.triage_level}: {a.triage_reason}")

    if a.patient_questions:
        st.subheader("أسئلة تسألها للمريض")
        for q in a.patient_questions:
            with st.expander(f"{q.priority.upper()} — {q.question}"):
                st.write("**لماذا؟**", q.why_ask)
                st.write("**تأثيرها على التشخيص:**", q.expected_impact_on_diagnosis)
                st.write("**اختصاص:**", q.specialty)

    if a.medication_safety:
        st.subheader("نصائح أمان دوائي في هذه المرحلة")
        for m in a.medication_safety:
            st.markdown(f"""
            <div class="orange-box">
            <b>{m.severity}: {m.medication_or_issue}</b><br>
            <b>Concern:</b> {m.concern}<br>
            <b>Why:</b> {m.why_it_matters}<br>
            <b>Check before treatment:</b> {", ".join(m.what_to_check_before_treatment) if m.what_to_check_before_treatment else "—"}<br>
            <b>Note:</b> {m.safe_clinical_note}
            </div>
            """, unsafe_allow_html=True)

    if a.red_flags:
        st.subheader("Red flags")
        for r in a.red_flags:
            st.markdown(f"""
            <div class="red-box">
            <b>{r.urgency}: {r.red_flag}</b><br>
            {r.why_it_matters}<br>
            <b>Action:</b> {r.immediate_action}
            </div>
            """, unsafe_allow_html=True)

    if a.exam_advice:
        st.subheader("نصائح للفحص السريري")
        for e in a.exam_advice:
            st.markdown(f"""
            <div class="blue-box">
            <b>{e.priority} / {e.related_specialty}: {e.exam_item}</b><br>
            <b>How:</b> {e.how_to_check}<br>
            <b>Look for:</b> {", ".join(e.abnormal_findings_to_look_for) if e.abnormal_findings_to_look_for else "—"}<br>
            <b>Why:</b> {e.why_it_matters}
            </div>
            """, unsafe_allow_html=True)

    if a.preliminary_differential:
        st.subheader("تشخيص تقريبي / Differential قبل النتائج النهائية")
        for d in a.preliminary_differential:
            st.markdown(f"""
            <div class="purple-box">
            <b>{d.probability_level} / {d.urgency}: {d.diagnosis_or_category}</b> — {d.specialty}<br>
            <b>Confirm/exclude:</b> {d.how_to_confirm_or_exclude}
            </div>
            """, unsafe_allow_html=True)
            with st.expander("Details"):
                st.write("Supporting:", d.supporting_data or ["—"])
                st.write("Against:", d.against_data or ["—"])
                st.write("Missing:", d.missing_data_needed or ["—"])

    if a.suggested_workup:
        st.subheader("الفحوصات المطلوبة حسب الحالة")
        for w in a.suggested_workup:
            st.markdown(f"""
            <div class="blue-box">
            <b>{w.priority} / {w.type}: {w.test_or_imaging}</b><br>
            {w.why_needed}<br>
            <b>What changes:</b> {w.what_result_would_change}
            </div>
            """, unsafe_allow_html=True)

    if a.suggested_imaging:
        st.subheader("الصور المطلوبة إذا كانت لازمة")
        for im in a.suggested_imaging:
            st.markdown(f"""
            <div class="blue-box">
            <b>{im.urgency}: {im.imaging_type} — {im.body_region}</b><br>
            <b>Indication:</b> {im.indication}<br>
            <b>Notes:</b> {im.contrast_or_special_notes}
            </div>
            """, unsafe_allow_html=True)

    if a.interpreted_results:
        st.subheader("تحليل الفحوصات والصور المرفوعة")
        for f in a.interpreted_results:
            with st.expander(f"{f.finding} — {f.confidence} confidence"):
                st.write("Source:", f.source)
                st.write("Value/description:", f.value_or_description)
                st.write("Interpretation:", f.interpretation)
                st.write("Effect on differential:", f.effect_on_differential)

    if a.updated_differential:
        st.subheader("Updated differential بعد النتائج")
        for d in a.updated_differential:
            st.markdown(f"""
            <div class="purple-box">
            <b>{d.probability_level} / {d.urgency}: {d.diagnosis_or_category}</b> — {d.specialty}<br>
            <b>Confirm/exclude:</b> {d.how_to_confirm_or_exclude}
            </div>
            """, unsafe_allow_html=True)

    if a.drug_support_after_results:
        st.subheader("دعم دوائي آمن بعد النتائج")
        for ds in a.drug_support_after_results:
            st.markdown(f"""
            <div class="orange-box">
            <b>Goal:</b> {ds.treatment_goal}<br>
            <b>Possible class/option:</b> {ds.possible_medication_class_or_option}<br>
            <b>Consider when:</b> {ds.when_it_may_be_considered}<br>
            <b>Cautions:</b> {", ".join(ds.contraindications_or_cautions) if ds.contraindications_or_cautions else "—"}<br>
            <b>Checks before use:</b> {", ".join(ds.required_checks_before_use) if ds.required_checks_before_use else "—"}<br>
            <b>Monitoring:</b> {ds.monitoring_or_followup}<br>
            <b>Note:</b> {ds.note}
            </div>
            """, unsafe_allow_html=True)

    if a.specialty_routes:
        st.subheader("اختصاصات أو إحالات")
        for s in a.specialty_routes:
            st.info(f"**{s.urgency}: {s.specialty}** — {s.why_activated}")

    st.subheader("ماذا أفعل الآن؟")
    st.write(a.what_to_do_now)

    st.subheader("ماذا أدخل في المرحلة التالية؟")
    st.write(a.what_to_enter_next)

    st.subheader("تحذير للطبيب")
    st.warning(a.doctor_caution)

    st.subheader("SOAP")
    st.write("**S:**", a.soap_note.subjective)
    st.write("**O:**", a.soap_note.objective)
    st.write("**A:**", a.soap_note.assessment)
    st.write("**P:**", a.soap_note.plan)

    st.subheader("شرح عربي مبسط للمريض")
    st.write(a.patient_friendly_explanation_arabic)

    if a.limitations:
        st.subheader("Limitations")
        for l in a.limitations:
            st.caption(f"- {l}")


def analyze_button(label, stage, files=None):
    if st.button(label, type="primary", use_container_width=True):
        with st.spinner("AI يحلل هذه المرحلة..."):
            try:
                ctx = current_context()
                a = run_ai(stage, specialty, ctx, files or [], model)
                st.session_state[f"analysis_{stage}"] = a
                st.session_state["last_analysis"] = a
                st.session_state["last_context"] = ctx
                save_report(ctx, a)
                st.success("تم التحليل")
            except Exception as e:
                st.error("حدث خطأ أثناء التحليل")
                st.exception(e)


# =========================================================
# Tab 2: Questions
# =========================================================

with tabs[1]:
    st.header("② AI Questions: أسئلة للمريض + أمان دوائي أولي")
    st.markdown("""
    <div class="stage-card">
    هذه المرحلة بعد إدخال القصة والأدوية والأعراض. الهدف: يعطيك أسئلة دقيقة تسألها للمريض، مع Red flags وتنبيه دوائي أولي.
    </div>
    """, unsafe_allow_html=True)

    analyze_button("Generate patient questions & medication safety", "1_questions")

    if "analysis_1_questions" in st.session_state:
        render_analysis(st.session_state["analysis_1_questions"])


# =========================================================
# Tab 3: Clinical Exam
# =========================================================

with tabs[2]:
    st.header("③ Clinical Exam: نصائح للفحص السريري ثم إدخال ما وجدته")

    st.markdown("""
    <div class="stage-card">
    اضغط أولًا ليقترح عليك الفحص السريري المناسب حسب الحالة. بعدها اكتب ماذا وجدت في خانة Exam findings.
    </div>
    """, unsafe_allow_html=True)

    analyze_button("Suggest clinical exam checklist", "2_exam_advice")

    if "analysis_2_exam_advice" in st.session_state:
        render_analysis(st.session_state["analysis_2_exam_advice"])

    st.subheader("اكتب هنا ماذا وجدت بالفحص السريري")
    exam_findings = st.text_area(
        "Exam findings / نتائج الفحص السريري",
        value=st.session_state.get("exam_findings", ""),
        height=170,
        placeholder="مثال: BP..., HR..., neuro exam normal, power 5/5, reflexes..., joint swelling..., fundoscopy..."
    )
    st.session_state["exam_findings"] = exam_findings


# =========================================================
# Tab 4: Preliminary Dx & Workup
# =========================================================

with tabs[3]:
    st.header("④ Preliminary Dx & Workup: تحليل بعد الأسئلة والفحص")

    st.markdown("""
    <div class="stage-card">
    بعد إدخال أجوبة المريض ونتائج الفحص السريري، يعطيك تشخيصًا تقريبيًا/تفريقيًا، والفحوصات والصور المطلوبة حسب الحالة.
    </div>
    """, unsafe_allow_html=True)

    analyze_button("Analyze after exam → preliminary differential + tests/images", "3_after_exam_analysis")

    if "analysis_3_after_exam_analysis" in st.session_state:
        render_analysis(st.session_state["analysis_3_after_exam_analysis"])


# =========================================================
# Tab 5: Results
# =========================================================

with tabs[4]:
    st.header("⑤ Labs / MRI / X-ray / CT / ECG / Reports")

    st.markdown("""
    <div class="stage-card">
    هنا ترفع الفحوصات والصور أو تكتب النتائج يدويًا. يقبل PDF وصور تقارير وملفات Excel/CSV. 
    الأفضل رفع تقرير MRI/X-ray/CT المكتوب أو صورة واضحة للتقرير.
    </div>
    """, unsafe_allow_html=True)

    manual_labs = st.text_area(
        "Manual labs / اكتب النتائج يدويًا",
        value=st.session_state.get("manual_labs", ""),
        height=160,
        placeholder="CBC, ESR, CRP, LFT, RFT, TSH, troponin..."
    )
    st.session_state["manual_labs"] = manual_labs

    imaging_text = st.text_area(
        "MRI / X-ray / CT report text",
        value=st.session_state.get("imaging_text", ""),
        height=150,
        placeholder="انسخ تقرير الصورة هنا إذا متاح..."
    )
    st.session_state["imaging_text"] = imaging_text

    other_results = st.text_area(
        "ECG / Echo / Other reports",
        value=st.session_state.get("other_results", ""),
        height=120
    )
    st.session_state["other_results"] = other_results

    uploaded_files = st.file_uploader(
        "Upload PDF/images/reports/labs",
        type=["pdf", "png", "jpg", "jpeg", "webp", "txt", "csv", "xlsx", "docx"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.success(f"تم رفع {len(uploaded_files)} ملف")
        for f in uploaded_files:
            st.write(f"- {f.name} ({round(len(f.getvalue())/1024, 1)} KB)")


# =========================================================
# Tab 6: Final Support
# =========================================================

with tabs[5]:
    st.header("⑥ Final Support: تحليل بعد الفحوصات + دعم دوائي آمن")

    st.markdown("""
    <div class="stage-card">
    بعد رفع النتائج، هذه المرحلة تساعدك على تضييق التشخيص، طلب صور/تحاليل إضافية إذا لزم، وتعطي دعمًا دوائيًا آمنًا مع موانع وتحذيرات وفحوصات قبل الدواء. لا تعطي أمرًا نهائيًا بدل الطبيب.
    </div>
    """, unsafe_allow_html=True)

    analyze_button("Analyze results → updated differential + medication support", "4_results_analysis", uploaded_files if "uploaded_files" in locals() else [])

    if "analysis_4_results_analysis" in st.session_state:
        render_analysis(st.session_state["analysis_4_results_analysis"])

    st.divider()
    analyze_button("Full clinical review using all stages", "5_full_review", uploaded_files if "uploaded_files" in locals() else [])

    if "analysis_5_full_review" in st.session_state:
        render_analysis(st.session_state["analysis_5_full_review"])


# =========================================================
# Tab 7: Report/History
# =========================================================

with tabs[6]:
    st.header("⑦ Report / History")

    if "last_analysis" in st.session_state:
        md = report_markdown(st.session_state["last_analysis"])
        st.download_button(
            "Download report Markdown",
            data=md.encode("utf-8"),
            file_name=f"medassist_ultra_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )
        st.download_button(
            "Download structured JSON",
            data=st.session_state["last_analysis"].model_dump_json(indent=2).encode("utf-8"),
            file_name=f"medassist_ultra_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    st.subheader("Saved reports on app server")
    reports = sorted(DATA_DIR.glob("case_*.md"), reverse=True)
    if reports:
        for r in reports[:20]:
            with st.expander(r.name):
                st.text(r.read_text(encoding="utf-8")[:5000])
    else:
        st.info("لا يوجد تقارير محفوظة بعد.")

    with st.expander("Preview full context sent to AI"):
        st.text(current_context())
