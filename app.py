import base64
import json
from datetime import datetime
from pathlib import Path
from typing import List, Literal

import streamlit as st
from openai import OpenAI
from pydantic import BaseModel, Field


# =========================================================
# MedAssist Neuro-General Guard v4.3
# Neurology-first + General Medicine coverage
# =========================================================

st.set_page_config(
    page_title="MedAssist Neuro-General Guard v4.3",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
DEFAULT_MODEL = "gpt-4o-mini"


# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 2rem;}
.workflow-card {
    border: 1px solid rgba(120,120,120,.25);
    border-radius: 16px;
    padding: 16px;
    background: rgba(120,120,120,.06);
    margin-bottom: 12px;
}
.title-big {font-size: 1.25rem; font-weight: 850; margin-bottom: 6px;}
.red-box {border-left:5px solid #ff4b4b;padding:11px 14px;border-radius:9px;background:rgba(255,75,75,.11);margin-bottom:8px;}
.blue-box {border-left:5px solid #2e86de;padding:11px 14px;border-radius:9px;background:rgba(46,134,222,.10);margin-bottom:8px;}
.purple-box {border-left:5px solid #7d5fff;padding:11px 14px;border-radius:9px;background:rgba(125,95,255,.10);margin-bottom:8px;}
.orange-box {border-left:5px solid #fd9644;padding:11px 14px;border-radius:9px;background:rgba(253,150,68,.10);margin-bottom:8px;}
.green-box {border-left:5px solid #00b894;padding:11px 14px;border-radius:9px;background:rgba(0,184,148,.10);margin-bottom:8px;}
.gray-box {border-left:5px solid #95a5a6;padding:11px 14px;border-radius:9px;background:rgba(149,165,166,.10);margin-bottom:8px;}
</style>
""", unsafe_allow_html=True)


# =========================================================
# Schemas
# =========================================================

class SafetyGate(BaseModel):
    emergency_now: Literal["yes", "no", "unclear"]
    same_day_assessment_needed: Literal["yes", "no", "unclear"]
    reason: str
    immediate_action: str
    must_not_miss_conditions: List[str] = Field(default_factory=list)


class ActivatedModule(BaseModel):
    module: Literal[
        "Neurology", "Emergency Medicine", "General Medicine", "Cardiology", "Pulmonology",
        "Endocrinology/Metabolic", "Infectious Disease", "Rheumatology", "Nephrology/Urology",
        "Gastroenterology/Hepatology", "Hematology", "ENT", "Psychiatry", "Orthopedics/MSK",
        "Dermatology", "Pediatrics", "Gynecology/Pregnancy", "Toxicology/Medication Safety"
    ]
    why_activated: str
    urgency: Literal["emergency", "same_day", "soon", "routine"]


class ReferenceAnchor(BaseModel):
    source_or_framework: str
    principle_used: str
    how_it_applies_here: str


class MissingCriticalData(BaseModel):
    item: str
    why_required: str
    blocks_safe_decision: bool


class ChecklistItem(BaseModel):
    checklist_name: str
    item: str
    status_from_case: Literal["present", "absent", "unknown", "not_applicable"]
    action_if_present_or_unknown: str


class MandatoryQuestion(BaseModel):
    question: str
    domain: Literal[
        "neurology", "general_medicine", "cardiology", "pulmonology", "infection",
        "endocrine_metabolic", "rheumatology", "renal_urology", "gastro_hepato",
        "hematology", "ENT", "psychiatry", "orthopedics_MSK", "gynecology_pregnancy",
        "medication_safety", "red_flags", "social_functional"
    ]
    why_ask: str
    how_answer_changes_decision: str
    priority: Literal["must_ask_now", "important", "routine"]


class RedFlag(BaseModel):
    flag: str
    domain: str
    status_from_data: Literal["present", "absent", "unknown"]
    why_dangerous: str
    action_threshold: str
    urgency: Literal["emergency", "same_day", "soon", "routine"]


class ExamChecklistItem(BaseModel):
    exam_item: str
    system: Literal[
        "neurologic", "cardiovascular", "respiratory", "abdominal", "ENT",
        "MSK", "skin", "rheumatologic", "psychiatric", "general_vitals", "other"
    ]
    how_to_do_it_briefly: str
    what_to_record: List[str] = Field(default_factory=list)
    abnormal_meaning: str
    normal_does_not_exclude: str
    priority: Literal["must_do_now", "important", "routine"]


class ExamInterpretationItem(BaseModel):
    entered_finding: str
    interpretation: str
    localization_or_system_if_relevant: str
    supports: List[str] = Field(default_factory=list)
    argues_against_but_does_not_exclude: List[str] = Field(default_factory=list)
    next_exam_or_test_triggered: str


class WorkupItem(BaseModel):
    test_or_action: str
    type: Literal[
        "lab", "imaging", "ECG", "EEG", "LP", "EMG_NCS", "bedside_test",
        "procedure", "referral", "monitoring", "other"
    ]
    priority: Literal["urgent", "important", "routine", "not_now"]
    why_needed: str
    what_result_changes: str
    avoid_if_not_indicated: str


class ImagingDecision(BaseModel):
    imaging_needed_now: Literal["yes", "no", "conditional", "unclear_need_more_data"]
    imaging_type: str
    body_region: str
    indication_or_reason: str
    why_not_needed_if_no: str
    urgency: Literal["emergency", "same_day", "soon", "routine", "not_indicated_now"]
    protocol_notes: str
    explicit_no_imaging_statement: str


class DifferentialItem(BaseModel):
    diagnosis_or_category: str
    specialty_domain: str
    probability: Literal["high", "medium", "low", "cannot_rank"]
    urgency: Literal["emergency", "same_day", "soon", "routine"]
    supporting_features: List[str] = Field(default_factory=list)
    features_against: List[str] = Field(default_factory=list)
    missing_data_needed: List[str] = Field(default_factory=list)
    confirm_or_exclude_step: str


class ResultInterpretation(BaseModel):
    source: str
    finding: str
    interpretation: str
    effect_on_differential: str
    confidence: Literal["high", "medium", "low"]


class MedicationSafety(BaseModel):
    medication_or_issue: str
    concern: str
    check_before_treatment: List[str] = Field(default_factory=list)
    avoid_or_caution: str
    safer_consideration: str
    severity: Literal["high", "moderate", "low"]


class TreatmentSupport(BaseModel):
    clinical_goal: str
    possible_class_or_option: str
    when_to_consider: str
    must_check_before: List[str] = Field(default_factory=list)
    avoid_if: List[str] = Field(default_factory=list)
    monitoring: str
    note_no_dosing: str


class FollowUpThreshold(BaseModel):
    situation: str
    action: str
    timeframe: Literal["immediate_ER", "same_day", "24_48h", "routine_followup", "self_monitor_with_return_precautions"]


class QualityControl(BaseModel):
    completeness_level: Literal["safe_enough_for_next_step", "partial_needs_more_data", "unsafe_missing_critical_data"]
    overtesting_check: str
    undertesting_check: str
    medication_safety_check: str
    general_medicine_mimics_check: str
    clinician_override_needed_when: str


class SOAP(BaseModel):
    subjective: str
    objective: str
    assessment: str
    plan: str


class GuardAnalysis(BaseModel):
    stage: str
    case_summary: str
    problem_representation: str
    safety_gate: SafetyGate
    triage_level: Literal["emergency", "same_day", "soon", "routine", "unclear"]
    triage_reason: str
    activated_modules: List[ActivatedModule] = Field(default_factory=list)
    missing_critical_data: List[MissingCriticalData] = Field(default_factory=list)
    guideline_checklist: List[ChecklistItem] = Field(default_factory=list)
    mandatory_questions: List[MandatoryQuestion] = Field(default_factory=list)
    red_flags: List[RedFlag] = Field(default_factory=list)
    exam_checklist: List[ExamChecklistItem] = Field(default_factory=list)
    exam_interpretation: List[ExamInterpretationItem] = Field(default_factory=list)
    recommended_workup: List[WorkupItem] = Field(default_factory=list)
    imaging_decision: List[ImagingDecision] = Field(default_factory=list)
    differential_diagnosis: List[DifferentialItem] = Field(default_factory=list)
    interpreted_results: List[ResultInterpretation] = Field(default_factory=list)
    medication_safety: List[MedicationSafety] = Field(default_factory=list)
    treatment_support_after_results: List[TreatmentSupport] = Field(default_factory=list)
    follow_up_thresholds: List[FollowUpThreshold] = Field(default_factory=list)
    trusted_reference_anchors: List[ReferenceAnchor] = Field(default_factory=list)
    quality_control: QualityControl
    strict_quality_check: str
    what_to_do_now: str
    what_to_enter_next: str
    referral_or_er_threshold: str
    patient_explanation_arabic: str
    soap_note: SOAP
    limitations: List[str] = Field(default_factory=list)


# =========================================================
# Prompt framework
# =========================================================

REFERENCE_FRAMEWORK = """
Clinical decision-support framework to apply:
A) Universal Safety Gate:
- unstable vitals, altered mental status, severe respiratory distress, chest pain concerning for ACS,
  stroke/TIA signs, sepsis/meningitis signs, anaphylaxis, severe dehydration/shock,
  severe trauma, overdose/toxicity, pregnancy emergency, suicidal/homicidal risk -> escalate.
B) Neurology modules:
- headache/raised ICP/CSF leak, seizure/syncope, stroke/TIA, vertigo/dizziness, neuropathy,
  myelopathy/radiculopathy, movement disorders, cognitive/psychiatric overlap.
C) General medicine mimics:
- hypoglycemia/electrolyte disturbance, thyroid disease, anemia, infection, autoimmune/inflammatory,
  cardiac arrhythmia/ACS, PE, medication adverse effect, intoxication/withdrawal, renal/hepatic disease.
D) Headache:
- Use SNNOOP10 red flags and migraine/tension phenotype, but do not make the whole system headache-only.
E) Imaging:
- Avoid routine imaging when not indicated. Do not miss imaging when emergency red flags exist.
F) Medication safety:
- Always consider allergy, pregnancy/postpartum, age, renal/liver disease, anticoagulants/antiplatelets,
  QT risk, sedatives/opioids/benzodiazepines, serotonin syndrome risk, NSAID bleeding/renal risk,
  antiepileptic adverse effects, medication overuse headache.
G) Treatment support:
- Provide classes/options and safety checks only. No doses. No final prescription.
"""


def get_api_key():
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return None


def system_prompt(strictness: str):
    return f"""
You are MedAssist Neuro-General Guard v4.3.

Identity:
- Neurology-first but NOT neurology-only.
- You must cover all neurological diseases AND general medicine mimics/causes.
- You act as clinical decision support for a clinician, not a final diagnostic authority.

Strictness level: {strictness}

Mandatory rules:
1. Start with Safety Gate in every stage.
2. Activate relevant specialty modules, not just neurology.
3. If the case is neurologic, deeply analyze neurological localization and differentials.
4. If symptoms may be non-neurologic, analyze general medicine causes and mimics.
5. Always ask missing critical questions before ranking too confidently.
6. Always provide exam checklist across relevant systems, not only neuro if another system is relevant.
7. Always interpret entered exam findings clearly.
8. Imaging must be justified. Do not over-order CT/MRI/X-ray.
9. Do not miss dangerous conditions: stroke/TIA, SAH, meningitis/encephalitis, raised ICP, seizure emergency,
   ACS, arrhythmia/syncope, PE, sepsis, hypoglycemia, DKA, severe electrolyte abnormalities, drug toxicity,
   pregnancy/postpartum emergencies, suicidal risk, trauma/bleeding.
10. Medication safety must appear whenever meds/allergies/risks are entered.
11. Treatment support must never include dosing; use medication classes/options, contraindications, checks, monitoring.
12. Use clean Arabic medical language with English medical terms when useful. Use "SOAP" not mistranslations.
13. Return exactly according to schema.

{REFERENCE_FRAMEWORK}
"""


def stage_instruction(stage, focus, body_system_focus):
    if stage == "questions":
        return f"""
STAGE 1 — Questions + Safety Gate.
Generate mandatory questions for neuro and general medicine, red flags, missing data, module routing, medication safety.
Neurology focus: {focus}
General body-system focus: {body_system_focus}
"""
    if stage == "exam":
        return f"""
STAGE 2 — Exam Advisor.
Generate clinical exam checklist across neurology and relevant general medicine systems, then interpret entered findings.
Neurology focus: {focus}
General body-system focus: {body_system_focus}
"""
    if stage == "preliminary":
        return f"""
STAGE 3 — Preliminary Differential + Workup.
Use history/exam. Provide differential across neurology and general medicine, strict imaging/workup decision, quality control.
Neurology focus: {focus}
General body-system focus: {body_system_focus}
"""
    if stage == "results":
        return f"""
STAGE 4 — Results Review + Medication Support.
Interpret labs/imaging/reports across neuro and general medicine. Update differential and treatment support without dosing.
Neurology focus: {focus}
General body-system focus: {body_system_focus}
"""
    return f"""
STAGE 5 — Full Neuro-General Guard Review.
Full safety-gated, multi-specialty clinical decision support report.
Neurology focus: {focus}
General body-system focus: {body_system_focus}
"""


def make_context(data):
    return f"""
PATIENT BASICS
Patient ID: {data.get("patient_id")}
Age: {data.get("age")}
Sex: {data.get("sex")}
Pregnancy/Postpartum: {data.get("pregnancy")}
Setting: {data.get("setting")}

CLINICAL SEARCH / DOCTOR FOCUS
Clinical search question: {data.get("clinical_search")}

CHIEF COMPLAINT AND HISTORY
Chief complaint: {data.get("complaint")}
Onset/timing: {data.get("onset")}
Course/progression: {data.get("course")}
Quality/sensation: {data.get("quality")}
Severity/function impact: {data.get("severity")}
Associated symptoms / ROS: {data.get("associated")}

NEURO SCREEN
Focal neuro symptoms: {data.get("focal")}
Headache/raised ICP/CSF leak features: {data.get("headache_icp_csf")}
Seizure/syncope features: {data.get("seizure_syncope")}
Dizziness/vertigo features: {data.get("vertigo")}
Numbness/weakness/back-neck symptoms: {data.get("weakness_neuropathy")}
Previous neuro episodes/history: {data.get("previous_neuro")}

GENERAL MEDICINE SCREEN
Cardiac symptoms/risk: {data.get("cardiac")}
Respiratory symptoms/risk: {data.get("resp")}
Infection/systemic symptoms: {data.get("infection")}
Endocrine/metabolic symptoms/risk: {data.get("endo")}
GI/renal/hepatic symptoms: {data.get("gi_renal")}
Rheum/MSK/skin symptoms: {data.get("rheum_msk_skin")}
Psychiatric/sleep/substance symptoms: {data.get("psych_sleep_substance")}
Trauma/cancer/immunosuppression: {data.get("risk_red")}

MEDICATION SAFETY
Current medications: {data.get("meds")}
Allergies: {data.get("allergies")}
Renal/liver/pregnancy/bleeding risks: {data.get("safety")}

VITALS
{data.get("vitals")}

EXAM FINDINGS ENTERED BY DOCTOR
General appearance/vitals: {data.get("general_exam")}
Neurologic exam: {data.get("neuro_exam")}
Cardiovascular/respiratory exam: {data.get("cardio_resp_exam")}
Abdominal/renal exam: {data.get("abd_renal_exam")}
ENT/MSK/skin/rheum exam: {data.get("ent_msk_skin_exam")}
Psychiatric/cognitive exam: {data.get("psych_exam")}

RESULTS
Labs: {data.get("labs")}
Imaging report text: {data.get("imaging")}
ECG/EEG/EMG/Echo/Other report: {data.get("other")}

ADDITIONAL DOCTOR QUESTION
{data.get("doctor_question")}
"""


def file_item(uploaded_file):
    raw = uploaded_file.getvalue()
    b64 = base64.b64encode(raw).decode("utf-8")
    name = uploaded_file.name
    mime = uploaded_file.type or "application/octet-stream"
    ext = name.lower().split(".")[-1] if "." in name else ""
    if ext in ["png", "jpg", "jpeg", "webp"]:
        ext2 = "jpeg" if ext == "jpg" else ext
        return {"type": "input_image", "image_url": f"data:image/{ext2};base64,{b64}"}
    if ext in ["txt", "csv"]:
        try:
            text = raw.decode("utf-8", errors="ignore")[:20000]
            return {"type": "input_text", "text": f"Uploaded file {name}:\n{text}"}
        except Exception:
            pass
    return {"type": "input_file", "filename": name, "file_data": f"data:{mime};base64,{b64}"}


def run_ai(stage, focus, body_system_focus, strictness, context, files, model):
    key = get_api_key()
    if not key:
        st.error("OPENAI_API_KEY غير موجود في Streamlit Secrets.")
        st.stop()

    client = OpenAI(api_key=key)
    content = [{"type": "input_text", "text": stage_instruction(stage, focus, body_system_focus) + "\n\nCASE CONTEXT:\n" + context}]
    for f in files or []:
        content.append(file_item(f))

    response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": system_prompt(strictness)},
            {"role": "user", "content": content},
        ],
        text_format=GuardAnalysis,
    )
    return response.output_parsed


# =========================================================
# Report/render
# =========================================================

def report_markdown(a):
    return "# MedAssist Neuro-General Guard v4.3 Report\n\n```json\n" + a.model_dump_json(indent=2) + "\n```"


def save_report(context, a):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md = report_markdown(a)
    (DATA_DIR / f"neuro_general_guard_{stamp}.md").write_text(md, encoding="utf-8")
    (DATA_DIR / f"neuro_general_guard_{stamp}.json").write_text(json.dumps({
        "created": stamp,
        "context": context,
        "analysis": a.model_dump()
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return md


def render(a):
    st.subheader("Clinical Dashboard")
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Triage", a.triage_level)
    d2.metric("Emergency", a.safety_gate.emergency_now)
    d3.metric("Modules", len(a.activated_modules))
    d4.metric("Red flags", len(a.red_flags))
    d5.metric("Diff Dx", len(a.differential_diagnosis))

    st.markdown(f"""
    <div class='workflow-card'>
    <div class='title-big'>🧠 Neuro-General Summary</div>
    <b>{a.case_summary}</b><br><br>
    <b>Problem representation:</b><br>{a.problem_representation}
    </div>
    """, unsafe_allow_html=True)

    if a.safety_gate.emergency_now == "yes":
        st.error(f"SAFETY GATE: EMERGENCY — {a.safety_gate.reason}")
    elif a.safety_gate.same_day_assessment_needed == "yes":
        st.warning(f"SAFETY GATE: SAME DAY — {a.safety_gate.reason}")
    else:
        st.success(f"SAFETY GATE: no immediate emergency detected — {a.safety_gate.reason}")

    st.markdown(f"""
    <div class='red-box'>
    <b>Must-not-miss:</b> {", ".join(a.safety_gate.must_not_miss_conditions) if a.safety_gate.must_not_miss_conditions else "—"}<br>
    <b>Immediate action:</b> {a.safety_gate.immediate_action}
    </div>
    """, unsafe_allow_html=True)

    if a.activated_modules:
        st.subheader("Activated Modules")
        for m in a.activated_modules:
            st.info(f"**{m.urgency}: {m.module}** — {m.why_activated}")

    if a.missing_critical_data:
        st.subheader("Missing Critical Data")
        for x in a.missing_critical_data:
            st.warning(f"{x.item}: {x.why_required} | blocks safe decision: {x.blocks_safe_decision}")

    if a.guideline_checklist:
        st.subheader("Clinical Checklists")
        for g in a.guideline_checklist:
            cls = "red-box" if g.status_from_case in ["present", "unknown"] else "green-box"
            st.markdown(f"""
            <div class='{cls}'>
            <b>{g.checklist_name}: {g.item}</b><br>
            <b>Status:</b> {g.status_from_case}<br>
            <b>Action:</b> {g.action_if_present_or_unknown}
            </div>
            """, unsafe_allow_html=True)

    if a.mandatory_questions:
        st.subheader("Mandatory Questions")
        for q in a.mandatory_questions:
            with st.expander(f"{q.priority} / {q.domain} — {q.question}"):
                st.write("**Why ask:**", q.why_ask)
                st.write("**Decision impact:**", q.how_answer_changes_decision)

    if a.red_flags:
        st.subheader("Red Flags")
        for r in a.red_flags:
            st.markdown(f"""
            <div class='red-box'>
            <b>{r.urgency} / {r.domain} / {r.status_from_data}: {r.flag}</b><br>
            {r.why_dangerous}<br>
            <b>Threshold/action:</b> {r.action_threshold}
            </div>
            """, unsafe_allow_html=True)

    if a.exam_checklist:
        st.subheader("Clinical Exam Advisor")
        for e in a.exam_checklist:
            st.markdown(f"""
            <div class='blue-box'>
            <b>{e.priority} / {e.system}: {e.exam_item}</b><br>
            <b>How:</b> {e.how_to_do_it_briefly}<br>
            <b>Record:</b> {", ".join(e.what_to_record) if e.what_to_record else "—"}<br>
            <b>Abnormal meaning:</b> {e.abnormal_meaning}<br>
            <b>Normal does not exclude:</b> {e.normal_does_not_exclude}
            </div>
            """, unsafe_allow_html=True)

    if a.exam_interpretation:
        st.subheader("Exam Interpretation")
        for e in a.exam_interpretation:
            st.markdown(f"""
            <div class='green-box'>
            <b>{e.entered_finding}</b><br>
            <b>Interpretation:</b> {e.interpretation}<br>
            <b>Localization/system:</b> {e.localization_or_system_if_relevant}<br>
            <b>Supports:</b> {", ".join(e.supports) if e.supports else "—"}<br>
            <b>Argues against but does not exclude:</b> {", ".join(e.argues_against_but_does_not_exclude) if e.argues_against_but_does_not_exclude else "—"}<br>
            <b>Next:</b> {e.next_exam_or_test_triggered}
            </div>
            """, unsafe_allow_html=True)

    if a.recommended_workup:
        st.subheader("Recommended Workup")
        for w in a.recommended_workup:
            st.markdown(f"""
            <div class='blue-box'>
            <b>{w.priority} / {w.type}: {w.test_or_action}</b><br>
            {w.why_needed}<br>
            <b>What changes:</b> {w.what_result_changes}<br>
            <b>Avoid if not indicated:</b> {w.avoid_if_not_indicated}
            </div>
            """, unsafe_allow_html=True)

    if a.imaging_decision:
        st.subheader("Imaging Decision")
        for im in a.imaging_decision:
            cls = "red-box" if im.imaging_needed_now == "yes" else ("orange-box" if im.imaging_needed_now in ["conditional", "unclear_need_more_data"] else "green-box")
            st.markdown(f"""
            <div class='{cls}'>
            <b>{im.imaging_needed_now} / {im.urgency}: {im.imaging_type} — {im.body_region}</b><br>
            <b>Reason:</b> {im.indication_or_reason}<br>
            <b>Why not if no:</b> {im.why_not_needed_if_no}<br>
            <b>Protocol:</b> {im.protocol_notes}<br>
            <b>Statement:</b> {im.explicit_no_imaging_statement}
            </div>
            """, unsafe_allow_html=True)

    if a.differential_diagnosis:
        st.subheader("Differential Diagnosis")
        for d in a.differential_diagnosis:
            st.markdown(f"""
            <div class='purple-box'>
            <b>{d.probability} / {d.urgency}: {d.diagnosis_or_category}</b><br>
            <b>Domain:</b> {d.specialty_domain}<br>
            <b>Confirm/exclude:</b> {d.confirm_or_exclude_step}
            </div>
            """, unsafe_allow_html=True)
            with st.expander("Details"):
                st.write("Supporting:", d.supporting_features or ["—"])
                st.write("Against:", d.features_against or ["—"])
                st.write("Missing:", d.missing_data_needed or ["—"])

    if a.interpreted_results:
        st.subheader("Results Interpretation")
        for x in a.interpreted_results:
            with st.expander(f"{x.finding} — {x.confidence}"):
                st.write("Source:", x.source)
                st.write("Interpretation:", x.interpretation)
                st.write("Effect:", x.effect_on_differential)

    if a.medication_safety:
        st.subheader("Medication Safety")
        for m in a.medication_safety:
            st.markdown(f"""
            <div class='orange-box'>
            <b>{m.severity}: {m.medication_or_issue}</b><br>
            <b>Concern:</b> {m.concern}<br>
            <b>Check:</b> {", ".join(m.check_before_treatment) if m.check_before_treatment else "—"}<br>
            <b>Avoid/caution:</b> {m.avoid_or_caution}<br>
            <b>Safer:</b> {m.safer_consideration}
            </div>
            """, unsafe_allow_html=True)

    if a.treatment_support_after_results:
        st.subheader("Treatment Support — no dosing")
        for t in a.treatment_support_after_results:
            st.markdown(f"""
            <div class='orange-box'>
            <b>Goal:</b> {t.clinical_goal}<br>
            <b>Option/class:</b> {t.possible_class_or_option}<br>
            <b>Consider when:</b> {t.when_to_consider}<br>
            <b>Must check:</b> {", ".join(t.must_check_before) if t.must_check_before else "—"}<br>
            <b>Avoid if:</b> {", ".join(t.avoid_if) if t.avoid_if else "—"}<br>
            <b>Monitoring:</b> {t.monitoring}<br>
            <b>Note:</b> {t.note_no_dosing}
            </div>
            """, unsafe_allow_html=True)

    if a.follow_up_thresholds:
        st.subheader("Follow-up / Return Precautions")
        for f in a.follow_up_thresholds:
            cls = "red-box" if f.timeframe == "immediate_ER" else ("orange-box" if f.timeframe in ["same_day", "24_48h"] else "green-box")
            st.markdown(f"""
            <div class='{cls}'>
            <b>{f.timeframe}</b><br>
            <b>Situation:</b> {f.situation}<br>
            <b>Action:</b> {f.action}
            </div>
            """, unsafe_allow_html=True)

    if a.trusted_reference_anchors:
        st.subheader("Trusted Reference Anchors")
        for ref in a.trusted_reference_anchors:
            st.info(f"**{ref.source_or_framework}:** {ref.principle_used} → {ref.how_it_applies_here}")

    st.subheader("Quality Control")
    qc = a.quality_control
    st.markdown(f"""
    <div class='gray-box'>
    <b>Completeness:</b> {qc.completeness_level}<br>
    <b>Overtesting:</b> {qc.overtesting_check}<br>
    <b>Undertesting:</b> {qc.undertesting_check}<br>
    <b>Medication safety:</b> {qc.medication_safety_check}<br>
    <b>General medicine mimics:</b> {qc.general_medicine_mimics_check}<br>
    <b>Clinician override:</b> {qc.clinician_override_needed_when}
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Strict Quality Check")
    st.write(a.strict_quality_check)
    st.subheader("What to do now")
    st.write(a.what_to_do_now)
    st.subheader("What to enter next")
    st.write(a.what_to_enter_next)
    st.subheader("Referral / ER Threshold")
    st.warning(a.referral_or_er_threshold)
    st.subheader("Patient Explanation Arabic")
    st.write(a.patient_explanation_arabic)
    st.subheader("SOAP")
    st.write("**S:**", a.soap_note.subjective)
    st.write("**O:**", a.soap_note.objective)
    st.write("**A:**", a.soap_note.assessment)
    st.write("**P:**", a.soap_note.plan)
    if a.limitations:
        st.subheader("Limitations")
        for l in a.limitations:
            st.caption("- " + l)


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:
    st.title("🧠 Neuro-General Guard v4.3")
    model = st.text_input("OpenAI model", value=DEFAULT_MODEL)
    strictness = st.selectbox("Safety strictness", ["Very strict", "Strict", "Balanced"], index=0)
    focus = st.selectbox("Neurology focus", [
        "General Neurology",
        "Headache / Migraine / Raised ICP / CSF leak",
        "Seizure / Syncope",
        "Stroke / TIA / Weakness",
        "Dizziness / Vertigo",
        "Neuropathy / Numbness",
        "Myelopathy / Radiculopathy / Neck-back pain",
        "Movement disorder",
        "Cognitive / psychiatric-neuro overlap"
    ])
    body_system_focus = st.selectbox("General medicine focus", [
        "Auto-route all relevant systems",
        "General Medicine",
        "Emergency Medicine",
        "Cardiology",
        "Pulmonology",
        "Endocrinology/Metabolic",
        "Infectious Disease",
        "Rheumatology",
        "Nephrology/Urology",
        "Gastroenterology/Hepatology",
        "Hematology",
        "ENT",
        "Psychiatry",
        "Orthopedics/MSK",
        "Dermatology",
        "Pediatrics",
        "Gynecology/Pregnancy"
    ])
    st.warning("Clinical decision support only. القرار النهائي للطبيب.")
    st.caption("استخدم Patient ID بدل اسم المريض.")


# =========================================================
# Main UI
# =========================================================

st.title("MedAssist Neuro-General Guard v4.3")
st.caption("يشمل أمراض الأعصاب + الطب العام: Safety Gate، module routing، فحص سريري شامل، differential متعدد الاختصاصات، أمان دوائي.")

top = st.columns(8)
top[0].metric("1", "Intake")
top[1].metric("2", "Neuro")
top[2].metric("3", "General")
top[3].metric("4", "Questions")
top[4].metric("5", "Exam")
top[5].metric("6", "Dx")
top[6].metric("7", "Results")
top[7].metric("8", "Safety")

tabs = st.tabs([
    "① Intake",
    "② Neuro Screen",
    "③ General Medicine Screen",
    "④ Questions",
    "⑤ Clinical Exam",
    "⑥ Dx & Workup",
    "⑦ Results / Imaging",
    "⑧ Full Review / Medication",
    "⑨ Report / Search"
])

uploaded_files = []

with tabs[0]:
    st.header("① Intake")
    clinical_search = st.text_input("🔎 Clinical search / سؤال سريري سريع", placeholder="مثال: ما التشخيصات المحتملة؟ هل الحالة عصبية أم قلبية؟ ما الفحص المطلوب؟")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        patient_id = st.text_input("Patient ID", value="Patient-001")
    with c2:
        age = st.number_input("Age", 0, 120, 30)
    with c3:
        sex = st.selectbox("Sex", ["Male", "Female", "Other/Unknown"])
    with c4:
        pregnancy = st.selectbox("Pregnancy/Postpartum", ["Not relevant/Unknown", "Not pregnant", "Pregnant", "Postpartum", "Possible"])

    setting = st.selectbox("Setting", ["Clinic", "ER", "Ward", "Telemedicine", "Other"])
    complaint = st.text_area("Chief complaint / الشكوى الأساسية", height=90)
    onset = st.text_area("Onset/timing / البداية والتوقيت", height=80)
    course = st.text_area("Course/progression / التطور", height=80)
    quality = st.text_area("Quality/sensation / طبيعة الإحساس", height=80)
    severity = st.text_area("Severity/function impact / الشدة والتأثير", height=80)
    associated = st.text_area("Associated symptoms / ROS / أعراض مرافقة", height=120)

    st.subheader("Medication safety")
    meds = st.text_area("Current medications", height=90)
    allergies = st.text_area("Allergies", height=70)
    safety = st.text_area("Renal/liver/pregnancy/bleeding risks", height=70)
    vitals = st.text_area("Vitals", height=80)
    doctor_question = st.text_area("Additional doctor question", height=80)

with tabs[1]:
    st.header("② Neuro Screen")
    focal = st.text_area("Focal neuro: weakness/numbness/speech/vision/gait", height=80)
    headache_icp_csf = st.text_area("Headache / raised ICP / CSF leak features", height=100, placeholder="thunderclap, papilledema, positional, Valsalva, clear rhinorrhea/otorrhea...")
    seizure_syncope = st.text_area("Seizure/syncope features", height=90)
    vertigo = st.text_area("Dizziness/vertigo features", height=90)
    weakness_neuropathy = st.text_area("Numbness/weakness/back-neck/radicular symptoms", height=90)
    previous_neuro = st.text_area("Previous neuro episodes/history", height=80)

with tabs[2]:
    st.header("③ General Medicine Screen")
    cardiac = st.text_area("Cardiac symptoms/risk", height=80, placeholder="chest pain, palpitations, syncope, dyspnea, risk factors...")
    resp = st.text_area("Respiratory symptoms/risk", height=80)
    infection = st.text_area("Infection/systemic symptoms", height=80, placeholder="fever, chills, weight loss, rash, neck stiffness...")
    endo = st.text_area("Endocrine/metabolic symptoms/risk", height=80, placeholder="diabetes, thyroid, hypoglycemia symptoms, electrolytes...")
    gi_renal = st.text_area("GI/renal/hepatic symptoms", height=80)
    rheum_msk_skin = st.text_area("Rheum/MSK/skin symptoms", height=80)
    psych_sleep_substance = st.text_area("Psychiatric/sleep/substance symptoms", height=80)
    risk_red = st.text_area("Trauma/cancer/immunosuppression/other risks", height=80)

with tabs[4]:
    st.header("⑤ Clinical Exam")
    general_exam = st.text_area("General appearance/vitals exam", height=80)
    neuro_exam = st.text_area("Neurologic exam", height=110, placeholder="mental status, cranial nerves, motor, reflexes, sensory, coordination, gait, fundoscopy...")
    cardio_resp_exam = st.text_area("Cardiovascular/respiratory exam", height=90)
    abd_renal_exam = st.text_area("Abdominal/renal exam", height=80)
    ent_msk_skin_exam = st.text_area("ENT/MSK/skin/rheum exam", height=90)
    psych_exam = st.text_area("Psychiatric/cognitive exam", height=80)

with tabs[6]:
    st.header("⑦ Results / Imaging")
    labs = st.text_area("Labs", height=120)
    imaging = st.text_area("Imaging report text: MRI/CT/X-ray/Ultrasound", height=150)
    other = st.text_area("ECG/EEG/EMG/Echo/Other report", height=120)
    uploaded_files = st.file_uploader("Upload PDF/images/reports", type=["pdf", "png", "jpg", "jpeg", "webp", "txt", "csv", "xlsx", "docx"], accept_multiple_files=True)
    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} file(s)")

def context_now():
    return make_context({
        "clinical_search": clinical_search,
        "patient_id": patient_id, "age": age, "sex": sex, "pregnancy": pregnancy,
        "setting": setting, "complaint": complaint, "onset": onset, "course": course,
        "quality": quality, "severity": severity, "associated": associated,
        "focal": focal, "headache_icp_csf": headache_icp_csf, "seizure_syncope": seizure_syncope,
        "vertigo": vertigo, "weakness_neuropathy": weakness_neuropathy, "previous_neuro": previous_neuro,
        "cardiac": cardiac, "resp": resp, "infection": infection, "endo": endo,
        "gi_renal": gi_renal, "rheum_msk_skin": rheum_msk_skin,
        "psych_sleep_substance": psych_sleep_substance, "risk_red": risk_red,
        "meds": meds, "allergies": allergies, "safety": safety, "vitals": vitals,
        "general_exam": general_exam, "neuro_exam": neuro_exam, "cardio_resp_exam": cardio_resp_exam,
        "abd_renal_exam": abd_renal_exam, "ent_msk_skin_exam": ent_msk_skin_exam,
        "psych_exam": psych_exam,
        "labs": labs, "imaging": imaging, "other": other, "doctor_question": doctor_question,
    })

def analyze_button(label, stage):
    if st.button(label, type="primary", use_container_width=True):
        with st.spinner("AI يحلل مع تغطية أعصاب + طب عام..."):
            try:
                ctx = context_now()
                a = run_ai(stage, focus, body_system_focus, strictness, ctx, uploaded_files, model)
                st.session_state[f"analysis_{stage}"] = a
                st.session_state["last_analysis"] = a
                st.session_state["last_context"] = ctx
                st.session_state["last_md"] = save_report(ctx, a)
                st.success("تم التحليل")
            except Exception as e:
                st.error("حدث خطأ أثناء التحليل")
                st.exception(e)

with tabs[3]:
    st.header("④ Questions")
    st.markdown("<div class='workflow-card'>يعطي أسئلة إلزامية حسب المسارات المفعّلة: أعصاب + طب عام + أمان دوائي.</div>", unsafe_allow_html=True)
    analyze_button("Generate questions + module routing", "questions")
    if "analysis_questions" in st.session_state:
        render(st.session_state["analysis_questions"])

with tabs[4]:
    st.divider()
    analyze_button("Suggest and interpret clinical exam", "exam")
    if "analysis_exam" in st.session_state:
        render(st.session_state["analysis_exam"])

with tabs[5]:
    st.header("⑥ Dx & Workup")
    st.markdown("<div class='workflow-card'>تحليل تفريقي متعدد الاختصاصات مع فحوصات وصور فقط عند الحاجة.</div>", unsafe_allow_html=True)
    analyze_button("Analyze Dx & Workup", "preliminary")
    if "analysis_preliminary" in st.session_state:
        render(st.session_state["analysis_preliminary"])

with tabs[7]:
    st.header("⑧ Full Review / Medication")
    analyze_button("Analyze results + medication safety", "results")
    if "analysis_results" in st.session_state:
        render(st.session_state["analysis_results"])

    st.divider()
    analyze_button("Full Neuro-General Guard Review", "full_review")
    if "analysis_full_review" in st.session_state:
        render(st.session_state["analysis_full_review"])

with tabs[8]:
    st.header("⑨ Report / Search")
    search_history = st.text_input("🔎 Search saved reports", placeholder="headache, seizure, chest pain, MRI, patient id...")
    reports = sorted(DATA_DIR.glob("neuro_general_guard_*.md"), reverse=True)

    if "last_analysis" in st.session_state:
        md = report_markdown(st.session_state["last_analysis"])
        st.download_button("Download Markdown report", data=md.encode("utf-8"), file_name=f"neuro_general_guard_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md", mime="text/markdown")
        st.download_button("Download JSON", data=st.session_state["last_analysis"].model_dump_json(indent=2).encode("utf-8"), file_name=f"neuro_general_guard_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json")

    if reports:
        shown = 0
        for r in reports[:50]:
            txt = r.read_text(encoding="utf-8")
            if search_history and search_history.lower() not in txt.lower() and search_history.lower() not in r.name.lower():
                continue
            shown += 1
            with st.expander(r.name):
                st.text(txt[:7000])
        if shown == 0:
            st.info("لا توجد نتائج مطابقة للبحث.")
    else:
        st.info("لا توجد تقارير محفوظة بعد.")

    with st.expander("Preview full context sent to AI"):
        st.text(context_now())
