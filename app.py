
import base64
import json
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

import streamlit as st
from openai import OpenAI
from pydantic import BaseModel, Field


# =========================================================
# MedAssist Level-5 AutoEvidence GPT-5 Strong v5.1
# General case-independent clinical decision-support engine.
# No hard-coded disease-specific patching.
# Workflow:
# Case -> web evidence search -> questions -> exam protocol -> diagnostic map -> results -> action pathway
# =========================================================

st.set_page_config(
    page_title="MedAssist Level-5 AutoEvidence GPT-5 Strong v5.1",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
DEFAULT_MODEL = "gpt-5.5"


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
# Pydantic schemas
# =========================================================

class SafetyGate(BaseModel):
    triage_level: Literal["emergency", "same_day", "24_48h", "routine", "unclear"]
    emergency_now: Literal["yes", "no", "unclear"]
    same_day_needed: Literal["yes", "no", "unclear"]
    reason: str
    immediate_action: str
    must_not_miss_conditions: List[str] = Field(default_factory=list)
    red_flag_thresholds: List[str] = Field(default_factory=list)


class ActivatedSpecialty(BaseModel):
    specialty: str
    urgency: Literal["emergency", "same_day", "24_48h", "routine", "watchful_waiting"]
    why_relevant: str
    what_this_specialty_must_rule_out: List[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    clinical_question: str
    recommendation_or_claim: str
    source_type: Literal["automatic_web_search", "uploaded_reference", "doctor_entered_reference", "clinical_reasoning_only", "not_verified"]
    source_title: str = ""
    source_organization: str = ""
    source_year_or_date: str = ""
    url_or_citation: str = ""
    evidence_point: str = ""
    evidence_strength: Literal["strong", "moderate", "low", "uncertain"]
    verification_status: Literal["verified_from_web_search", "verified_from_uploaded_or_entered_reference", "framework_based_not_live_checked", "not_verified", "needs_manual_reference_check"]
    applicability_to_this_case: str
    limitation_or_caution: str


class ReasoningSummary(BaseModel):
    hypothesis_or_problem: str
    why_considered: str
    features_supporting: List[str] = Field(default_factory=list)
    features_against: List[str] = Field(default_factory=list)
    missing_data_that_would_change_it: List[str] = Field(default_factory=list)
    next_best_step: str


class MissingDataItem(BaseModel):
    item: str
    why_it_matters: str
    blocks_safe_decision: bool
    how_to_get_it: str


class StageQuestion(BaseModel):
    question: str
    stage: Literal["before_exam", "after_exam", "before_tests", "after_tests", "before_treatment", "followup"]
    domain: str
    priority: Literal["must_ask_now", "important", "routine"]
    why_ask: str
    how_answer_changes_decision: str


class ExamProtocolStep(BaseModel):
    section: str
    exam_item: str
    timing: Literal["must_do_now", "important", "routine", "conditional"]
    patient_position_or_setup: str
    how_to_perform_step_by_step: List[str] = Field(default_factory=list)
    record_exactly: List[str] = Field(default_factory=list)
    normal_expected: str
    abnormal_meaning: List[str] = Field(default_factory=list)
    stop_or_escalate_if: str


class ExamInterpretation(BaseModel):
    finding_entered: str
    interpretation: str
    supports: List[str] = Field(default_factory=list)
    argues_against_but_does_not_exclude: List[str] = Field(default_factory=list)
    next_step_triggered: str


class DifferentialDiagnosis(BaseModel):
    diagnosis_or_category: str
    domain: str
    probability: Literal["high", "medium", "low", "cannot_rank"]
    urgency: Literal["emergency", "same_day", "24_48h", "routine"]
    why_possible: List[str] = Field(default_factory=list)
    why_less_likely: List[str] = Field(default_factory=list)
    key_missing_data: List[str] = Field(default_factory=list)
    rule_in_rule_out_step: str
    evidence_support: str


class WorkupItem(BaseModel):
    test_or_action: str
    category: Literal["bedside", "lab", "cardiac", "neurophysiology", "imaging", "procedure", "monitoring", "referral", "other"]
    priority: Literal["urgent_now", "same_day", "conditional", "routine", "not_now"]
    why_needed: str
    what_result_changes: str
    when_to_avoid_or_defer: str


class DiagnosticMapItem(BaseModel):
    test_or_image: str
    category: Literal[
        "bedside", "lab", "cardiac", "neurophysiology", "imaging_brain",
        "imaging_spine", "imaging_vascular", "imaging_chest",
        "imaging_abdomen_pelvis", "procedure", "specialist_referral", "other"
    ]
    timing: Literal["must_do_now", "same_day_if_available", "conditional_next_step", "specialist_level", "not_indicated_now"]
    indication_in_this_case: str
    trigger_to_order: str
    why_not_now_if_not_indicated: str
    what_result_would_change: str
    protocol_or_notes: str
    danger_if_missed_when_indicated: str


class ImagingDecision(BaseModel):
    imaging_type: str
    body_region: str
    needed_now: Literal["yes", "no", "conditional", "unclear"]
    timing: Literal["emergency", "same_day", "24_48h", "routine", "not_indicated_now"]
    indication_or_reason: str
    trigger_if_not_now: str
    protocol_notes: str
    overtesting_warning: str


class ResultInterpretation(BaseModel):
    source: str
    finding: str
    interpretation: str
    effect_on_differential: str
    next_step: str
    confidence: Literal["high", "medium", "low"]


class MedicationSafetyItem(BaseModel):
    medication_or_class: str
    issue: str
    safety_checks_before_use: List[str] = Field(default_factory=list)
    avoid_or_use_caution_if: List[str] = Field(default_factory=list)
    monitoring_needed: str
    non_dosing_note: str


class ActionPathwayItem(BaseModel):
    step: str
    when: Literal["now", "if_abnormal", "if_normal_but_recurrent_or_persistent", "if_red_flag", "followup"]
    action: str
    reason: str
    escalation_threshold: str


class QualityControl(BaseModel):
    completeness: Literal["safe_enough_for_next_step", "partial_needs_more_data", "unsafe_missing_critical_data"]
    overtesting_check: str
    undertesting_check: str
    evidence_check: str
    medication_safety_check: str
    clinician_override_needed_when: str


class SOAP(BaseModel):
    subjective: str
    objective: str
    assessment: str
    plan: str


class Level5Analysis(BaseModel):
    stage: str
    case_summary: str
    problem_representation: str
    safety_gate: SafetyGate
    activated_specialties: List[ActivatedSpecialty] = Field(default_factory=list)
    evidence_verification: List[EvidenceItem] = Field(default_factory=list)
    clinical_reasoning_summary: List[ReasoningSummary] = Field(default_factory=list)
    missing_data: List[MissingDataItem] = Field(default_factory=list)
    questions_before_exam: List[StageQuestion] = Field(default_factory=list)
    questions_after_exam: List[StageQuestion] = Field(default_factory=list)
    questions_before_tests: List[StageQuestion] = Field(default_factory=list)
    questions_after_tests: List[StageQuestion] = Field(default_factory=list)
    questions_before_treatment: List[StageQuestion] = Field(default_factory=list)
    followup_questions: List[StageQuestion] = Field(default_factory=list)
    exam_protocol: List[ExamProtocolStep] = Field(default_factory=list)
    exam_interpretation: List[ExamInterpretation] = Field(default_factory=list)
    differential_diagnosis: List[DifferentialDiagnosis] = Field(default_factory=list)
    recommended_workup: List[WorkupItem] = Field(default_factory=list)
    comprehensive_diagnostic_map: List[DiagnosticMapItem] = Field(default_factory=list)
    imaging_decisions: List[ImagingDecision] = Field(default_factory=list)
    result_interpretation: List[ResultInterpretation] = Field(default_factory=list)
    medication_safety: List[MedicationSafetyItem] = Field(default_factory=list)
    action_pathway: List[ActionPathwayItem] = Field(default_factory=list)
    quality_control: QualityControl
    what_to_do_now: str
    what_to_enter_next: str
    referral_or_er_threshold: str
    patient_explanation_arabic: str
    soap_note: SOAP
    limitations: List[str] = Field(default_factory=list)


# =========================================================
# General prompts
# =========================================================

GENERAL_METHOD = """
You are MedAssist Level-5 AutoEvidence GPT-5 Strong v5.1.

Role:
- General clinical decision-support engine for clinicians.
- Do NOT specialize in only one case or one disease.
- Do NOT rely on hard-coded disease-specific patches.
- For every case, dynamically:
  1. Build a problem representation.
  2. Identify safety gate and must-not-miss diagnoses.
  3. Generate broad differential across relevant specialties.
  4. Use automatic evidence research text when available.
  5. Ask missing questions.
  6. Give a detailed clinical exam protocol.
  7. Give recommended workup.
  8. Give a comprehensive diagnostic map, including tests/images not indicated now and exact triggers.
  9. Interpret results if entered.
  10. Provide action pathway.
  11. Provide medication safety without dosing.

Important:
- Show a clinical reasoning summary, but do not expose hidden chain-of-thought.
- Use concise clinical reasoning: supports / against / missing data / next step.
- Never claim a source was checked unless evidence text or uploaded reference supports it.
- If evidence is absent or vague, mark it as not_verified or needs_manual_reference_check.
- Do not give final diagnosis. Give differential and next safe steps.
- Do not give medication doses.
- Be strict about emergency red flags.
- Avoid both overtesting and undertesting.
- Always include a full diagnostic map. This map is not an order set; it is a roadmap with triggers.
- Use Arabic-friendly medical language with English medical terms when useful.
"""


def get_api_key():
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return None

def model_fallbacks(preferred_model: str):
    """Return model fallback chain without duplicates."""
    chain = [
        preferred_model,
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-4o",
        "gpt-4o-mini",
    ]
    out = []
    for m in chain:
        if m and m not in out:
            out.append(m)
    return out


def selected_model_from_preset(preset: str, manual_model: str):
    if preset == "GPT-5.5 strongest":
        return "gpt-5.5"
    if preset == "GPT-5.4 strong":
        return "gpt-5.4"
    if preset == "GPT-5.4 mini balanced":
        return "gpt-5.4-mini"
    if preset == "Manual":
        return manual_model
    return manual_model



def stage_instruction(stage: str, depth: str, source_mode: str):
    return f"""
STAGE: {stage}
Reasoning depth: {depth}
Evidence source mode: {source_mode}

Instructions for this stage:
- If depth is Level 5, be comprehensive and systematic.
- Do not rely on any single prebuilt disease workflow.
- Search/evidence text may include several sources; use it, but verify applicability.
- If the case is incomplete, say what is missing instead of over-ranking.
- Always return according to the schema.

Stage-specific:
- questions: emphasize missing history and red flags before exam/tests.
- exam_protocol: tell clinician how to perform clinical exam step-by-step.
- exam_interpretation: interpret entered exam findings and update hypotheses.
- dx_workup: produce differential, workup, diagnostic map, imaging decisions, action pathway.
- results: interpret labs/imaging/reports and update differential/action.
- full_review: complete report.
"""


def make_context(data):
    return f"""
PATIENT BASICS
Patient ID: {data.get("patient_id")}
Age: {data.get("age")}
Sex: {data.get("sex")}
Pregnancy/Postpartum: {data.get("pregnancy")}
Setting: {data.get("setting")}

CLINICAL QUESTION / DOCTOR GOAL
{data.get("clinical_question")}

MAIN HISTORY
Chief complaint: {data.get("complaint")}
Onset/timing: {data.get("onset")}
Course/progression: {data.get("course")}
Quality/sensation: {data.get("quality")}
Severity/function impact: {data.get("severity")}
Associated symptoms / ROS: {data.get("associated")}
Past medical history: {data.get("pmh")}
Family/social/substance history: {data.get("social")}
Current medications: {data.get("meds")}
Allergies: {data.get("allergies")}
Medication safety risks: {data.get("safety")}
Vitals: {data.get("vitals")}

SYSTEM SCREENS
Neurology screen: {data.get("neuro_screen")}
Cardiac/vascular screen: {data.get("cardiac_screen")}
Respiratory screen: {data.get("resp_screen")}
Infection/systemic screen: {data.get("infection_screen")}
Endocrine/metabolic screen: {data.get("endo_screen")}
GI/renal/hepatic screen: {data.get("gi_renal_screen")}
Rheum/MSK/skin screen: {data.get("rheum_msk_skin_screen")}
Psychiatric/sleep screen: {data.get("psych_sleep_screen")}
Trauma/toxicology/other risks: {data.get("risk_screen")}

EXAM FINDINGS ENTERED BY CLINICIAN
General/vitals exam: {data.get("general_exam")}
Neurologic exam: {data.get("neuro_exam")}
Cardiovascular exam: {data.get("cardio_exam")}
Respiratory exam: {data.get("resp_exam")}
Abdominal/renal exam: {data.get("abd_exam")}
ENT/MSK/skin/rheum exam: {data.get("ent_msk_skin_exam")}
Psych/cognitive exam: {data.get("psych_exam")}
Other exam: {data.get("other_exam")}

RESULTS ENTERED
Labs: {data.get("labs")}
Imaging reports: {data.get("imaging")}
ECG/EEG/EMG/Echo/Other: {data.get("other_results")}

DOCTOR ENTERED REFERENCES / GUIDELINES
{data.get("reference_notes")}

AUTOMATIC WEB EVIDENCE RESEARCH
{data.get("evidence_text")}
"""


def build_evidence_query(stage, context, depth):
    return f"""
You are doing automatic medical evidence research for a clinical decision-support app.

Goal:
Research the case dynamically, not from a preselected disease workflow.

Depth: {depth}
Stage: {stage}

Search strategy:
1. Identify the main clinical problem and 5-10 plausible or dangerous hypotheses.
2. Search authoritative open medical sources for:
   - red flags and triage
   - initial clinical exam
   - initial workup
   - imaging indications / when not to image
   - specialist referral thresholds
   - medication safety
3. Prefer official guidelines, medical societies, government/public health sources, peer-reviewed reviews, systematic reviews, and reputable clinical references.
4. Do not use random blogs/forums.
5. Do not claim access to paid sources like UpToDate unless an accessible source is actually returned.
6. Return source title, organization, year/date if visible, URL/citation if visible, and the evidence point in your own words.
7. Note uncertainty and when manual specialist/guideline review is needed.

Case context:
{context}
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
        text = raw.decode("utf-8", errors="ignore")[:30000]
        return {"type": "input_text", "text": f"Uploaded text file {name}:\n{text}"}
    return {"type": "input_file", "filename": name, "file_data": f"data:{mime};base64,{b64}"}


def run_evidence_search(stage, context, depth, model):
    key = get_api_key()
    if not key:
        return "Automatic evidence search skipped: OPENAI_API_KEY missing."

    client = OpenAI(api_key=key)
    query = build_evidence_query(stage, context, depth)

    errors = []
    for candidate_model in model_fallbacks(model):
        try:
            response = client.responses.create(
                model=candidate_model,
                tools=[{"type": "web_search"}],
                tool_choice="required",
                input=query,
            )
            return f"Evidence search model used: {candidate_model}\n\n" + ((getattr(response, "output_text", None) or str(response))[:25000])
        except Exception as e1:
            errors.append(f"{candidate_model} required-search error: {e1}")
            try:
                response = client.responses.create(
                    model=candidate_model,
                    tools=[{"type": "web_search"}],
                    input=query,
                )
                return f"Evidence search fallback used: {candidate_model} without required tool_choice.\n\n" + ((getattr(response, "output_text", None) or str(response))[:25000])
            except Exception as e2:
                errors.append(f"{candidate_model} fallback error: {e2}")

    return "Automatic evidence search failed. Do not mark claims as web-verified. Errors:\n" + "\n".join(errors[-8:])


def run_ai(stage, context, evidence_files, model, depth, source_mode):
    key = get_api_key()
    if not key:
        st.error("OPENAI_API_KEY غير موجود في Streamlit Secrets.")
        st.stop()

    client = OpenAI(api_key=key)
    content = [{"type": "input_text", "text": stage_instruction(stage, depth, source_mode) + "\n\nCASE CONTEXT:\n" + context}]

    for f in evidence_files or []:
        content.append(file_item(f))

    errors = []
    for candidate_model in model_fallbacks(model):
        try:
            response = client.responses.parse(
                model=candidate_model,
                input=[
                    {"role": "system", "content": GENERAL_METHOD + f"\n\nClinical analysis model used: {candidate_model}"},
                    {"role": "user", "content": content},
                ],
                text_format=Level5Analysis,
            )
            parsed = response.output_parsed
            parsed.limitations.append(f"Clinical analysis model used: {candidate_model}")
            if candidate_model != model:
                parsed.limitations.append(f"Preferred model {model} was unavailable or failed; fallback model used.")
            return parsed
        except Exception as e:
            errors.append(f"{candidate_model}: {e}")

    raise RuntimeError("All model fallbacks failed:\n" + "\n".join(errors[-8:]))


# =========================================================
# Generic cleanup: not case-specific
# =========================================================

def _lc(x):
    return (x or "").lower()


def canonical_key(name: str):
    t = _lc(name)
    if "ecg" in t or "electrocardiogram" in t:
        if "holter" in t or "ambulatory" in t or "event" in t:
            return "ambulatory_ecg"
        return "ecg"
    if "orthostatic" in t:
        return "orthostatic_vitals"
    if "glucose" in t:
        return "glucose"
    if "cbc" in t:
        return "cbc"
    if "bmp" in t or "electrolyte" in t or "renal" in t or "magnesium" in t or "calcium" in t or "mg/ca" in t:
        return "bmp_electrolytes_renal"
    if "troponin" in t:
        return "troponin"
    if "d-dimer" in t or "d dimer" in t:
        return "d_dimer"
    if "ct brain" in t or "ct head" in t:
        return "ct_brain"
    if "mri brain" in t:
        return "mri_brain"
    if "cta" in t or "mra" in t:
        return "cta_mra"
    if "eeg" in t:
        return "eeg"
    if "echo" in t:
        return "echo"
    return "".join(ch for ch in t if ch.isalnum() or ch == " ")[:70]


def timing_rank(x):
    order = {
        "urgent_now": 0, "must_do_now": 0, "emergency": 0,
        "same_day": 1, "same_day_if_available": 1,
        "conditional": 2, "conditional_next_step": 2,
        "routine": 3, "specialist_level": 3,
        "not_now": 4, "not_indicated_now": 4,
    }
    return order.get(str(x), 9)


def dedupe_keep_strongest(items, name_fn, timing_fn):
    best = {}
    sequence = []
    for item in items:
        key = canonical_key(name_fn(item))
        if key not in best:
            best[key] = item
            sequence.append(key)
        else:
            if timing_rank(timing_fn(item)) < timing_rank(timing_fn(best[key])):
                best[key] = item
    return [best[k] for k in sequence]


def generic_cleanup(a: Level5Analysis) -> Level5Analysis:
    # Dedupe lists without disease-specific logic.
    seen = set()
    specs = []
    for s in a.activated_specialties:
        key = _lc(s.specialty)
        if key not in seen:
            specs.append(s)
            seen.add(key)
    a.activated_specialties = specs

    a.recommended_workup = dedupe_keep_strongest(a.recommended_workup, lambda x: x.test_or_action, lambda x: x.priority)
    a.comprehensive_diagnostic_map = dedupe_keep_strongest(a.comprehensive_diagnostic_map, lambda x: x.test_or_image, lambda x: x.timing)
    a.comprehensive_diagnostic_map = sorted(a.comprehensive_diagnostic_map, key=lambda x: (timing_rank(x.timing), x.category, x.test_or_image))

    # If no map returned, add a generic warning map placeholder.
    if not a.comprehensive_diagnostic_map:
        a.comprehensive_diagnostic_map.append(DiagnosticMapItem(
            test_or_image="Diagnostic map missing from model output",
            category="other",
            timing="conditional_next_step",
            indication_in_this_case="The model did not return a diagnostic map.",
            trigger_to_order="Repeat analysis or provide more case details.",
            why_not_now_if_not_indicated="Not applicable.",
            what_result_would_change="Not applicable.",
            protocol_or_notes="Use this as a quality-control flag.",
            danger_if_missed_when_indicated="Without a map, tests/images may be under-considered."
        ))

    return a


# =========================================================
# Rendering
# =========================================================

def report_markdown(a):
    return "# MedAssist Level-5 AutoEvidence GPT-5 Strong v5.1 Report\n\n```json\n" + a.model_dump_json(indent=2) + "\n```"


def save_report(context, a):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md = report_markdown(a)
    (DATA_DIR / f"level5_report_{stamp}.md").write_text(md, encoding="utf-8")
    (DATA_DIR / f"level5_report_{stamp}.json").write_text(json.dumps({
        "created": stamp,
        "context": context,
        "analysis": a.model_dump()
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return md


def box(class_name, html):
    st.markdown(f"<div class='{class_name}'>{html}</div>", unsafe_allow_html=True)


def render_questions(title, items):
    if items:
        st.subheader(title)
        for i, q in enumerate(items, 1):
            with st.expander(f"{i}. {q.priority} / {q.domain} — {q.question}"):
                st.write("**Stage:**", q.stage)
                st.write("**Why:**", q.why_ask)
                st.write("**Decision impact:**", q.how_answer_changes_decision)


def render(a: Level5Analysis):
    st.subheader("Level-5 Clinical Dashboard")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Triage", a.safety_gate.triage_level)
    c2.metric("Emergency", a.safety_gate.emergency_now)
    c3.metric("Specialties", len(a.activated_specialties))
    c4.metric("Differentials", len(a.differential_diagnosis))
    c5.metric("Diagnostic map", len(a.comprehensive_diagnostic_map))

    box("workflow-card", f"""
    <div class='title-big'>🧠 Level-5 Summary</div>
    <b>{a.case_summary}</b><br><br>
    <b>Problem representation:</b><br>{a.problem_representation}
    """)

    if a.safety_gate.triage_level == "emergency":
        st.error(f"SAFETY GATE: EMERGENCY — {a.safety_gate.reason}")
    elif a.safety_gate.triage_level == "same_day":
        st.warning(f"SAFETY GATE: SAME DAY — {a.safety_gate.reason}")
    else:
        st.info(f"SAFETY GATE: {a.safety_gate.triage_level} — {a.safety_gate.reason}")

    box("red-box", f"""
    <b>Immediate action:</b> {a.safety_gate.immediate_action}<br>
    <b>Must-not-miss:</b> {", ".join(a.safety_gate.must_not_miss_conditions) if a.safety_gate.must_not_miss_conditions else "—"}<br>
    <b>Red flag thresholds:</b> {", ".join(a.safety_gate.red_flag_thresholds) if a.safety_gate.red_flag_thresholds else "—"}
    """)

    if a.activated_specialties:
        st.subheader("Activated Specialties")
        for s in a.activated_specialties:
            box("blue-box", f"""
            <b>{s.urgency}: {s.specialty}</b><br>
            <b>Why:</b> {s.why_relevant}<br>
            <b>Must rule out:</b> {", ".join(s.what_this_specialty_must_rule_out) if s.what_this_specialty_must_rule_out else "—"}
            """)

    if a.evidence_verification:
        st.subheader("Evidence Verification")
        for e in a.evidence_verification:
            cls = "green-box" if e.verification_status in ["verified_from_web_search", "verified_from_uploaded_or_entered_reference"] else "orange-box"
            box(cls, f"""
            <b>Question:</b> {e.clinical_question}<br>
            <b>Claim:</b> {e.recommendation_or_claim}<br>
            <b>Source:</b> {e.source_title} — {e.source_organization} — {e.source_year_or_date}<br>
            <b>Citation:</b> {e.url_or_citation}<br>
            <b>Evidence point:</b> {e.evidence_point}<br>
            <b>Strength:</b> {e.evidence_strength}<br>
            <b>Status:</b> {e.verification_status}<br>
            <b>Applicability:</b> {e.applicability_to_this_case}<br>
            <b>Caution:</b> {e.limitation_or_caution}
            """)

    if a.clinical_reasoning_summary:
        st.subheader("Clinical Reasoning Summary")
        for r in a.clinical_reasoning_summary:
            with st.expander(r.hypothesis_or_problem):
                st.write("**Why considered:**", r.why_considered)
                st.write("**Supports:**", r.features_supporting or ["—"])
                st.write("**Against:**", r.features_against or ["—"])
                st.write("**Missing data:**", r.missing_data_that_would_change_it or ["—"])
                st.write("**Next step:**", r.next_best_step)

    if a.missing_data:
        st.subheader("Missing Data")
        for m in a.missing_data:
            st.warning(f"**{m.item}** — {m.why_it_matters} | blocks safe decision: {m.blocks_safe_decision} | get it: {m.how_to_get_it}")

    render_questions("Questions BEFORE exam", a.questions_before_exam)
    render_questions("Questions AFTER exam", a.questions_after_exam)
    render_questions("Questions BEFORE tests", a.questions_before_tests)
    render_questions("Questions AFTER tests", a.questions_after_tests)
    render_questions("Questions BEFORE treatment", a.questions_before_treatment)
    render_questions("Follow-up questions", a.followup_questions)

    if a.exam_protocol:
        st.subheader("Detailed Clinical Exam Protocol — كيف أفحص؟")
        for e in a.exam_protocol:
            cls = "red-box" if e.timing == "must_do_now" else "blue-box"
            box(cls, f"""
            <b>{e.timing} / {e.section}: {e.exam_item}</b><br>
            <b>Setup:</b> {e.patient_position_or_setup}<br>
            <b>Steps:</b><br>{"<br>".join([str(i+1)+". "+x for i,x in enumerate(e.how_to_perform_step_by_step)]) if e.how_to_perform_step_by_step else "—"}<br>
            <b>Record:</b> {", ".join(e.record_exactly) if e.record_exactly else "—"}<br>
            <b>Normal:</b> {e.normal_expected}<br>
            <b>Abnormal meaning:</b> {", ".join(e.abnormal_meaning) if e.abnormal_meaning else "—"}<br>
            <b>Stop/escalate:</b> {e.stop_or_escalate_if}
            """)

    if a.exam_interpretation:
        st.subheader("Exam Interpretation")
        for e in a.exam_interpretation:
            box("green-box", f"""
            <b>{e.finding_entered}</b><br>
            <b>Interpretation:</b> {e.interpretation}<br>
            <b>Supports:</b> {", ".join(e.supports) if e.supports else "—"}<br>
            <b>Argues against but does not exclude:</b> {", ".join(e.argues_against_but_does_not_exclude) if e.argues_against_but_does_not_exclude else "—"}<br>
            <b>Next:</b> {e.next_step_triggered}
            """)

    if a.differential_diagnosis:
        st.subheader("Differential Diagnosis")
        for d in a.differential_diagnosis:
            box("purple-box", f"""
            <b>{d.probability} / {d.urgency}: {d.diagnosis_or_category}</b><br>
            <b>Domain:</b> {d.domain}<br>
            <b>Rule in/out:</b> {d.rule_in_rule_out_step}<br>
            <b>Evidence:</b> {d.evidence_support}
            """)
            with st.expander("Details"):
                st.write("Why possible:", d.why_possible or ["—"])
                st.write("Why less likely:", d.why_less_likely or ["—"])
                st.write("Missing:", d.key_missing_data or ["—"])

    if a.recommended_workup:
        st.subheader("Recommended Workup — الفحوصات المطلوبة كخطوة عملية")
        for w in a.recommended_workup:
            cls = "red-box" if w.priority == "urgent_now" else ("blue-box" if w.priority == "same_day" else "gray-box")
            box(cls, f"""
            <b>{w.priority} / {w.category}: {w.test_or_action}</b><br>
            <b>Why:</b> {w.why_needed}<br>
            <b>What changes:</b> {w.what_result_changes}<br>
            <b>Defer/avoid:</b> {w.when_to_avoid_or_defer}
            """)

    if a.comprehensive_diagnostic_map:
        st.subheader("Comprehensive Diagnostic Map — كل الفحوصات والصور مع الشروط")
        for d in a.comprehensive_diagnostic_map:
            cls = "red-box" if d.timing == "must_do_now" else ("orange-box" if d.timing in ["same_day_if_available", "conditional_next_step"] else ("purple-box" if d.timing == "specialist_level" else "gray-box"))
            box(cls, f"""
            <b>{d.timing} / {d.category}: {d.test_or_image}</b><br>
            <b>Indication in this case:</b> {d.indication_in_this_case}<br>
            <b>Trigger to order:</b> {d.trigger_to_order}<br>
            <b>Why not now:</b> {d.why_not_now_if_not_indicated}<br>
            <b>What result changes:</b> {d.what_result_would_change}<br>
            <b>Protocol/notes:</b> {d.protocol_or_notes}<br>
            <b>Danger if missed when indicated:</b> {d.danger_if_missed_when_indicated}
            """)

    if a.imaging_decisions:
        st.subheader("Imaging Decisions")
        for im in a.imaging_decisions:
            cls = "red-box" if im.needed_now == "yes" else ("orange-box" if im.needed_now in ["conditional", "unclear"] else "green-box")
            box(cls, f"""
            <b>{im.needed_now} / {im.timing}: {im.imaging_type} — {im.body_region}</b><br>
            <b>Reason:</b> {im.indication_or_reason}<br>
            <b>Trigger if not now:</b> {im.trigger_if_not_now}<br>
            <b>Protocol:</b> {im.protocol_notes}<br>
            <b>Overtesting warning:</b> {im.overtesting_warning}
            """)

    if a.result_interpretation:
        st.subheader("Results Interpretation")
        for r in a.result_interpretation:
            with st.expander(f"{r.finding} — {r.confidence}"):
                st.write("Source:", r.source)
                st.write("Interpretation:", r.interpretation)
                st.write("Effect:", r.effect_on_differential)
                st.write("Next:", r.next_step)

    if a.medication_safety:
        st.subheader("Medication Safety — no dosing")
        for m in a.medication_safety:
            box("orange-box", f"""
            <b>{m.medication_or_class}</b><br>
            <b>Issue:</b> {m.issue}<br>
            <b>Checks:</b> {", ".join(m.safety_checks_before_use) if m.safety_checks_before_use else "—"}<br>
            <b>Avoid/caution:</b> {", ".join(m.avoid_or_use_caution_if) if m.avoid_or_use_caution_if else "—"}<br>
            <b>Monitoring:</b> {m.monitoring_needed}<br>
            <b>Note:</b> {m.non_dosing_note}
            """)

    if a.action_pathway:
        st.subheader("Action Pathway — ماذا أفعل الآن وماذا بعد النتائج؟")
        for ap in a.action_pathway:
            cls = "red-box" if ap.when == "if_red_flag" else ("orange-box" if ap.when in ["if_abnormal", "if_normal_but_recurrent_or_persistent"] else "green-box")
            box(cls, f"""
            <b>{ap.when}: {ap.step}</b><br>
            <b>Action:</b> {ap.action}<br>
            <b>Reason:</b> {ap.reason}<br>
            <b>Escalation:</b> {ap.escalation_threshold}
            """)

    st.subheader("Quality Control")
    qc = a.quality_control
    box("gray-box", f"""
    <b>Completeness:</b> {qc.completeness}<br>
    <b>Overtesting:</b> {qc.overtesting_check}<br>
    <b>Undertesting:</b> {qc.undertesting_check}<br>
    <b>Evidence:</b> {qc.evidence_check}<br>
    <b>Medication safety:</b> {qc.medication_safety_check}<br>
    <b>Override:</b> {qc.clinician_override_needed_when}
    """)

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
    st.title("🧠 GPT-5 Strong v5.1")
    model_preset = st.selectbox(
        "Model power preset",
        ["GPT-5.5 strongest", "GPT-5.4 strong", "GPT-5.4 mini balanced", "Manual"],
        index=0,
        help="If your API account does not have access to the selected GPT-5 model, the app will automatically try fallbacks."
    )
    manual_model = st.text_input("Manual clinical model", value=DEFAULT_MODEL)
    model = selected_model_from_preset(model_preset, manual_model)
    st.success(f"Clinical model selected: {model}")

    evidence_preset = st.selectbox(
        "Evidence search model preset",
        ["GPT-5.4 mini balanced", "GPT-5.4 strong", "GPT-5.5 strongest", "Manual"],
        index=0
    )
    manual_evidence_model = st.text_input("Manual evidence model", value="gpt-5.4-mini")
    evidence_model = selected_model_from_preset(evidence_preset, manual_evidence_model)
    st.info(f"Evidence model selected: {evidence_model}")

    depth = st.selectbox("Reasoning depth", ["Level 5 — Deep systematic", "Level 4 — Comprehensive", "Level 3 — Balanced"], index=0)
    auto_evidence = st.checkbox("Automatic Evidence Web Search", value=True)
    source_mode = st.selectbox("Evidence mode", ["Open authoritative web + uploaded references", "Uploaded/entered references only", "Clinical reasoning only"], index=0)
    st.warning("Clinical decision support only. القرار النهائي للطبيب.")
    st.caption("لا تستخدم اسم المريض الحقيقي. استخدم Patient ID.")


# =========================================================
# UI
# =========================================================

st.title("MedAssist Level-5 AutoEvidence GPT-5 Strong v5.1")
st.caption("محرك Level-5 عام باستخدام GPT-5 Strong افتراضيًا: يقرأ أي حالة، يبحث، يفكر بشكل منظم، ثم يعطي أسئلة، فحص سريري، differential، خريطة فحوصات وصور، وaction pathway.")

tabs = st.tabs([
    "① Intake",
    "② System Screens",
    "③ Evidence / References",
    "④ Exam Findings",
    "⑤ Results / Reports",
    "⑥ Questions",
    "⑦ Exam Protocol",
    "⑧ Dx + Diagnostic Map",
    "⑨ Results Review",
    "⑩ Full Level-5 Review",
    "⑪ Report / Search"
])

uploaded_files = []

with tabs[0]:
    st.header("① Intake")
    clinical_question = st.text_area("Clinical question / هدف الطبيب", height=70, placeholder="مثال: ما التشخيصات المحتملة؟ ما الفحوصات والصور المطلوبة؟")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        patient_id = st.text_input("Patient ID", value="patient-001")
    with c2:
        age = st.number_input("Age", 0, 120, 30)
    with c3:
        sex = st.selectbox("Sex", ["Male", "Female", "Other/Unknown"])
    with c4:
        pregnancy = st.selectbox("Pregnancy/Postpartum", ["Not relevant/Unknown", "Not pregnant", "Pregnant", "Postpartum", "Possible"])

    setting = st.selectbox("Setting", ["Clinic", "ER", "Ward", "Telemedicine", "Other"])
    complaint = st.text_area("Chief complaint", height=80)
    onset = st.text_area("Onset/timing", height=70)
    course = st.text_area("Course/progression", height=70)
    quality = st.text_area("Quality/sensation", height=70)
    severity = st.text_area("Severity/function impact", height=70)
    associated = st.text_area("Associated symptoms / ROS", height=120)
    pmh = st.text_area("Past medical history", height=70)
    social = st.text_area("Family/social/substance history", height=70)
    meds = st.text_area("Current medications", height=70)
    allergies = st.text_area("Allergies", height=50)
    safety = st.text_area("Medication safety risks: renal/liver/bleeding/pregnancy/QT/sedation", height=70)
    vitals = st.text_area("Vitals", height=80)

with tabs[1]:
    st.header("② System Screens")
    neuro_screen = st.text_area("Neurology screen", height=90)
    cardiac_screen = st.text_area("Cardiac/vascular screen", height=90)
    resp_screen = st.text_area("Respiratory screen", height=80)
    infection_screen = st.text_area("Infection/systemic screen", height=80)
    endo_screen = st.text_area("Endocrine/metabolic screen", height=80)
    gi_renal_screen = st.text_area("GI/renal/hepatic screen", height=80)
    rheum_msk_skin_screen = st.text_area("Rheum/MSK/skin screen", height=80)
    psych_sleep_screen = st.text_area("Psychiatric/sleep screen", height=80)
    risk_screen = st.text_area("Trauma/toxicology/other risks", height=80)

with tabs[2]:
    st.header("③ Evidence / References")
    reference_notes = st.text_area("Doctor-entered guideline notes / reference excerpts", height=160)
    st.info("في v5.0 التطبيق لا يعتمد على حالة واحدة. إذا كان البحث الأوتوماتيكي مفعّلًا سيبحث حسب الحالة المدخلة.")

with tabs[3]:
    st.header("④ Exam Findings")
    general_exam = st.text_area("General/vitals exam", height=80)
    neuro_exam = st.text_area("Neurologic exam", height=100)
    cardio_exam = st.text_area("Cardiovascular exam", height=90)
    resp_exam = st.text_area("Respiratory exam", height=80)
    abd_exam = st.text_area("Abdominal/renal exam", height=80)
    ent_msk_skin_exam = st.text_area("ENT/MSK/skin/rheum exam", height=90)
    psych_exam = st.text_area("Psych/cognitive exam", height=80)
    other_exam = st.text_area("Other exam", height=70)

with tabs[4]:
    st.header("⑤ Results / Reports")
    labs = st.text_area("Labs", height=120)
    imaging = st.text_area("Imaging reports", height=140)
    other_results = st.text_area("ECG/EEG/EMG/Echo/Other", height=120)
    uploaded_files = st.file_uploader("Upload reports/images/PDFs", type=["pdf", "png", "jpg", "jpeg", "webp", "txt", "csv", "xlsx", "docx"], accept_multiple_files=True)

def context_now():
    return make_context({
        "patient_id": patient_id, "age": age, "sex": sex, "pregnancy": pregnancy, "setting": setting,
        "clinical_question": clinical_question, "complaint": complaint, "onset": onset, "course": course,
        "quality": quality, "severity": severity, "associated": associated, "pmh": pmh, "social": social,
        "meds": meds, "allergies": allergies, "safety": safety, "vitals": vitals,
        "neuro_screen": neuro_screen, "cardiac_screen": cardiac_screen, "resp_screen": resp_screen,
        "infection_screen": infection_screen, "endo_screen": endo_screen, "gi_renal_screen": gi_renal_screen,
        "rheum_msk_skin_screen": rheum_msk_skin_screen, "psych_sleep_screen": psych_sleep_screen, "risk_screen": risk_screen,
        "general_exam": general_exam, "neuro_exam": neuro_exam, "cardio_exam": cardio_exam,
        "resp_exam": resp_exam, "abd_exam": abd_exam, "ent_msk_skin_exam": ent_msk_skin_exam,
        "psych_exam": psych_exam, "other_exam": other_exam,
        "labs": labs, "imaging": imaging, "other_results": other_results,
        "reference_notes": reference_notes,
        "evidence_text": st.session_state.get("evidence_text", "")
    })

def analyze_button(label, stage):
    if st.button(label, type="primary", use_container_width=True):
        with st.spinner("Level-5 analysis running..."):
            try:
                base_context = context_now()
                if auto_evidence and source_mode == "Open authoritative web + uploaded references":
                    with st.spinner("Automatic evidence search..."):
                        st.session_state["evidence_text"] = run_evidence_search(stage, base_context, depth, evidence_model)
                elif source_mode == "Uploaded/entered references only":
                    st.session_state["evidence_text"] = "Automatic web search disabled. Use doctor-entered/uploaded references only."
                else:
                    st.session_state["evidence_text"] = "Evidence web search disabled. Clinical reasoning only; mark evidence as not live verified."

                ctx = context_now()
                a = run_ai(stage, ctx, uploaded_files, model, depth, source_mode)
                a = generic_cleanup(a)
                st.session_state[f"analysis_{stage}"] = a
                st.session_state["last_analysis"] = a
                st.session_state["last_context"] = ctx
                st.session_state["last_md"] = save_report(ctx, a)
                st.success("Done")
            except Exception as e:
                st.error("حدث خطأ أثناء التحليل")
                st.exception(e)

with tabs[5]:
    st.header("⑥ Questions")
    analyze_button("Generate Level-5 questions", "questions")
    if "analysis_questions" in st.session_state:
        render(st.session_state["analysis_questions"])

with tabs[6]:
    st.header("⑦ Exam Protocol")
    analyze_button("Generate detailed exam protocol", "exam_protocol")
    if "analysis_exam_protocol" in st.session_state:
        render(st.session_state["analysis_exam_protocol"])

with tabs[7]:
    st.header("⑧ Dx + Diagnostic Map")
    analyze_button("Analyze differential + full diagnostic map", "dx_workup")
    if "analysis_dx_workup" in st.session_state:
        render(st.session_state["analysis_dx_workup"])

with tabs[8]:
    st.header("⑨ Results Review")
    analyze_button("Interpret results and update plan", "results")
    if "analysis_results" in st.session_state:
        render(st.session_state["analysis_results"])

with tabs[9]:
    st.header("⑩ Full Level-5 Review")
    analyze_button("Full Level-5 review", "full_review")
    if "analysis_full_review" in st.session_state:
        render(st.session_state["analysis_full_review"])

with tabs[10]:
    st.header("⑪ Report / Search")
    if st.session_state.get("evidence_text"):
        with st.expander("Raw automatic evidence research"):
            st.text(st.session_state["evidence_text"][:25000])

    if "last_analysis" in st.session_state:
        md = report_markdown(st.session_state["last_analysis"])
        st.download_button("Download Markdown report", data=md.encode("utf-8"), file_name=f"level5_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md", mime="text/markdown")
        st.download_button("Download JSON", data=st.session_state["last_analysis"].model_dump_json(indent=2).encode("utf-8"), file_name=f"level5_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json")

    search = st.text_input("Search saved reports")
    reports = sorted(DATA_DIR.glob("level5_report_*.md"), reverse=True)
    for r in reports[:50]:
        txt = r.read_text(encoding="utf-8")
        if search and search.lower() not in txt.lower() and search.lower() not in r.name.lower():
            continue
        with st.expander(r.name):
            st.text(txt[:7000])

    with st.expander("Preview context sent to AI"):
        st.text(context_now())
