import base64
import json
from datetime import datetime
from pathlib import Path
from typing import List, Literal

import streamlit as st
from openai import OpenAI
from pydantic import BaseModel, Field


# =========================================================
# MedAssist Neuro Ultra v4.1
# UI of Ultra v3 + strict neurology logic of v4
# =========================================================

st.set_page_config(
    page_title="MedAssist Neuro Ultra v4.1",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

DEFAULT_MODEL = "gpt-4o-mini"


# =========================================================
# Styling
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
.muted {font-size:.9rem;color:#777;}
.red-box {border-left:5px solid #ff4b4b;padding:11px 14px;border-radius:9px;background:rgba(255,75,75,.11);margin-bottom:8px;}
.blue-box {border-left:5px solid #2e86de;padding:11px 14px;border-radius:9px;background:rgba(46,134,222,.10);margin-bottom:8px;}
.purple-box {border-left:5px solid #7d5fff;padding:11px 14px;border-radius:9px;background:rgba(125,95,255,.10);margin-bottom:8px;}
.orange-box {border-left:5px solid #fd9644;padding:11px 14px;border-radius:9px;background:rgba(253,150,68,.10);margin-bottom:8px;}
.green-box {border-left:5px solid #00b894;padding:11px 14px;border-radius:9px;background:rgba(0,184,148,.10);margin-bottom:8px;}
.gray-box {border-left:5px solid #95a5a6;padding:11px 14px;border-radius:9px;background:rgba(149,165,166,.10);margin-bottom:8px;}
</style>
""", unsafe_allow_html=True)


# =========================================================
# Structured output
# =========================================================

class ReferenceAnchor(BaseModel):
    guideline_or_source: str
    principle_used: str
    how_it_applies_here: str


class MissingCriticalData(BaseModel):
    item: str
    why_required: str
    blocks_safe_decision: bool


class MandatoryQuestion(BaseModel):
    question: str
    category: Literal[
        "headache_red_flags", "onset_timing", "focal_neurology", "raised_icp",
        "csf_leak", "seizure_syncope", "vertigo", "infection", "trauma",
        "vascular_risk", "medication_safety", "sleep_psych", "general"
    ]
    why_ask: str
    how_answer_changes_decision: str
    priority: Literal["must_ask_now", "important", "routine"]


class RedFlag(BaseModel):
    flag: str
    status_from_data: Literal["present", "absent", "unknown"]
    why_dangerous: str
    action_threshold: str
    urgency: Literal["emergency", "same_day", "soon", "routine"]


class ExamChecklistItem(BaseModel):
    exam_item: str
    how_to_do_it_briefly: str
    what_to_record: List[str] = Field(default_factory=list)
    abnormal_meaning: str
    normal_does_not_exclude: str
    priority: Literal["must_do_now", "important", "routine"]


class ExamInterpretationItem(BaseModel):
    entered_finding: str
    interpretation: str
    neuro_localization_if_relevant: str
    supports: List[str] = Field(default_factory=list)
    argues_against_but_does_not_exclude: List[str] = Field(default_factory=list)
    next_exam_or_test_triggered: str


class ImagingDecision(BaseModel):
    imaging_needed_now: Literal["yes", "no", "conditional", "unclear_need_more_data"]
    imaging_type: str
    indication_or_reason: str
    why_not_needed_if_no: str
    urgency: Literal["emergency", "same_day", "soon", "routine", "not_indicated_now"]
    protocol_notes: str


class WorkupItem(BaseModel):
    test_or_action: str
    type: Literal["lab", "imaging", "EEG", "LP", "EMG_NCS", "ECG", "bedside", "referral", "other"]
    priority: Literal["urgent", "important", "routine", "not_now"]
    why_needed: str
    what_result_changes: str


class DifferentialItem(BaseModel):
    diagnosis_or_category: str
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


class SOAP(BaseModel):
    subjective: str
    objective: str
    assessment: str
    plan: str


class NeuroUltraAnalysis(BaseModel):
    stage: str
    case_summary: str
    neuro_problem_representation: str
    triage_level: Literal["emergency", "same_day", "soon", "routine", "unclear"]
    triage_reason: str
    missing_critical_data: List[MissingCriticalData] = Field(default_factory=list)
    mandatory_questions: List[MandatoryQuestion] = Field(default_factory=list)
    neuro_red_flags: List[RedFlag] = Field(default_factory=list)
    exam_checklist: List[ExamChecklistItem] = Field(default_factory=list)
    exam_interpretation: List[ExamInterpretationItem] = Field(default_factory=list)
    imaging_decision: List[ImagingDecision] = Field(default_factory=list)
    recommended_workup: List[WorkupItem] = Field(default_factory=list)
    differential_diagnosis: List[DifferentialItem] = Field(default_factory=list)
    interpreted_results: List[ResultInterpretation] = Field(default_factory=list)
    medication_safety: List[MedicationSafety] = Field(default_factory=list)
    treatment_support_after_results: List[TreatmentSupport] = Field(default_factory=list)
    trusted_reference_anchors: List[ReferenceAnchor] = Field(default_factory=list)
    strict_quality_check: str
    what_to_do_now: str
    what_to_enter_next: str
    referral_or_er_threshold: str
    patient_explanation_arabic: str
    soap_note: SOAP
    limitations: List[str] = Field(default_factory=list)


# =========================================================
# Prompts
# =========================================================

TRUSTED_RULES = """
Trusted neurology reference anchors to apply inside reasoning:
1. American Headache Society neuroimaging guidance:
   Migraine-compatible headache + normal neurologic exam + no atypical features/red flags -> neuroimaging is generally not necessary.
2. SNNOOP10 headache red flags:
   systemic symptoms/fever, neoplasm, neurologic deficit/altered mental status, sudden thunderclap,
   older age new headache, pattern change/recent onset, positional headache, Valsalva/exertion/sex trigger,
   papilledema, progressive/atypical, pregnancy/puerperium, painful eye with autonomic features,
   post-traumatic onset, immunosuppression, medication overuse/new drug.
3. First unprovoked seizure:
   separate seizure from syncope; check provoking factors; EEG and brain imaging are often considered when first unprovoked seizure is suspected.
4. Acute dizziness/vertigo:
   use timing/triggers; central signs include diplopia, dysarthria, limb ataxia, severe gait ataxia, direction-changing nystagmus.
   HINTS only for trained clinicians in acute vestibular syndrome.
5. Stroke/TIA:
   sudden focal deficit, speech disturbance, facial droop, arm/leg weakness, visual field loss, ataxia -> emergency/time-sensitive pathway.
6. Raised ICP/CSF leak:
   ask papilledema, transient visual obscurations, pulsatile tinnitus, Valsalva/positional headache, clear rhinorrhea/otorrhea, trauma/surgery.
   Consider neuro-ophthalmology/ENT skull base when consistent.
"""


def get_api_key():
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return None


def system_prompt():
    return f"""
You are MedAssist Neuro Ultra v4.1, a neurology-first clinical decision support assistant for a licensed clinician.

Core rules:
- Never provide final diagnosis.
- Never provide medication dosing.
- Do not over-order CT/MRI.
- Do not under-triage red flags.
- Before diagnosis, identify mandatory missing questions and exam findings.
- Interpret clinical exam clearly:
  1) what the finding means
  2) neuro-localization if relevant
  3) what it supports
  4) what it argues against but does not exclude
  5) what exam/test it triggers next
- Medication safety must appear whenever medications/allergies/risks are entered.
- Use clean Arabic medical language with useful English clinical terms.
- Avoid mistranslations. Use "SOAP" not "صابون".
- Return exactly according to schema.

{TRUSTED_RULES}
"""


def stage_instruction(stage, neuro_focus):
    if stage == "questions":
        return f"""
STAGE: 1 — Strict neuro questions and medication safety.
Goal: generate mandatory patient questions, red flags, missing critical data, and medication safety.
Neurology focus: {neuro_focus}
"""
    if stage == "exam":
        return f"""
STAGE: 2 — Neuro exam advisor.
Goal: give a clear neurological exam checklist and explain how each finding is interpreted.
Neurology focus: {neuro_focus}
"""
    if stage == "preliminary":
        return f"""
STAGE: 3 — Preliminary neuro differential and workup.
Goal: analyze intake + entered exam findings, interpret exam, decide imaging strictly, propose workup.
Neurology focus: {neuro_focus}
"""
    if stage == "results":
        return f"""
STAGE: 4 — Results and imaging review.
Goal: interpret labs/MRI/CT/Xray/EEG/ECG/reports/images, update differential, medication safety and treatment support without dosing.
Neurology focus: {neuro_focus}
"""
    return f"""
STAGE: 5 — Full strict neuro review.
Goal: use all data to produce full neurology-focused clinical decision support report.
Neurology focus: {neuro_focus}
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

NEURO COMPLAINT
Main complaint: {data.get("complaint")}
Onset/timing: {data.get("onset")}
Course/progression: {data.get("course")}
Quality/sensation: {data.get("quality")}
Severity/function impact: {data.get("severity")}
Associated symptoms: {data.get("associated")}

NEURO RED-FLAG SCREEN
Focal neuro symptoms: {data.get("focal")}
Fever/neck stiffness/systemic symptoms: {data.get("systemic")}
Trauma/anticoagulants/cancer/immunosuppression: {data.get("risk_red")}
Seizure/syncope features: {data.get("seizure_syncope")}
Dizziness/vertigo features: {data.get("vertigo")}
Raised ICP/CSF leak features: {data.get("icp_csf")}
Prior similar episodes/headache history: {data.get("previous")}

MEDICATION SAFETY
Current medications: {data.get("meds")}
Allergies: {data.get("allergies")}
Renal/liver/pregnancy/bleeding risks: {data.get("safety")}

VITALS
{data.get("vitals")}

NEURO EXAM FINDINGS ENTERED BY DOCTOR
Mental status: {data.get("mental")}
Cranial nerves including pupils/EOM/visual fields/fundoscopy: {data.get("cn")}
Motor/tone/reflexes: {data.get("motor")}
Sensory: {data.get("sensory")}
Coordination/gait: {data.get("coordination")}
Fundoscopy/meningism/ENT/spine/other: {data.get("special_exam")}

RESULTS
Labs: {data.get("labs")}
MRI/CT/X-ray report text: {data.get("imaging")}
EEG/EMG/ECG/Other report: {data.get("other")}

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
    return {"type": "input_file", "filename": name, "file_data": f"data:{mime};base64,{b64}"}


def run_ai(stage, focus, context, files, model):
    key = get_api_key()
    if not key:
        st.error("OPENAI_API_KEY غير موجود في Streamlit Secrets.")
        st.stop()

    client = OpenAI(api_key=key)
    content = [{"type": "input_text", "text": stage_instruction(stage, focus) + "\n\nCASE CONTEXT:\n" + context}]
    for f in files or []:
        content.append(file_item(f))

    response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": content},
        ],
        text_format=NeuroUltraAnalysis,
    )
    return response.output_parsed


def report_markdown(a):
    lines = []
    lines.append("# MedAssist Neuro Ultra v4.1 Report\n")
    lines.append(f"## Stage\n{a.stage}\n")
    lines.append(f"## Summary\n{a.case_summary}\n")
    lines.append(f"## Neuro problem representation\n{a.neuro_problem_representation}\n")
    lines.append(f"## Triage\n**{a.triage_level}** — {a.triage_reason}\n")

    def section(title, items, formatter):
        lines.append(f"\n## {title}")
        if not items:
            lines.append("—")
        for it in items:
            lines.append(formatter(it))

    section("Missing Critical Data", a.missing_critical_data, lambda x: f"- **{x.item}** — {x.why_required}. Blocks safe decision: {x.blocks_safe_decision}")
    section("Mandatory Questions", a.mandatory_questions, lambda q: f"- **{q.priority} / {q.category}:** {q.question} — {q.why_ask}. Decision impact: {q.how_answer_changes_decision}")
    section("Red Flags", a.neuro_red_flags, lambda r: f"- **{r.urgency} / {r.status_from_data}: {r.flag}** — {r.why_dangerous}. Threshold/action: {r.action_threshold}")
    section("Exam Checklist", a.exam_checklist, lambda e: f"- **{e.priority}: {e.exam_item}** — How: {e.how_to_do_it_briefly}. Record: {', '.join(e.what_to_record)}. Abnormal meaning: {e.abnormal_meaning}. Normal does not exclude: {e.normal_does_not_exclude}")
    section("Exam Interpretation", a.exam_interpretation, lambda e: f"- **{e.entered_finding}** — {e.interpretation}. Localization: {e.neuro_localization_if_relevant}. Supports: {', '.join(e.supports)}. Against but not exclude: {', '.join(e.argues_against_but_does_not_exclude)}. Next: {e.next_exam_or_test_triggered}")
    section("Imaging Decision", a.imaging_decision, lambda im: f"- **{im.imaging_needed_now} / {im.urgency}: {im.imaging_type}** — {im.indication_or_reason}. Why not if no: {im.why_not_needed_if_no}. Protocol: {im.protocol_notes}")
    section("Workup", a.recommended_workup, lambda w: f"- **{w.priority} / {w.type}: {w.test_or_action}** — {w.why_needed}. Changes: {w.what_result_changes}")
    section("Differential", a.differential_diagnosis, lambda d: f"- **{d.probability} / {d.urgency}: {d.diagnosis_or_category}** — Support: {', '.join(d.supporting_features)}. Against: {', '.join(d.features_against)}. Missing: {', '.join(d.missing_data_needed)}. Step: {d.confirm_or_exclude_step}")
    section("Interpreted Results", a.interpreted_results, lambda x: f"- **{x.finding}** from {x.source}: {x.interpretation}. Effect: {x.effect_on_differential}. Confidence: {x.confidence}")
    section("Medication Safety", a.medication_safety, lambda m: f"- **{m.severity}: {m.medication_or_issue}** — {m.concern}. Check: {', '.join(m.check_before_treatment)}. Avoid/caution: {m.avoid_or_caution}. Safer: {m.safer_consideration}")
    section("Treatment Support", a.treatment_support_after_results, lambda t: f"- **{t.clinical_goal}: {t.possible_class_or_option}** — Consider: {t.when_to_consider}. Check: {', '.join(t.must_check_before)}. Avoid: {', '.join(t.avoid_if)}. Monitoring: {t.monitoring}. Note: {t.note_no_dosing}")
    section("Trusted Reference Anchors", a.trusted_reference_anchors, lambda r: f"- **{r.guideline_or_source}:** {r.principle_used} → {r.how_it_applies_here}")

    lines.append(f"\n## Strict Quality Check\n{a.strict_quality_check}")
    lines.append(f"\n## What to do now\n{a.what_to_do_now}")
    lines.append(f"\n## What to enter next\n{a.what_to_enter_next}")
    lines.append(f"\n## Referral / ER threshold\n{a.referral_or_er_threshold}")
    lines.append(f"\n## Patient explanation Arabic\n{a.patient_explanation_arabic}")
    lines.append("\n## SOAP")
    lines.append(f"**S:** {a.soap_note.subjective}")
    lines.append(f"**O:** {a.soap_note.objective}")
    lines.append(f"**A:** {a.soap_note.assessment}")
    lines.append(f"**P:** {a.soap_note.plan}")
    lines.append("\n## Limitations")
    for l in a.limitations:
        lines.append(f"- {l}")
    return "\n".join(lines)


def save_report(context, a):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = DATA_DIR / f"neuro_ultra_{stamp}.json"
    md_path = DATA_DIR / f"neuro_ultra_{stamp}.md"
    json_path.write_text(json.dumps({
        "created": stamp,
        "context": context,
        "analysis": a.model_dump()
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    md = report_markdown(a)
    md_path.write_text(md, encoding="utf-8")
    return md


def render(a):
    st.subheader("Clinical Dashboard")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Triage", a.triage_level)
    d2.metric("Missing critical", len(a.missing_critical_data))
    d3.metric("Red flags", len(a.neuro_red_flags))
    d4.metric("Differential", len(a.differential_diagnosis))

    st.markdown(f"""
    <div class='workflow-card'>
    <div class='title-big'>🧠 Neuro Summary</div>
    <b>{a.case_summary}</b><br><br>
    <b>Problem representation:</b><br>{a.neuro_problem_representation}
    </div>
    """, unsafe_allow_html=True)

    if a.triage_level in ["emergency", "same_day"]:
        st.error(f"{a.triage_level}: {a.triage_reason}")
    elif a.triage_level == "soon":
        st.warning(f"{a.triage_level}: {a.triage_reason}")
    else:
        st.info(f"{a.triage_level}: {a.triage_reason}")

    if a.missing_critical_data:
        st.subheader("Missing Critical Data")
        for x in a.missing_critical_data:
            st.warning(f"{x.item}: {x.why_required} | blocks safe decision: {x.blocks_safe_decision}")

    if a.mandatory_questions:
        st.subheader("Mandatory Neuro Questions")
        for q in a.mandatory_questions:
            with st.expander(f"{q.priority} — {q.question}"):
                st.write("**Category:**", q.category)
                st.write("**Why ask:**", q.why_ask)
                st.write("**Decision impact:**", q.how_answer_changes_decision)

    if a.neuro_red_flags:
        st.subheader("Neurology Red Flags")
        for r in a.neuro_red_flags:
            st.markdown(f"""
            <div class='red-box'>
            <b>{r.urgency} / {r.status_from_data}: {r.flag}</b><br>
            {r.why_dangerous}<br>
            <b>Threshold/action:</b> {r.action_threshold}
            </div>
            """, unsafe_allow_html=True)

    if a.exam_checklist:
        st.subheader("Clinical Exam Advisor — ماذا أفحص؟ وكيف أفسر؟")
        for e in a.exam_checklist:
            st.markdown(f"""
            <div class='blue-box'>
            <b>{e.priority}: {e.exam_item}</b><br>
            <b>How:</b> {e.how_to_do_it_briefly}<br>
            <b>Record:</b> {", ".join(e.what_to_record) if e.what_to_record else "—"}<br>
            <b>Abnormal meaning:</b> {e.abnormal_meaning}<br>
            <b>Normal does not exclude:</b> {e.normal_does_not_exclude}
            </div>
            """, unsafe_allow_html=True)

    if a.exam_interpretation:
        st.subheader("Interpretation of Entered Exam Findings")
        for e in a.exam_interpretation:
            st.markdown(f"""
            <div class='green-box'>
            <b>{e.entered_finding}</b><br>
            <b>Interpretation:</b> {e.interpretation}<br>
            <b>Localization:</b> {e.neuro_localization_if_relevant}<br>
            <b>Supports:</b> {", ".join(e.supports) if e.supports else "—"}<br>
            <b>Argues against but does not exclude:</b> {", ".join(e.argues_against_but_does_not_exclude) if e.argues_against_but_does_not_exclude else "—"}<br>
            <b>Next triggered:</b> {e.next_exam_or_test_triggered}
            </div>
            """, unsafe_allow_html=True)

    if a.imaging_decision:
        st.subheader("Strict Imaging Decision")
        for im in a.imaging_decision:
            cls = "blue-box" if im.imaging_needed_now in ["yes", "conditional"] else "green-box"
            st.markdown(f"""
            <div class='{cls}'>
            <b>{im.imaging_needed_now} / {im.urgency}: {im.imaging_type}</b><br>
            <b>Reason:</b> {im.indication_or_reason}<br>
            <b>Why not if no:</b> {im.why_not_needed_if_no}<br>
            <b>Protocol:</b> {im.protocol_notes}
            </div>
            """, unsafe_allow_html=True)

    if a.recommended_workup:
        st.subheader("Recommended Workup")
        for w in a.recommended_workup:
            st.markdown(f"""
            <div class='blue-box'>
            <b>{w.priority} / {w.type}: {w.test_or_action}</b><br>
            {w.why_needed}<br>
            <b>What changes:</b> {w.what_result_changes}
            </div>
            """, unsafe_allow_html=True)

    if a.differential_diagnosis:
        st.subheader("Differential Diagnosis")
        for d in a.differential_diagnosis:
            st.markdown(f"""
            <div class='purple-box'>
            <b>{d.probability} / {d.urgency}: {d.diagnosis_or_category}</b><br>
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
            <b>Check before treatment:</b> {", ".join(m.check_before_treatment) if m.check_before_treatment else "—"}<br>
            <b>Avoid/caution:</b> {m.avoid_or_caution}<br>
            <b>Safer consideration:</b> {m.safer_consideration}
            </div>
            """, unsafe_allow_html=True)

    if a.treatment_support_after_results:
        st.subheader("Treatment Support After Results — no dosing")
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

    if a.trusted_reference_anchors:
        st.subheader("Trusted Reference Anchors Used")
        for ref in a.trusted_reference_anchors:
            st.info(f"**{ref.guideline_or_source}:** {ref.principle_used} → {ref.how_it_applies_here}")

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
    st.title("🧠 Neuro Ultra v4.1")
    model = st.text_input("OpenAI model", value=DEFAULT_MODEL)
    focus = st.selectbox("Neurology focus", [
        "Headache / Migraine / Raised ICP",
        "Seizure / Syncope",
        "Stroke / TIA / Weakness",
        "Dizziness / Vertigo",
        "Neuropathy / Numbness",
        "Neck/back pain with neuro symptoms",
        "CSF leak / ENT-neuro overlap",
        "Movement disorder",
        "Cognitive / psychiatric-neuro overlap",
        "General Neurology"
    ])
    st.warning("Clinical decision support only. القرار النهائي للطبيب.")
    st.caption("استعمل Patient ID بدل الاسم.")


# =========================================================
# Main UI
# =========================================================

st.title("MedAssist Neuro Ultra v4.1")
st.caption("شكل Ultra v3 + صرامة Neuro Strict v4 + خانة بحث/سؤال سريري واضحة")

top = st.columns(6)
top[0].metric("1", "Intake")
top[1].metric("2", "Questions")
top[2].metric("3", "Exam")
top[3].metric("4", "Dx/Workup")
top[4].metric("5", "Results")
top[5].metric("6", "Report")

tabs = st.tabs([
    "① Neuro Intake",
    "② Strict Questions",
    "③ Clinical Exam",
    "④ Preliminary Dx & Workup",
    "⑤ Results / Imaging",
    "⑥ Full Review / Medication",
    "⑦ Report / Search"
])

with tabs[0]:
    st.header("① Neuro Intake")
    st.markdown("<div class='workflow-card'>أدخل القصة العصبية والأدوية، واكتب في خانة البحث ماذا تريد من النظام بالضبط.</div>", unsafe_allow_html=True)

    clinical_search = st.text_input("🔎 Clinical search / سؤال سريري سريع", placeholder="مثال: هل يحتاج CT؟ ما أسئلة red flags للصداع؟ ماذا أفحص سريريًا؟")

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
    complaint = st.text_area("Main neuro complaint / الشكوى العصبية الأساسية", height=90)
    onset = st.text_area("Onset/timing / البداية والتوقيت", height=80)
    course = st.text_area("Course/progression / التطور", height=80)
    quality = st.text_area("Quality/sensation / طبيعة الإحساس", height=80)
    severity = st.text_area("Severity/function impact / الشدة والتأثير", height=80)
    associated = st.text_area("Associated symptoms / أعراض مرافقة", height=110)

    st.subheader("Neuro red-flag screen")
    focal = st.text_area("Weakness/numbness/speech/vision/gait", height=80)
    systemic = st.text_area("Fever/neck stiffness/systemic symptoms", height=80)
    risk_red = st.text_area("Trauma/anticoagulants/cancer/immunosuppression", height=80)
    seizure_syncope = st.text_area("Seizure/syncope features", height=80)
    vertigo = st.text_area("Dizziness/vertigo features", height=80)
    icp_csf = st.text_area("Raised ICP / CSF leak features", height=80)
    previous = st.text_area("Previous similar episodes/headache history", height=80)

    st.subheader("Medication safety")
    meds = st.text_area("Current medications", height=90)
    allergies = st.text_area("Allergies", height=70)
    safety = st.text_area("Renal/liver/pregnancy/bleeding risks", height=70)
    vitals = st.text_area("Vitals", height=80)
    doctor_question = st.text_area("Additional doctor question", height=80)


with tabs[2]:
    st.header("③ Clinical Exam")
    st.markdown("<div class='workflow-card'>اكتب هنا ما وجدته بالفحص. سيقوم النظام بتفسير الفحص سريريًا وعصبيًا.</div>", unsafe_allow_html=True)
    mental = st.text_area("Mental status", height=80)
    cn = st.text_area("Cranial nerves including pupils/EOM/visual fields/fundoscopy", height=100)
    motor = st.text_area("Motor / tone / reflexes", height=100)
    sensory = st.text_area("Sensory", height=80)
    coordination = st.text_area("Coordination / gait", height=90)
    special_exam = st.text_area("Meningism / ENT / spine / other", height=100)


with tabs[4]:
    st.header("⑤ Results / Imaging")
    labs = st.text_area("Labs", height=120)
    imaging = st.text_area("MRI/CT/X-ray report text", height=150)
    other = st.text_area("EEG/EMG/ECG/Other report", height=120)
    uploaded_files = st.file_uploader("Upload PDF/images/reports", type=["pdf", "png", "jpg", "jpeg", "webp", "txt", "csv", "xlsx", "docx"], accept_multiple_files=True)
    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} file(s)")


def context_now():
    return make_context({
        "clinical_search": clinical_search,
        "patient_id": patient_id, "age": age, "sex": sex, "pregnancy": pregnancy,
        "setting": setting, "complaint": complaint, "onset": onset, "course": course,
        "quality": quality, "severity": severity, "associated": associated,
        "focal": focal, "systemic": systemic, "risk_red": risk_red,
        "seizure_syncope": seizure_syncope, "vertigo": vertigo, "icp_csf": icp_csf,
        "previous": previous, "meds": meds, "allergies": allergies, "safety": safety,
        "vitals": vitals, "mental": mental, "cn": cn, "motor": motor,
        "sensory": sensory, "coordination": coordination, "special_exam": special_exam,
        "labs": labs, "imaging": imaging, "other": other, "doctor_question": doctor_question,
    })


def analyze_button(label, stage):
    if st.button(label, type="primary", use_container_width=True):
        with st.spinner("AI يحلل المرحلة..."):
            try:
                ctx = context_now()
                a = run_ai(stage, focus, ctx, uploaded_files if 'uploaded_files' in globals() else [], model)
                st.session_state[f"analysis_{stage}"] = a
                st.session_state["last_analysis"] = a
                st.session_state["last_context"] = ctx
                st.session_state["last_md"] = save_report(ctx, a)
                st.success("تم التحليل")
            except Exception as e:
                st.error("حدث خطأ أثناء التحليل")
                st.exception(e)


with tabs[1]:
    st.header("② Strict Questions")
    st.markdown("<div class='workflow-card'>هذه المرحلة تعطي أسئلة إلزامية للمريض، red flags، واحتياطات دوائية قبل أي تشخيص.</div>", unsafe_allow_html=True)
    analyze_button("Generate strict neuro questions", "questions")
    if "analysis_questions" in st.session_state:
        render(st.session_state["analysis_questions"])


with tabs[2]:
    st.divider()
    analyze_button("Suggest and interpret neuro exam", "exam")
    if "analysis_exam" in st.session_state:
        render(st.session_state["analysis_exam"])


with tabs[3]:
    st.header("④ Preliminary Dx & Workup")
    st.markdown("<div class='workflow-card'>بعد إدخال القصة والفحص، يعطي differential صارم، قرار تصوير مضبوط، وفحوصات مقترحة.</div>", unsafe_allow_html=True)
    analyze_button("Analyze preliminary neuro Dx/workup", "preliminary")
    if "analysis_preliminary" in st.session_state:
        render(st.session_state["analysis_preliminary"])


with tabs[5]:
    st.header("⑥ Full Review / Medication")
    st.markdown("<div class='workflow-card'>بعد النتائج والصور، يحلل ويعطي medication support بدون جرعات، مع الموانع والفحوصات قبل العلاج.</div>", unsafe_allow_html=True)
    analyze_button("Analyze results and medication safety", "results")
    if "analysis_results" in st.session_state:
        render(st.session_state["analysis_results"])

    st.divider()
    analyze_button("Full strict neuro review", "full_review")
    if "analysis_full_review" in st.session_state:
        render(st.session_state["analysis_full_review"])


with tabs[6]:
    st.header("⑦ Report / Search")
    search_history = st.text_input("🔎 Search saved reports", placeholder="headache, seizure, MRI, patient id...")
    reports = sorted(DATA_DIR.glob("neuro_ultra_*.md"), reverse=True)

    if "last_analysis" in st.session_state:
        md = report_markdown(st.session_state["last_analysis"])
        st.download_button("Download Markdown report", data=md.encode("utf-8"), file_name=f"neuro_ultra_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md", mime="text/markdown")
        st.download_button("Download JSON", data=st.session_state["last_analysis"].model_dump_json(indent=2).encode("utf-8"), file_name=f"neuro_ultra_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json")

    if reports:
        shown = 0
        for r in reports[:50]:
            txt = r.read_text(encoding="utf-8")
            if search_history and search_history.lower() not in txt.lower() and search_history.lower() not in r.name.lower():
                continue
            shown += 1
            with st.expander(r.name):
                st.text(txt[:6000])
        if shown == 0:
            st.info("لا توجد نتائج مطابقة للبحث.")
    else:
        st.info("لا توجد تقارير محفوظة بعد.")

    with st.expander("Preview full context sent to AI"):
        st.text(context_now())
