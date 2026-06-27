import base64
import json
from datetime import datetime
from pathlib import Path
from typing import List, Literal

import streamlit as st
from openai import OpenAI
from pydantic import BaseModel, Field


# =========================================================
# MedAssist Neuro-General AutoEvidence Guard v4.7.3.1
# Adds: staged questions before/after every clinical step
# =========================================================

st.set_page_config(
    page_title="MedAssist Neuro-General AutoEvidence Guard v4.7.3.1",
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
.question-box {border-left:5px solid #9b59b6;padding:11px 14px;border-radius:9px;background:rgba(155,89,182,.10);margin-bottom:8px;}
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



class EvidenceVerification(BaseModel):
    clinical_question: str
    recommendation_or_claim: str
    evidence_source_type: Literal[
        "automatic_web_search", "uploaded_guideline_or_reference", "built_in_guideline_framework",
        "clinical_reasoning_only", "not_verified_live", "needs_specialist_review"
    ]
    reference_name_or_note: str
    source_title: str = ""
    source_organization: str = ""
    source_year_or_date: str = ""
    source_url_or_citation: str = ""
    exact_evidence_point: str = ""
    evidence_strength: Literal["strong", "moderate", "low", "uncertain"]
    verification_status: Literal[
        "verified_from_web_search", "verified_from_uploaded_material", "framework_based_not_live_checked",
        "not_live_verified", "insufficient_evidence", "needs_manual_reference_check"
    ]
    how_it_changes_recommendation: str
    caution: str



class MissingCriticalData(BaseModel):
    item: str
    why_required: str
    blocks_safe_decision: bool


class ChecklistItem(BaseModel):
    checklist_name: str
    item: str
    status_from_case: Literal["present", "absent", "unknown", "not_applicable"]
    action_if_present_or_unknown: str


class StageQuestion(BaseModel):
    question: str
    stage: Literal[
        "before_exam", "after_exam", "before_labs_imaging", "after_labs_imaging",
        "before_medication", "after_medication_safety", "general_followup"
    ]
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


class ExamProtocolStep(BaseModel):
    exam_section: Literal[
        "general_vitals", "neurologic", "cardiovascular", "respiratory", "abdominal",
        "ENT", "MSK", "skin", "rheumatologic", "psychiatric", "orthostatic_vitals",
        "bedside_tests", "other"
    ]
    exam_item: str
    patient_position_or_setup: str
    how_to_perform_step_by_step: List[str] = Field(default_factory=list)
    what_to_record_exactly: List[str] = Field(default_factory=list)
    normal_expected_finding: str
    abnormal_findings_and_meaning: List[str] = Field(default_factory=list)
    safety_precaution_or_stop_condition: str
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

    questions_before_exam: List[StageQuestion] = Field(default_factory=list)
    questions_after_exam: List[StageQuestion] = Field(default_factory=list)
    questions_before_labs_imaging: List[StageQuestion] = Field(default_factory=list)
    questions_after_labs_imaging: List[StageQuestion] = Field(default_factory=list)
    questions_before_medication: List[StageQuestion] = Field(default_factory=list)
    questions_after_medication_safety: List[StageQuestion] = Field(default_factory=list)
    followup_questions: List[StageQuestion] = Field(default_factory=list)

    red_flags: List[RedFlag] = Field(default_factory=list)
    detailed_exam_protocol: List[ExamProtocolStep] = Field(default_factory=list)
    exam_interpretation: List[ExamInterpretationItem] = Field(default_factory=list)
    recommended_workup: List[WorkupItem] = Field(default_factory=list)
    imaging_decision: List[ImagingDecision] = Field(default_factory=list)
    differential_diagnosis: List[DifferentialItem] = Field(default_factory=list)
    interpreted_results: List[ResultInterpretation] = Field(default_factory=list)
    medication_safety: List[MedicationSafety] = Field(default_factory=list)
    treatment_support_after_results: List[TreatmentSupport] = Field(default_factory=list)
    follow_up_thresholds: List[FollowUpThreshold] = Field(default_factory=list)
    trusted_reference_anchors: List[ReferenceAnchor] = Field(default_factory=list)
    evidence_verification: List[EvidenceVerification] = Field(default_factory=list)
    quality_control: QualityControl
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

REFERENCE_FRAMEWORK = """
Clinical decision-support framework:
A) Universal Safety Gate:
- unstable vitals, altered mental status, severe respiratory distress, ACS-like chest pain,
  stroke/TIA signs, sepsis/meningitis signs, anaphylaxis, severe dehydration/shock,
  severe trauma, overdose/toxicity, pregnancy emergency, suicidal/homicidal risk -> escalate.
B) Neurology:
- headache/raised ICP/CSF leak, seizure/syncope, stroke/TIA, dizziness/vertigo, neuropathy,
  myelopathy/radiculopathy, movement disorders, cognitive/psychiatric overlap.
C) General medicine mimics:
- hypoglycemia/electrolytes, thyroid, anemia, infection, autoimmune/inflammatory,
  arrhythmia/ACS, PE, medication adverse effect, intoxication/withdrawal, renal/hepatic disease.
D) Headache:
- Use SNNOOP10 red flags and migraine/tension phenotype when headache is relevant.
E) Near-syncope/palpitations HARD RULES:
- If palpitations + near-syncope/presyncope are present: ALWAYS activate Cardiology as same_day.
- Also activate General Medicine, Endocrinology/Metabolic, and Toxicology/Medication Safety when relevant.
- Mandatory before-exam questions must include:
  complete syncope, exertional syncope, chest pain, severe dyspnea, sudden onset/offset of palpitations,
  family history of sudden cardiac death, known structural heart disease, stimulant/caffeine/decongestant use,
  dehydration/vomiting/diarrhea, bleeding/anemia symptoms, glucose/diabetes symptoms.
- Initial workup should usually include ECG, orthostatic BP/HR, capillary glucose, BMP/electrolytes, renal function,
  CBC when anemia/bleeding/fatigue is possible, and medication/substance review.
- TSH is conditional: recurrent/persistent palpitations, weight loss, tremor, heat intolerance, goiter, or unexplained tachycardia.
- Holter/event monitor is conditional: recurrent episodes or high suspicion when initial ECG is non-diagnostic.
- Troponin is conditional only with chest pain, ischemic symptoms, abnormal ECG, or high cardiac risk.
- PE workup is conditional only with significant dyspnea, pleuritic chest pain, hypoxia, tachycardia with PE risks, or clinical suspicion.
- Do NOT rank TIA as medium/high without focal neurologic symptoms such as unilateral weakness/numbness, aphasia/dysarthria,
  vision loss/diplopia, ataxia, or other focal brainstem/cortical signs.
- If no focal neurologic deficit and symptoms are posture-related with palpitations, TIA should be low or cannot_rank.
- ER threshold must include: complete syncope, exertional syncope, chest pain, severe dyspnea, abnormal ECG, hypotension/shock,
  persistent tachyarrhythmia, focal neurologic deficit, new severe headache, seizure, or SpO2 drop.
F) Imaging:
- Avoid routine imaging when not indicated. Do not miss imaging when emergency red flags exist.
G) Medication safety:
- Always consider allergy, pregnancy/postpartum, age, renal/liver disease, anticoagulants/antiplatelets,
  QT risk, sedatives/opioids/benzodiazepines, serotonin syndrome risk, NSAID bleeding/renal risk,
  antiepileptic adverse effects, medication overuse headache.
H) Treatment support:
- Provide classes/options and safety checks only. No doses. No final prescription.
"""


def get_api_key():
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return None


def system_prompt(strictness: str):
    return f"""
You are MedAssist Neuro-General AutoEvidence Guard v4.7.3.1.

Identity:
- Neurology-first but not neurology-only.
- You cover neurological diseases and general medicine mimics/causes.
- You are clinical decision support for a clinician, not a final diagnosis authority.

Strictness level: {strictness}

Mandatory rules:
1. Start with Safety Gate in every stage.
2. Activate relevant specialty modules.
3. Before giving final recommendations in any stage, perform Evidence Verification:
   - If AUTOMATIC WEB EVIDENCE RESEARCH is present in the context, use it and mark evidence_source_type as automatic_web_search and verification_status as verified_from_web_search when it directly supports the recommendation.
   - Prefer recent guidelines/systematic reviews/official medical society sources when available.
   - Preserve source names/citations in reference_name_or_note.
   - If no uploaded reference is provided, do NOT pretend you searched live references.
   - Use built-in clinical framework only as framework_based_not_live_checked.
   - For uncertain/high-risk recommendations, mark needs_manual_reference_check or needs_specialist_review.
   - Clearly separate guideline-based statements from clinical reasoning.
4. Always produce staged questions when relevant:
   - questions_before_exam
   - questions_after_exam
   - questions_before_labs_imaging
   - questions_after_labs_imaging
   - questions_before_medication
   - questions_after_medication_safety
   - followup_questions
5. These questions must be specific to the current case and the current stage.
6. If the stage is "exam_protocol", tell the doctor HOW to perform detailed clinical exam step by step before exam results.
7. If exam findings are entered, interpret them clearly.
8. If results are entered, generate questions_after_labs_imaging that depend on those results.
9. If treatment/medication support is considered, generate questions_before_medication and medication safety questions.
10. Imaging must be justified. Avoid over-testing and under-testing.
11. For palpitations + near-syncope, you MUST show Cardiology in activated_modules.
12. For palpitations + near-syncope without focal neurologic signs, TIA must be low or cannot_rank, not medium/high.
13. Evidence Verification must include source_title, source_organization, source_year_or_date when available,
    source_url_or_citation when available, exact_evidence_point, and limitations/caution.
14. Medication support must never include dosing.
15. Use clean Arabic medical language with useful English medical terms. Use "SOAP".
16. Return exactly according to schema.

{REFERENCE_FRAMEWORK}
"""


def stage_instruction(stage, focus, body_system_focus):
    if stage == "questions":
        return f"""
STAGE 1 — Questions Before Exam.
Generate questions_before_exam strongly, plus missing critical data, red flags, modules, medication safety. For palpitations + near-syncope, include all mandatory cardiac/syncope questions and activate Cardiology.
Also include questions_before_labs_imaging if early workup is likely.
Neurology focus: {focus}
General medicine focus: {body_system_focus}
"""
    if stage == "exam_protocol":
        return f"""
STAGE 2 — Detailed Exam Protocol Before Findings.
Generate detailed_exam_protocol and questions_before_exam. For palpitations + near-syncope, include orthostatic vitals, 12-lead ECG, focused cardiac exam, respiratory exam, and focused neuro exam.
Tell clinician HOW to perform exam before results are entered.
Neurology focus: {focus}
General medicine focus: {body_system_focus}
"""
    if stage == "exam_interpretation":
        return f"""
STAGE 3 — Interpret Entered Exam Findings.
Interpret exam findings.
Generate questions_after_exam and questions_before_labs_imaging based on the exam. If neuro exam is normal and symptoms are posture-related with palpitations, lower TIA ranking and prioritize cardiac/orthostatic/metabolic causes.
Neurology focus: {focus}
General medicine focus: {body_system_focus}
"""
    if stage == "preliminary":
        return f"""
STAGE 4 — Dx and Workup Before Results.
Use history and exam. Generate differential, workup, imaging decision. Enforce: Cardiology activation for palpitations + near-syncope; TIA low/cannot_rank without focal neurologic signs; no routine brain imaging without neuro red flags.
Generate evidence_verification for major diagnostic/workup/imaging recommendations.
Generate questions_before_labs_imaging and questions_before_medication if treatment might be considered.
Neurology focus: {focus}
General medicine focus: {body_system_focus}
"""
    if stage == "results":
        return f"""
STAGE 5 — Results Review.
Interpret labs/imaging/reports.
Generate evidence_verification for major result interpretations and treatment safety recommendations.
Generate questions_after_labs_imaging, questions_before_medication, questions_after_medication_safety.
Neurology focus: {focus}
General medicine focus: {body_system_focus}
"""
    return f"""
STAGE 6 — Full Review.
Use all data and fill all staged question sections where relevant. Include evidence_verification for all major recommendations.
Neurology focus: {focus}
General medicine focus: {body_system_focus}
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

REFERENCE / EVIDENCE MATERIAL ENTERED BY DOCTOR
Reference notes / guideline excerpts: {data.get("reference_notes")}

AUTOMATIC WEB EVIDENCE RESEARCH
{data.get("automatic_evidence_research")}

Evidence instruction:
- If automatic web evidence is present and relevant, cite it in evidence_verification.
- If uploaded/reference material is present, use it.
- If no evidence text is present, do not claim live search; mark recommendations as framework_based_not_live_checked or not_live_verified.

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
Other exam findings: {data.get("other_exam")}

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



AUTHORITATIVE_MEDICAL_DOMAINS = [
    "nice.org.uk", "nhs.uk", "who.int", "cdc.gov", "nih.gov", "ncbi.nlm.nih.gov",
    "aan.com", "heart.org", "stroke.org", "acc.org", "escardio.org",
    "idsociety.org", "thoracic.org", "chestnet.org", "ersnet.org",
    "diabetes.org", "endocrine.org", "thyroid.org",
    "kdigo.org", "acr.org", "rheumatology.org", "eular.org",
    "gastrojournal.org", "acg.gi.org", "aasld.org",
    "ashpublications.org", "hematology.org",
    "aafp.org", "bmj.com", "msdmanuals.com",
    "rcem.ac.uk", "acep.org", "saem.org",
    "acog.org", "rcog.org.uk",
    "epilepsy.com", "ilae.org"
]


def build_evidence_query(stage, focus, body_system_focus, context):
    return f"""
You are performing automatic evidence search for a clinician-facing clinical decision support app.

Task:
Search authoritative medical sources for the current clinical problem before recommendations are generated.
Prioritize guidelines, consensus statements, systematic reviews, and official society/government medical sources.
Do NOT use forums, random blogs, or low-quality sites.
Do NOT claim access to paid sources such as UpToDate unless explicitly available in the search result.
Return concise evidence notes with source names and visible citations.

Clinical focus:
- Stage: {stage}
- Neurology focus: {focus}
- General medicine focus: {body_system_focus}

Patient/case context:
{context}

Return:
1. Key guideline/evidence points relevant to triage, red flags, clinical exam, workup, imaging, and medication safety.
2. For each major recommendation, include:
   - source title
   - organization/publisher
   - year/date if visible
   - source URL/citation if visible
   - exact evidence point in your own words
   - limitation or when it does not apply
3. Specifically search evidence for syncope/presyncope with palpitations, ECG, orthostatic vitals, ambulatory ECG monitoring, and neuroimaging indications.
4. What is uncertain or needs specialist/manual reference check.
5. Sources/citations.
"""


def run_automatic_evidence_search(stage, focus, body_system_focus, context, evidence_model, source_scope, force_search=True):
    key = get_api_key()
    if not key:
        return "Automatic evidence search skipped: OPENAI_API_KEY is missing."

    client = OpenAI(api_key=key)
    query = build_evidence_query(stage, focus, body_system_focus, context)

    if source_scope == "Authoritative medical domains only":
        query += "\n\nIMPORTANT SOURCE RULE: Prefer authoritative medical domains only. Search and use sources from this list when possible: " + ", ".join(AUTHORITATIVE_MEDICAL_DOMAINS) + ". If a source is not from this list, say why it was used."

    # Do NOT pass filters here. Some models reject web_search filters.
    tool = {"type": "web_search"}

    def _call(required: bool, selected_model: str):
        kwargs = dict(
            model=selected_model,
            tools=[tool],
            input=query,
        )
        if required:
            kwargs["tool_choice"] = "required"
        response = client.responses.create(**kwargs)
        evidence_text = getattr(response, "output_text", None)
        if not evidence_text:
            evidence_text = str(response)
        return evidence_text[:20000]

    try:
        return _call(force_search, evidence_model)
    except Exception as e1:
        # Fallback 1: same model, no required tool choice
        try:
            return "Automatic evidence search fallback used: same model without required tool_choice.\n\n" + _call(False, evidence_model)
        except Exception as e2:
            # Fallback 2: gpt-4o-mini, no required tool choice
            try:
                return "Automatic evidence search fallback used: gpt-4o-mini without required tool_choice.\n\n" + _call(False, "gpt-4o-mini")
            except Exception as e3:
                return f"Automatic evidence search failed. Do not mark recommendations as web-verified. Errors: primary={e1}; fallback1={e2}; fallback2={e3}"



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
# Deterministic guardrails
# These are applied AFTER the model output so critical workflow rules
# cannot be ignored by the AI response.
# =========================================================

def _lc(x):
    return (x or "").lower()


def _has_any(text, phrases):
    t = _lc(text)
    return any(p.lower() in t for p in phrases)


def _module_present(a, module_name):
    return any(m.module == module_name for m in a.activated_modules)


def _dedupe_by_key(items, key_fn):
    seen = set()
    out = []
    for item in items:
        key = key_fn(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _append_question_if_missing(question_list, question, stage, domain, why, impact, priority="must_ask_now"):
    q_text = question.lower()
    if not any(q_text[:28] in _lc(q.question) or _lc(q.question)[:28] in q_text for q in question_list):
        question_list.append(StageQuestion(
            question=question,
            stage=stage,
            domain=domain,
            why_ask=why,
            how_answer_changes_decision=impact,
            priority=priority
        ))


def _append_workup_if_missing(a, test_name, type_, priority, why, changes, avoid):
    key = test_name.lower()
    if not any(key in _lc(w.test_or_action) for w in a.recommended_workup):
        a.recommended_workup.append(WorkupItem(
            test_or_action=test_name,
            type=type_,
            priority=priority,
            why_needed=why,
            what_result_changes=changes,
            avoid_if_not_indicated=avoid
        ))


def _append_followup_if_missing(a, situation, action, timeframe):
    key = situation.lower()
    if not any(key in _lc(f.situation) for f in a.follow_up_thresholds):
        a.follow_up_thresholds.append(FollowUpThreshold(
            situation=situation,
            action=action,
            timeframe=timeframe
        ))


def _no_focal_neuro_context(context):
    t = _lc(context)
    absent_markers = [
        "no focal", "no weakness", "no numbness", "no speech", "no vision loss",
        "no diplopia", "no ataxia", "no focal neurological deficits",
        "no focal neurologic", "لا يوجد ضعف", "لا يوجد تنميل", "لا يوجد اضطراب كلام",
        "لا يوجد فقدان نظر", "لا يوجد علامات عصبية بؤرية"
    ]
    positive_focal = [
        "unilateral weakness", "aphasia", "dysarthria", "vision loss", "diplopia",
        "ataxia", "hemiparesis", "focal deficit", "فقدان نظر", "حبسة", "ضعف جهة",
        "ترنح", "ازدواجية الرؤية"
    ]
    return _has_any(t, absent_markers) and not _has_any(t, positive_focal)


def _emergency_trigger_present(context):
    t = _lc(context)
    # Positive emergency markers. Negated phrases below prevent false triggers.
    if "no chest pain" not in t and ("chest pain" in t or "ألم صدر" in t):
        return True
    if "no complete loss" not in t and ("complete loss of consciousness" in t or "complete syncope" in t):
        return True
    if "no seizure" not in t and ("seizure" in t or "tongue bite" in t or "incontinence" in t):
        return True
    if "no focal" not in t and ("focal neurologic" in t or "focal neurological" in t):
        return True
    if "severe dyspnea" in t or "severe shortness of breath" in t or "spo2 8" in t or "spo2 7" in t:
        return True
    if "abnormal ecg" in t or "hypotension/shock" in t or "shock" in t:
        return True
    if "exertional syncope" in t or "syncope during exertion" in t:
        return True
    return False


def apply_deterministic_guardrails(a: GuardAnalysis, context: str) -> GuardAnalysis:
    t = _lc(context)

    palpitations = _has_any(t, ["palpitations", "palpitation", "خفقان"])
    presyncope = _has_any(t, ["near-syncope", "presyncope", "near faint", "near-faint", "may faint", "قرب الإغماء", "قرب الاغماء"])
    posture_related = _has_any(t, ["worse when standing", "worsened by standing", "improves when sitting", "عند الوقوف", "يتحسن عند الجلوس"])
    no_focal = _no_focal_neuro_context(t)

    # Always deduplicate modules/checklists first.
    a.activated_modules = _dedupe_by_key(a.activated_modules, lambda m: m.module)
    a.guideline_checklist = _dedupe_by_key(a.guideline_checklist, lambda g: (g.checklist_name.lower(), g.item.lower()))

    if palpitations and presyncope:
        # Force same-day Cardiology module.
        if not _module_present(a, "Cardiology"):
            a.activated_modules.insert(0, ActivatedModule(
                module="Cardiology",
                urgency="same_day",
                why_activated="Palpitations with near-syncope/presyncope require same-day cardiac rhythm evaluation and ECG."
            ))

        # Add supporting modules if absent.
        if not _module_present(a, "Endocrinology/Metabolic"):
            a.activated_modules.append(ActivatedModule(
                module="Endocrinology/Metabolic",
                urgency="same_day",
                why_activated="Hypoglycemia, electrolyte disturbance, and thyroid disease can mimic dizziness/palpitations."
            ))
        if not _module_present(a, "Toxicology/Medication Safety"):
            a.activated_modules.append(ActivatedModule(
                module="Toxicology/Medication Safety",
                urgency="same_day",
                why_activated="Stimulants, decongestants, caffeine, QT-risk drugs, and medication adverse effects must be reviewed."
            ))

        # Downgrade emergency to same_day if stable and no emergency trigger.
        if a.safety_gate.emergency_now == "yes" and not _emergency_trigger_present(t):
            a.safety_gate.emergency_now = "no"
            a.safety_gate.same_day_assessment_needed = "yes"
            a.safety_gate.reason = (
                "Palpitations with near-syncope require same-day ECG and orthostatic/metabolic evaluation. "
                "Emergency escalation is required if complete syncope, exertional syncope, chest pain, severe dyspnea, abnormal ECG, shock, or focal neurologic deficit appears."
            )
            a.triage_level = "same_day"
            a.triage_reason = a.safety_gate.reason

        # Ensure must-not-miss list has cardiac priority.
        for condition in ["Serious arrhythmia", "Orthostatic hypotension", "Hypoglycemia", "Anemia/bleeding", "Electrolyte disturbance"]:
            if condition not in a.safety_gate.must_not_miss_conditions:
                a.safety_gate.must_not_miss_conditions.append(condition)

        # Strong pre-exam questions.
        q = a.questions_before_exam
        _append_question_if_missing(q, "Did the patient have complete syncope or only near-syncope/presyncope?", "before_exam", "cardiology",
                                    "Complete syncope increases risk and changes triage.", "Complete syncope, especially recurrent or injurious, increases urgency.")
        _append_question_if_missing(q, "Did symptoms occur during exertion or while supine?", "before_exam", "cardiology",
                                    "Exertional or supine syncope is a higher-risk cardiac pattern.", "If yes, escalate cardiac evaluation urgently.")
        _append_question_if_missing(q, "Is there chest pain, severe shortness of breath, or new diaphoresis?", "before_exam", "cardiology",
                                    "These can indicate ACS, arrhythmia, PE, or unstable cardiopulmonary disease.", "If present, emergency pathway and troponin/ACS/PE evaluation may be needed.")
        _append_question_if_missing(q, "Do palpitations start and stop suddenly, and how fast is the pulse during the episode?", "before_exam", "cardiology",
                                    "Sudden onset/offset suggests paroxysmal arrhythmia.", "May trigger rhythm monitoring/Holter/event monitor even if initial ECG is normal.")
        _append_question_if_missing(q, "Is there family history of sudden cardiac death or known structural heart disease?", "before_exam", "cardiology",
                                    "Family/structural risk increases concern for dangerous arrhythmia.", "If positive, lowers threshold for urgent cardiology review.")
        _append_question_if_missing(q, "Any caffeine, stimulants, decongestants, thyroid medications, recreational drugs, or QT-risk drugs?", "before_exam", "medication_safety",
                                    "Substances and medications can provoke tachycardia or arrhythmia.", "May redirect management toward medication/substance review.")
        _append_question_if_missing(q, "Any vomiting, diarrhea, poor intake, dehydration, or recent heat exposure?", "before_exam", "general_medicine",
                                    "Volume depletion can cause orthostatic symptoms.", "If positive, supports orthostatic vitals and fluid status assessment.")
        _append_question_if_missing(q, "Any bleeding, melena, heavy menses, pallor, or progressive fatigue suggesting anemia?", "before_exam", "hematology",
                                    "Anemia/bleeding can cause tachycardia, dizziness, and presyncope.", "If positive, CBC and bleeding evaluation become higher priority.")
        _append_question_if_missing(q, "Was capillary glucose checked during symptoms or is there diabetes/fasting/recent diet change?", "before_exam", "endocrine_metabolic",
                                    "Hypoglycemia can mimic dizziness/palpitations.", "Low glucose changes immediate management.")

        # Workup guardrails.
        _append_workup_if_missing(a, "Capillary blood glucose", "bedside_test", "urgent",
                                  "Hypoglycemia can cause dizziness, palpitations, and presyncope.", "Low glucose requires immediate correction.", "Do not delay if patient is symptomatic.")
        _append_workup_if_missing(a, "CBC", "lab", "important",
                                  "Screens for anemia/bleeding/infection when dizziness with tachycardia is present.", "Anemia or leukocytosis redirects evaluation.", "May be less urgent if no anemia/bleeding/systemic clues and patient is stable.")
        _append_workup_if_missing(a, "TSH if recurrent palpitations or unexplained persistent tachycardia", "lab", "routine",
                                  "Hyperthyroidism can cause palpitations and tachycardia.", "Abnormal TSH changes outpatient endocrine/cardiac follow-up.", "Not a substitute for ECG or urgent evaluation.")
        _append_workup_if_missing(a, "Holter/event monitor if ECG is normal but episodes recur", "monitoring", "important",
                                  "Intermittent arrhythmias may be missed on a single ECG.", "Captured rhythm during symptoms can confirm arrhythmia.", "Not first-line replacement for immediate ECG in current symptoms.")

        # TIA guardrail.
        if no_focal and posture_related:
            for d in a.differential_diagnosis:
                if "transient ischemic" in _lc(d.diagnosis_or_category) or "tia" in _lc(d.diagnosis_or_category):
                    d.probability = "low"
                    if "No focal neurological deficits; symptoms are posture-related with palpitations." not in d.features_against:
                        d.features_against.append("No focal neurological deficits; symptoms are posture-related with palpitations.")
                    d.confirm_or_exclude_step = (
                        "Do not prioritize TIA unless focal neurologic symptoms appear "
                        "(weakness, speech disturbance, vision loss/diplopia, ataxia) or the clinical picture changes."
                    )

        # Referral / ER thresholds.
        a.referral_or_er_threshold = (
            "Immediate ER/cardiology escalation if complete syncope, exertional syncope, chest pain, severe dyspnea, "
            "abnormal ECG, hypotension/shock, persistent tachyarrhythmia, SpO2 drop, seizure, new severe headache, "
            "or any focal neurologic deficit. Otherwise same-day ECG, orthostatic vitals, glucose/electrolytes, and follow-up based on results."
        )
        _append_followup_if_missing(a, "Complete syncope, exertional syncope, chest pain, severe dyspnea, abnormal ECG, hypotension, persistent tachyarrhythmia, SpO2 drop, seizure, or focal neurologic deficit",
                                    "Immediate ER/cardiology escalation.", "immediate_ER")

        # Evidence guardrail for TIA overclaim/mismatched citations.
        for ev in a.evidence_verification:
            ev_all = " ".join([_lc(ev.clinical_question), _lc(ev.recommendation_or_claim), _lc(ev.source_title), _lc(ev.reference_name_or_note), _lc(ev.source_url_or_citation)])
            if ("tia" in ev_all or "transient ischemic" in ev_all) and no_focal:
                ev.evidence_strength = "low"
                ev.verification_status = "needs_manual_reference_check"
                ev.how_it_changes_recommendation = "In this case, absence of focal neurologic signs makes TIA lower priority than cardiac/orthostatic/metabolic causes."
                ev.caution = "Do not treat nonspecific dizziness/presyncope as TIA unless focal neurologic deficits or a true TIA syndrome appears."
            if "pmc3295536" in _lc(ev.source_url_or_citation) and ("tia" in _lc(ev.source_title) or "transient ischemic" in _lc(ev.source_title)):
                ev.verification_status = "needs_manual_reference_check"
                ev.caution = "Potential citation mismatch: verify that the URL actually matches the stated TIA source before relying on it."

    # Final dedupe again.
    a.activated_modules = _dedupe_by_key(a.activated_modules, lambda m: m.module)
    a.guideline_checklist = _dedupe_by_key(a.guideline_checklist, lambda g: (g.checklist_name.lower(), g.item.lower()))
    a.recommended_workup = _dedupe_by_key(a.recommended_workup, lambda w: (w.type, w.test_or_action.lower()))
    a.follow_up_thresholds = _dedupe_by_key(a.follow_up_thresholds, lambda f: (f.timeframe, f.situation.lower()))
    return a


# =========================================================
# Report/render
# =========================================================

def report_markdown(a):
    return "# MedAssist Neuro-General AutoEvidence Guard v4.7.3.1 Report\n\n```json\n" + a.model_dump_json(indent=2) + "\n```"


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


def render_questions(title, questions):
    if questions:
        st.subheader(title)
        for idx, q in enumerate(questions, start=1):
            st.markdown(f"**{idx}. {q.priority} / {q.domain}**")
            st.write("**Question:**", q.question)
            st.write("**Why:**", q.why_ask)
            st.write("**How answer changes decision:**", q.how_answer_changes_decision)
            st.divider()


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

    render_questions("Questions BEFORE clinical exam", a.questions_before_exam)
    render_questions("Questions AFTER exam findings", a.questions_after_exam)
    render_questions("Questions BEFORE labs/imaging", a.questions_before_labs_imaging)
    render_questions("Questions AFTER labs/imaging results", a.questions_after_labs_imaging)
    render_questions("Questions BEFORE medication/treatment", a.questions_before_medication)
    render_questions("Questions AFTER medication safety review", a.questions_after_medication_safety)
    render_questions("Follow-up questions", a.followup_questions)

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

    if a.detailed_exam_protocol:
        st.subheader("Detailed Clinical Exam Protocol — كيف تعمل الفحص؟")
        for e in a.detailed_exam_protocol:
            st.markdown(f"""
            <div class='blue-box'>
            <b>{e.priority} / {e.exam_section}: {e.exam_item}</b><br>
            <b>Patient position/setup:</b> {e.patient_position_or_setup}<br>
            <b>How to perform:</b><br>
            {"<br>".join([str(i+1) + ". " + step for i, step in enumerate(e.how_to_perform_step_by_step)]) if e.how_to_perform_step_by_step else "—"}<br>
            <b>Record exactly:</b> {", ".join(e.what_to_record_exactly) if e.what_to_record_exactly else "—"}<br>
            <b>Normal expected:</b> {e.normal_expected_finding}<br>
            <b>Abnormal meaning:</b> {", ".join(e.abnormal_findings_and_meaning) if e.abnormal_findings_and_meaning else "—"}<br>
            <b>Safety/stop condition:</b> {e.safety_precaution_or_stop_condition}
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

    if a.evidence_verification:
        st.subheader("Evidence Verification — تدقيق المرجع قبل النتيجة")
        for ev in a.evidence_verification:
            box = "green-box" if ev.verification_status == "verified_from_uploaded_material" else ("orange-box" if ev.verification_status in ["framework_based_not_live_checked", "not_live_verified"] else "red-box")
            st.markdown(f"""
            <div class='{box}'>
            <b>Clinical question:</b> {ev.clinical_question}<br>
            <b>Recommendation/claim:</b> {ev.recommendation_or_claim}<br>
            <b>Source type:</b> {ev.evidence_source_type}<br>
            <b>Reference:</b> {ev.reference_name_or_note}<br>
            <b>Source title:</b> {ev.source_title}<br>
            <b>Organization:</b> {ev.source_organization}<br>
            <b>Year/date:</b> {ev.source_year_or_date}<br>
            <b>URL/citation:</b> {ev.source_url_or_citation}<br>
            <b>Evidence point:</b> {ev.exact_evidence_point}<br>
            <b>Evidence strength:</b> {ev.evidence_strength}<br>
            <b>Verification status:</b> {ev.verification_status}<br>
            <b>Impact:</b> {ev.how_it_changes_recommendation}<br>
            <b>Caution:</b> {ev.caution}
            </div>
            """, unsafe_allow_html=True)

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
    st.title("🧠 Neuro-General AutoEvidence Guard v4.7.3")
    model = st.text_input("OpenAI model", value=DEFAULT_MODEL)
    auto_evidence_search = st.checkbox("Automatic Evidence Web Search", value=True)
    evidence_model = st.text_input("Evidence search model", value="gpt-4.1-mini")
    evidence_source_scope = st.selectbox("Evidence source scope", ["Authoritative medical domains only", "Broad web search"], index=0, help="In v4.7.1 this is enforced by prompt, not by API filters, to avoid model compatibility errors.")
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

st.title("MedAssist Neuro-General AutoEvidence Guard v4.7.3.1")
st.caption("الجديد: v4.7.3 — Guardrails برمجية بعد جواب AI: Cardiology إجباري، إزالة التكرار، إصلاح TIA، وإصلاح عرض الأسئلة.")

top = st.columns(10)
top[0].metric("1", "Intake")
top[1].metric("2", "Neuro")
top[2].metric("3", "General")
top[3].metric("4", "Q Before")
top[4].metric("5", "Exam Plan")
top[5].metric("6", "Findings")
top[6].metric("7", "Q After")
top[7].metric("8", "Dx")
top[8].metric("9", "Results")
top[9].metric("10", "Evidence")

tabs = st.tabs([
    "① Intake",
    "② Neuro Screen",
    "③ General Medicine Screen",
    "④ Evidence / Guidelines",
    "⑤ Questions Before Exam",
    "⑥ Exam Protocol: كيف أفحص؟",
    "⑦ Enter Exam Findings",
    "⑧ Questions After Exam",
    "⑨ Dx & Workup",
    "⑩ Results / Imaging",
    "⑪ Full Review / Medication",
    "⑫ Report / Search"
])

uploaded_files = []

with tabs[0]:
    st.header("① Intake")
    clinical_search = st.text_input("🔎 Clinical search / سؤال سريري سريع", placeholder="مثال: ما الأسئلة قبل الفحص وبعد الفحص لهذه الحالة؟")

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
    headache_icp_csf = st.text_area("Headache / raised ICP / CSF leak features", height=100)
    seizure_syncope = st.text_area("Seizure/syncope features", height=90)
    vertigo = st.text_area("Dizziness/vertigo features", height=90)
    weakness_neuropathy = st.text_area("Numbness/weakness/back-neck/radicular symptoms", height=90)
    previous_neuro = st.text_area("Previous neuro episodes/history", height=80)

with tabs[2]:
    st.header("③ General Medicine Screen")
    cardiac = st.text_area("Cardiac symptoms/risk", height=80)
    resp = st.text_area("Respiratory symptoms/risk", height=80)
    infection = st.text_area("Infection/systemic symptoms", height=80)
    endo = st.text_area("Endocrine/metabolic symptoms/risk", height=80)
    gi_renal = st.text_area("GI/renal/hepatic symptoms", height=80)
    rheum_msk_skin = st.text_area("Rheum/MSK/skin symptoms", height=80)
    psych_sleep_substance = st.text_area("Psychiatric/sleep/substance symptoms", height=80)
    risk_red = st.text_area("Trauma/cancer/immunosuppression/other risks", height=80)

with tabs[3]:
    st.header("④ Evidence / Guidelines")
    st.markdown("<div class='workflow-card'>ضع هنا مقتطفات من المراجع أو guideline notes. إذا لم تضع مرجعًا، التطبيق سيقول إن التوصية غير مفحوصة live ولا يدّعي أنه بحث في المراجع.</div>", unsafe_allow_html=True)
    reference_notes = st.text_area(
        "Reference notes / guideline excerpts",
        height=180,
        placeholder="مثال: NICE headache guideline excerpt, AAN seizure guidance, AHA syncope/stroke principle, local hospital protocol..."
    )
    st.info("إذا كان Automatic Evidence Web Search مفعّلًا، سيبحث التطبيق تلقائيًا في المصادر الطبية المفتوحة قبل التحليل. يمكنك أيضًا رفع PDF أو وضع guideline notes هنا.")

with tabs[6]:
    st.header("⑦ Enter Exam Findings")
    st.markdown("<div class='workflow-card'>بعد أن يعطيك التطبيق طريقة الفحص في المرحلة ⑤، اكتب هنا ماذا وجدت أنت بالفحص.</div>", unsafe_allow_html=True)
    general_exam = st.text_area("General appearance/vitals exam", height=80)
    neuro_exam = st.text_area("Neurologic exam", height=130)
    cardio_resp_exam = st.text_area("Cardiovascular/respiratory exam", height=100)
    abd_renal_exam = st.text_area("Abdominal/renal exam", height=80)
    ent_msk_skin_exam = st.text_area("ENT/MSK/skin/rheum exam", height=100)
    psych_exam = st.text_area("Psychiatric/cognitive exam", height=80)
    other_exam = st.text_area("Other exam findings", height=80)

with tabs[9]:
    st.header("⑩ Results / Imaging")
    labs = st.text_area("Labs", height=120)
    imaging = st.text_area("Imaging report text: MRI/CT/X-ray/Ultrasound", height=150)
    other = st.text_area("ECG/EEG/EMG/Echo/Other report", height=120)
    uploaded_files = st.file_uploader("Upload PDF/images/reports", type=["pdf", "png", "jpg", "jpeg", "webp", "txt", "csv", "xlsx", "docx"], accept_multiple_files=True)
    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} file(s)")

automatic_evidence_research = st.session_state.get('automatic_evidence_research', '')

def context_now():
    return make_context({
        "clinical_search": clinical_search, "reference_notes": reference_notes, "automatic_evidence_research": automatic_evidence_research,
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
        "psych_exam": psych_exam, "other_exam": other_exam,
        "labs": labs, "imaging": imaging, "other": other, "doctor_question": doctor_question,
    })

def analyze_button(label, stage):
    if st.button(label, type="primary", use_container_width=True):
        with st.spinner("AI يحلل..."):
            try:
                # First build context without fresh evidence
                ctx_without_fresh_evidence = context_now()

                # Automatic web evidence search before the clinical answer
                if auto_evidence_search:
                    with st.spinner("يبحث أوتوماتيكيًا في المصادر الطبية الموثوقة..."):
                        fresh_evidence = run_automatic_evidence_search(
                            stage=stage,
                            focus=focus,
                            body_system_focus=body_system_focus,
                            context=ctx_without_fresh_evidence,
                            evidence_model=evidence_model,
                            source_scope=evidence_source_scope,
                            force_search=True,
                        )
                    st.session_state["automatic_evidence_research"] = fresh_evidence
                else:
                    st.session_state["automatic_evidence_research"] = "Automatic evidence web search disabled by clinician."

                # Rebuild context including evidence text
                ctx = context_now()
                a = run_ai(stage, focus, body_system_focus, strictness, ctx, uploaded_files, model)
                a = apply_deterministic_guardrails(a, ctx)
                st.session_state[f"analysis_{stage}"] = a
                st.session_state["last_analysis"] = a
                st.session_state["last_context"] = ctx
                st.session_state["last_md"] = save_report(ctx, a)
                st.success("تم التحليل")
            except Exception as e:
                st.error("حدث خطأ أثناء التحليل")
                st.exception(e)

with tabs[4]:
    st.header("⑤ Questions Before Exam")
    st.markdown("<div class='workflow-card'>أسئلة حسب الحالة قبل الفحص السريري، مع red flags وأمان دوائي.</div>", unsafe_allow_html=True)
    analyze_button("Generate questions BEFORE exam", "questions")
    if "analysis_questions" in st.session_state:
        render(st.session_state["analysis_questions"])

with tabs[5]:
    st.header("⑥ Exam Protocol: كيف أفحص؟")
    analyze_button("Generate detailed clinical exam protocol", "exam_protocol")
    if "analysis_exam_protocol" in st.session_state:
        render(st.session_state["analysis_exam_protocol"])

with tabs[7]:
    st.header("⑧ Questions After Exam")
    st.markdown("<div class='workflow-card'>بعد إدخال نتائج الفحص، يعطيك أسئلة جديدة مبنية على الموجودات.</div>", unsafe_allow_html=True)
    analyze_button("Interpret exam + generate questions AFTER exam", "exam_interpretation")
    if "analysis_exam_interpretation" in st.session_state:
        render(st.session_state["analysis_exam_interpretation"])

with tabs[8]:
    st.header("⑨ Dx & Workup")
    analyze_button("Analyze Dx & Workup after exam", "preliminary")
    if "analysis_preliminary" in st.session_state:
        render(st.session_state["analysis_preliminary"])

with tabs[10]:
    st.header("⑪ Full Review / Medication")
    analyze_button("Analyze results + questions AFTER labs/imaging + medication safety", "results")
    if "analysis_results" in st.session_state:
        render(st.session_state["analysis_results"])

    st.divider()
    analyze_button("Full Neuro-General Guard Review", "full_review")
    if "analysis_full_review" in st.session_state:
        render(st.session_state["analysis_full_review"])

with tabs[11]:
    st.header("⑫ Report / Search")
    search_history = st.text_input("🔎 Search saved reports", placeholder="headache, seizure, chest pain, MRI, patient id...")
    reports = sorted(DATA_DIR.glob("neuro_general_guard_*.md"), reverse=True)

    if st.session_state.get("automatic_evidence_research"):
        with st.expander("Automatic Evidence Web Search — raw evidence notes"):
            st.text(st.session_state.get("automatic_evidence_research", "")[:20000])

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
