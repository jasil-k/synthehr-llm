# generator.py
# Phase 2 Complete: Pydantic models + Faker demographics + LangChain ICD grounding

import json
import uuid
import random
from datetime import date
from pathlib import Path

from faker import Faker
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

fake = Faker()

import time
import threading
from collections import deque

class RateLimiter:
    """
    Sliding-window rate limiter. Blocks .wait() calls so that no more than
    `max_calls` happen within any rolling `period_seconds` window.
    Thread-safe (harmless now, future-proofs against batching/async later).
    """
    def __init__(self, max_calls: int = 5, period_seconds: float = 60.0):
        self.max_calls = max_calls
        self.period = period_seconds
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            now = time.monotonic()
            # drop timestamps that have fully exited the window
            while self._timestamps and now - self._timestamps[0] >= self.period:
                self._timestamps.popleft()

            if len(self._timestamps) >= self.max_calls:
                sleep_for = self.period - (now - self._timestamps[0]) + 0.25
                if sleep_for > 0:
                    print(f"  ⏳ Gemini rate limit guard: sleeping {sleep_for:.1f}s "
                          f"(5 RPM free-tier limit)...")
                    time.sleep(sleep_for)
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= self.period:
                    self._timestamps.popleft()

            self._timestamps.append(time.monotonic())


# Single shared limiter instance for ALL Gemini calls in this module.
# Free tier = 5 RPM. Set max_calls=5 to match; change here if your quota changes.
_gemini_rate_limiter = RateLimiter(max_calls=5, period_seconds=60.0)

# ─── Load the codes database ─────────────────────────────────────────────────

CODES_DB_PATH = Path(__file__).parent / "codes_db.json"

with open(CODES_DB_PATH, "r") as f:
    CODES_DB: dict = json.load(f)

# ─── Pydantic models ──────────────────────────────────────────────────────────

class ICDCode(BaseModel):
    code: str = Field(..., description="Valid ICD-10 code, e.g. E11")
    description: str = Field(..., description="Human-readable diagnosis name")
    category: str = Field(..., description="Medical category, e.g. Cardiovascular")

class PatientDemographics(BaseModel):
    patient_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    gender: str
    dob: date
    age_at_generation: int
    blood_type: str
    smoker: bool
    ethnicity: str

from datetime import timedelta

class VisitEvent(BaseModel):
    visit_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    date: date
    visit_type: str
    referenced_codes: list[str] = Field(default_factory=list)
    notes: str
    systolic_bp: int | None = None
    diastolic_bp: int | None = None
    glucose_mg_dl: float | None = None
    heart_rate: int | None = None
    weight_kg: float | None = None


class PatientRecord(BaseModel):
    demographics: PatientDemographics
    primary_conditions: list[ICDCode] = Field(default_factory=list)
    timeline: list[VisitEvent] = Field(default_factory=list)

# ─── Demographic generation ───────────────────────────────────────────────────

BLOOD_TYPES = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
BLOOD_TYPE_WEIGHTS = [35, 6, 8, 2, 3, 1, 37, 7]

ETHNICITIES = [
    "White", "Black or African American", "Hispanic or Latino",
    "Asian", "South Asian", "Middle Eastern", "Mixed"
]

def calculate_age(dob: date) -> int:
    today = date.today()
    return today.year - dob.year - (
        (today.month, today.day) < (dob.month, dob.day)
    )

def generate_demographics(
    min_age: int = 30,
    max_age: int = 80,
    force_gender: str | None = None
) -> PatientDemographics:
    gender = force_gender or random.choice(["Male", "Female"])
    name = fake.name_male() if gender == "Male" else fake.name_female()
    dob = fake.date_of_birth(minimum_age=min_age, maximum_age=max_age)
    age = calculate_age(dob)
    blood_type = random.choices(BLOOD_TYPES, weights=BLOOD_TYPE_WEIGHTS, k=1)[0]
    smoker = random.choices([True, False], weights=[14, 86], k=1)[0]
    ethnicity = random.choice(ETHNICITIES)

    return PatientDemographics(
        name=name,
        gender=gender,
        dob=dob,
        age_at_generation=age,
        blood_type=blood_type,
        smoker=smoker,
        ethnicity=ethnicity
    )

# ─── ICD-10 grounding tools (plain functions) ────────────────────────────────

def lookup_icd_code(query: str) -> dict:
    """
    Fuzzy-search the local ICD-10 database.
    Returns the best matching code dict, or an error dict.
    """
    query_lower = query.lower()
    best_match = None
    best_score = 0

    for code, data in CODES_DB.items():
        words = query_lower.split()
        score = sum(
            1 for w in words
            if w in data["description"].lower() or w in data["category"].lower()
        )
        if score > best_score:
            best_score = score
            best_match = (code, data)

    if best_match and best_score > 0:
        code, data = best_match
        return {
            "code": code,
            "description": data["description"],
            "category": data["category"],
            "common_comorbidities": data["common_comorbidities"]
        }

    return {"error": f"No matching ICD-10 code found for: '{query}'"}


def list_all_conditions() -> list[dict]:
    """Returns all conditions available in the local ICD-10 database."""
    return [
        {
            "code": code,
            "description": data["description"],
            "category": data["category"]
        }
        for code, data in CODES_DB.items()
    ]

# ─── Grounded condition assignment via structured LLM output ─────────────────

def assign_conditions_grounded(
    demographics: PatientDemographics,
    target_conditions: list[str] | None = None
) -> list[ICDCode]:
    """
    Asks the LLM to pick diagnoses, but constrains it to only choose
    from codes that exist in our local database — no hallucination possible.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import SystemMessage, HumanMessage

    # Build the allowed-codes list to inject into the prompt
    all_codes = list_all_conditions()
    codes_block = "\n".join(
        f"  {c['code']}: {c['description']} ({c['category']})"
        for c in all_codes
    )

    target_instruction = ""
    if target_conditions:
        target_instruction = (
            f"\nYou MUST include conditions related to: "
            f"{', '.join(target_conditions)}"
        )

    system_prompt = f"""You are a clinical informatics AI assigning ICD-10 diagnoses.

ALLOWED CODES (you may ONLY use codes from this list — no others):
{codes_block}

RULES:
1. Select 2 to 4 conditions that are medically plausible for this patient.
2. Consider age, gender, and smoking status.
3. You MUST only use codes from the ALLOWED CODES list above.
4. Return ONLY a valid JSON array — no explanation, no markdown, no preamble.

REQUIRED OUTPUT FORMAT:
[
  {{"code": "E11", "description": "Type 2 diabetes mellitus", "category": "Endocrine"}},
  {{"code": "I10", "description": "Essential (primary) hypertension", "category": "Cardiovascular"}}
]"""

    human_prompt = f"""Patient:
- Name: {demographics.name}
- Age: {demographics.age_at_generation}
- Gender: {demographics.gender}
- Smoker: {demographics.smoker}
- Blood type: {demographics.blood_type}
- Ethnicity: {demographics.ethnicity}
{target_instruction}

Assign appropriate diagnoses from the allowed list."""

    llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)
    _gemini_rate_limiter.wait()
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])

    raw = response.content

    if isinstance(raw, list):
        raw = "".join(
            part.get("text", "")
            if isinstance(part, dict)
            else str(part)
            for part in raw
        )

    raw = str(raw).strip()

    # Strip markdown fences if present
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("["):
                raw = part
                break

    codes_data = json.loads(raw)

    # Final validation: reject any code not in our database
    validated = []
    for item in codes_data:
        if item["code"] in CODES_DB:
            validated.append(ICDCode(**item))
        else:
            print(f"  ⚠ Rejected hallucinated code: {item['code']}")

    return validated


# ─── Master record assembly ───────────────────────────────────────────────────

def generate_patient_record(
    min_age: int = 30,
    max_age: int = 80,
    force_gender: str | None = None,
    target_conditions: list[str] | None = None,
    num_years: int = 5
) -> PatientRecord:
    demographics = generate_demographics(
        min_age=min_age, max_age=max_age, force_gender=force_gender
    )
    conditions = assign_conditions_grounded(
        demographics=demographics, target_conditions=target_conditions
    )
    record = PatientRecord(demographics=demographics, primary_conditions=conditions)
    record.timeline = generate_timeline(record, num_years=num_years)
    return record

# ═══════════════════════════════════════════════
# PHASE 3: Longitudinal Timeline Engine
# ═══════════════════════════════════════════════

VISIT_TYPES = [
    "Annual Physical", "Follow-up Visit", "Specialist Consult",
    "Urgent Care Visit", "Routine Lab Review"
]

def generate_timeline(
    record: "PatientRecord",
    num_years: int = 5,
    min_visits: int = 3,
    max_visits: int = 8
) -> list[VisitEvent]:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import SystemMessage, HumanMessage

    demographics = record.demographics
    conditions = record.primary_conditions
    if not conditions:
        return []

    today = date.today()
    earliest_possible = demographics.dob + timedelta(days=1)
    window_start = today - timedelta(days=365 * num_years)
    start_date = max(window_start, earliest_possible)
    end_date = today

    condition_codes = [c.code for c in conditions]
    conditions_block = "\n".join(f"  {c.code}: {c.description}" for c in conditions)

    system_prompt = f"""You are a clinical documentation AI generating a realistic
longitudinal visit history for a SYNTHETIC patient used for software testing only.
No real patient data is involved.

PATIENT'S DIAGNOSED CONDITIONS (you may ONLY reference these ICD-10 codes):
{conditions_block}

DATE CONSTRAINTS (strict):
- Every visit date MUST be on or after {start_date.isoformat()}
- Every visit date MUST be on or before {end_date.isoformat()}
- Visits MUST be listed in chronological order (earliest first)

RULES:
1. Generate between {min_visits} and {max_visits} visits.
2. Each visit's "referenced_codes" must be a subset of the allowed codes above.
3. Vitals must be clinically plausible given the referenced conditions, e.g.:
   - Hypertension (I10) -> elevated systolic/diastolic BP
   - Diabetes (E11) -> elevated glucose_mg_dl
   - Heart failure (I50) / COPD (J44) -> may show abnormal heart_rate
   Set any irrelevant vital field to null rather than guessing.
4. "notes" must be SOAP format (Subjective/Objective/Assessment/Plan),
   2-4 sentences total, clinically realistic but concise.
5. Vary visit_type across: {", ".join(VISIT_TYPES)}.
6. Return ONLY a valid JSON array - no explanation, no markdown, no preamble.

REQUIRED OUTPUT FORMAT (one object per visit):
[
  {{
    "date": "YYYY-MM-DD",
    "visit_type": "Follow-up Visit",
    "referenced_codes": ["I10", "E11"],
    "notes": "S: Patient reports... O: BP 148/92... A: ... P: ...",
    "systolic_bp": 148,
    "diastolic_bp": 92,
    "glucose_mg_dl": 142.0,
    "heart_rate": 78,
    "weight_kg": 84.5
  }}
]"""

    human_prompt = f"""Patient:
- Name: {demographics.name}
- Age: {demographics.age_at_generation}
- Gender: {demographics.gender}
- Smoker: {demographics.smoker}
- DOB: {demographics.dob.isoformat()}

Generate the {num_years}-year visit timeline now."""

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4)
    _gemini_rate_limiter.wait()
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])

    raw = response.content.strip()
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("["):
                raw = part
                break

    try:
        visits_data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  ⚠ Failed to parse timeline JSON: {e}")
        return []

    validated_visits = []
    for item in visits_data:
        try:
            visit_date = date.fromisoformat(item["date"])
        except (KeyError, ValueError, TypeError):
            print(f"  ⚠ Rejected visit with invalid date: {item.get('date')}")
            continue

        if visit_date < start_date or visit_date > end_date:
            print(f"  ⚠ Rejected visit outside allowed date range: {visit_date}")
            continue

        refs = item.get("referenced_codes", [])
        valid_refs = [c for c in refs if c in condition_codes]
        if len(valid_refs) != len(refs):
            dropped = set(refs) - set(valid_refs)
            print(f"  ⚠ Dropped unassigned/hallucinated codes from visit: {dropped}")

        validated_visits.append(VisitEvent(
            date=visit_date,
            visit_type=item.get("visit_type", "Follow-up Visit"),
            referenced_codes=valid_refs,
            notes=item.get("notes", ""),
            systolic_bp=item.get("systolic_bp"),
            diastolic_bp=item.get("diastolic_bp"),
            glucose_mg_dl=item.get("glucose_mg_dl"),
            heart_rate=item.get("heart_rate"),
            weight_kg=item.get("weight_kg"),
        ))

    validated_visits.sort(key=lambda v: v.date)
    return validated_visits


# ─── Phase 2 final test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    print("=" * 55)
    print("STEP 1: Demographics generation")
    print("=" * 55)
    demo = generate_demographics(min_age=40, max_age=70)
    print(f"  Name      : {demo.name}")
    print(f"  Age       : {demo.age_at_generation}")
    print(f"  Gender    : {demo.gender}")
    print(f"  Smoker    : {demo.smoker}")
    print(f"  Ethnicity : {demo.ethnicity}")

    print()
    print("=" * 55)
    print("STEP 2: ICD grounding tool (no LLM needed)")
    print("=" * 55)
    for query in ["diabetes", "hypertension", "heart failure", "banana"]:
        result = lookup_icd_code(query)
        if "error" in result:
            print(f"  '{query}' → ❌ {result['error']}")
        else:
            print(f"  '{query}' → ✅ [{result['code']}] {result['description']}")

    print()
    print("=" * 55)
    print("STEP 3: Grounded LLM condition assignment")
    print("=" * 55)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("  ⚠ Skipping — GOOGLE_API_KEY not set in .env")
    else:
        conditions = assign_conditions_grounded(
            demographics=demo,
            target_conditions=["diabetes", "hypertension"]
        )
        print("\n  Assigned conditions:")
        for c in conditions:
            print(f"    [{c.code}] {c.description} ({c.category})")

    print()
    print("=" * 55)
    print("STEP 4: Full cohort generation (3 patients)")
    print("=" * 55)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("  ⚠ Skipping — GOOGLE_API_KEY not set in .env")
    else:
        cohort = []
        for i in range(3):
            print(f"\n  Generating patient {i+1}/3...")
            record = generate_patient_record(
                min_age=40,
                max_age=75,
                target_conditions=["hypertension"]
            )
            cohort.append(record)
            print(f"  ✅ {record.demographics.name}, "
                  f"{record.demographics.age_at_generation}y, "
                  f"{record.demographics.gender}")
            for c in record.primary_conditions:
                print(f"       [{c.code}] {c.description}")

        print(f"\n  Cohort generated: {len(cohort)} patients")
        print(f"  Total conditions assigned: "
              f"{sum(len(r.primary_conditions) for r in cohort)}")
        print(f"  All records valid Pydantic objects: "
              f"{all(isinstance(r, PatientRecord) for r in cohort)}")

