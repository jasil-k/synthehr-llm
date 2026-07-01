import json
from datetime import date

import streamlit as st
import pandas as pd

from evaluation import render_evaluation_section

from generator import (
    generate_patient_record,
    list_all_conditions,
    PatientRecord,
)

st.set_page_config(page_title="SynthEHR-LLM", layout="wide")

PREMIUM_DARK_CSS = """
<style>

@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');



/* ---------- Core Engine Reset ---------- */

html, body, [class*="css"], .stApp {

    font-family: 'Plus Jakarta Sans', sans-serif !important;

    background-color: #030712!important;

    color: #94a3b8 !important;

}



/* ---------- Top Header Bar Hide (Glitches Prevention) ---------- */

header[data-testid="stHeader"] {

    background: transparent !important;

}

/* ---------- Hide Streamlit Platform Chrome ---------- */
#MainMenu {
    visibility: hidden !important;
    display: none !important;
}

footer {
    visibility: hidden !important;
    display: none !important;
}

div[data-testid="stToolbarActions"] {
    display: none !important;
}

div[data-testid="stStatusWidget"] {
    display: none !important;
}

a[href*="github.com"] {
    display: none !important;
}

/* ---------- Headings & Labels ---------- */

h1, h2, h3, h4, h5, h6 {

    color: #ffffff !important;

    font-weight: 700 !important;

    letter-spacing: -0.02em !important;

}

h1 {

    font-size: 2.25rem !important;

    margin-bottom: 0.5rem !important;

}



/* ---------- Premium Curved Input Fields ---------- */

div[data-baseweb="select"] > div,

div[data-baseweb="input"] > div,

input, textarea, div[data-baseweb="number-input"] input {

    background-color: #111622 !important;

    border: 1px solid #1f293d !important;

    color: #ffffff !important;

    border-radius: 12px !important;

    padding: 0.2rem 0.5rem !important;

    transition: all 0.2s ease !important;

}



div[data-baseweb="select"] > div:hover, input:hover {

    border-color: #0095FF !important;

}



/* Focus and dropdown lists */

div[data-baseweb="popover"] ul {

    background-color: #111622 !important;

    border: 1px solid #1f293d !important;

}



/* ---------- Premium Flat Container Blocks ---------- */

div[data-testid="stForm"],

div[data-testid="stExpander"],

div[data-testid="stDataFrame"],

div.stAlert {

    background: #111622 !important;

    border: 1px solid #1b2436 !important;

    border-radius: 20px !important;

    padding: 1.5rem !important;

    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4) !important;

    transition: transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1), border-color 0.3s ease, box-shadow 0.3s ease !important;

}



/* Interactive Floating & Deep Soft Glow */

div[data-testid="stForm"]:hover,

div[data-testid="stExpander"]:hover,

div[data-testid="stDataFrame"]:hover {

    transform: translateY(-4px);

    border-color: #0095FF !important;

    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.6), 0 0 20px rgba(0, 149, 255, 0.2) !important;

}



/* ---------- Metric Box Overhaul (Matches Phone App Look) ---------- */

div[data-testid="stMetric"] {

    background: #161f30 !important;

    border: 1px solid #222f49 !important;

    border-radius: 16px !important;

    padding: 1rem 1.25rem !important;

    text-align: center !important;

}

div[data-testid="stMetricValue"] {

    color: #ffffff !important;

    font-size: 1.6rem !important;

    font-weight: 700 !important;

}

div[data-testid="stMetricLabel"] {

    color: #64748b !important;

    font-size: 0.8rem !important;

    text-transform: uppercase !important;

    letter-spacing: 0.05em !important;

}



/* ---------- Pill-Shaped Rounded Action Buttons ---------- */

.stButton > button,

.stDownloadButton > button,

.stFormSubmitButton > button {

    background: #0095FF !important;

    color: #ffffff !important;

    font-weight: 600 !important;

    letter-spacing: 0.02em !important;

    border: none !important;

    border-radius: 100px !important; /* Perfect Pill Capsule Shape */

    padding: 0.75rem 2rem !important;

    width: 100% !important;

    transition: all 0.25s cubic-bezier(0.25, 0.8, 0.25, 1) !important;

    box-shadow: 0 4px 14px rgba(0, 149, 255, 0.4) !important;

}



.stButton > button:hover,

.stDownloadButton > button:hover,

.stFormSubmitButton > button:hover {

    background: #26a4ff !important;

    transform: translateY(-2px);

    box-shadow: 0 8px 25px rgba(0, 149, 255, 0.6) !important;

}



/* Secondary Actions Neutral Treatment */

div[data-testid="stMarkdownContainer"] + button {

    background: #1a2235 !important;

    border: 1px solid #2d3b58 !important;

    color: #cbd5e1 !important;

    box-shadow: none !important;

}

div[data-testid="stMarkdownContainer"] + button:hover {

    background: #24304a !important;

    border-color: #0095FF !important;

    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;

}



/* ---------- Custom Sidebar Look ---------- */

section[data-testid="stSidebar"] {

    background-color: #090c10 !important;

    border-right: 1px solid #161a23 !important;

}

/* ---------- Sidebar Radio as Stacked Buttons ---------- */

section[data-testid="stSidebar"] div[role="radiogroup"] {
    display: flex !important;
    flex-direction: column !important;
    gap: 0.6rem !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: #111622 !important;
    border: 1px solid #1b2436 !important;
    border-radius: 14px !important;
    padding: 0.75rem 1rem !important;
    width: 100% !important;
    margin: 0 !important;
    transition: transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1), border-color 0.3s ease, box-shadow 0.3s ease !important;
    cursor: pointer !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    transform: translateY(-2px);
    border-color: #0095FF !important;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.5), 0 0 16px rgba(0, 149, 255, 0.25) !important;
}

/* Hide the native radio circle */
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
    display: none !important;
}

/* Label text styling */
section[data-testid="stSidebar"] div[role="radiogroup"] label p {
    color: #cbd5e1 !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    text-align: center !important;
    width: 100% !important;
}

/* Selected option gets a permanent subtle highlight (not full glow) */
section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"],
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    border-color: #2d3b58 !important;
    background: #161f30 !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
    color: #ffffff !important;
}

/* ---------- Customizing Progress/Tables ---------- */

div[data-testid="stProgress"] > div > div {

    background: #0095FF !important;

}

div[data-testid="stDataFrame"] * {

    background-color: transparent !important;

}



/* ---------- Seamless Cross-Device Adaptation ---------- */

@media (max-width: 1024px) {

    div[data-testid="stForm"] {

        padding: 1rem !important;

    }

    .stButton > button {

        padding: 0.65rem 1.5rem !important;

    }

}

</style>

"""



# Injects the upgraded UI engine
st.markdown(PREMIUM_DARK_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# SESSION STATE INIT
# ═══════════════════════════════════════════════

if "single_patient" not in st.session_state:
    st.session_state.single_patient = None

if "cohort" not in st.session_state:
    st.session_state.cohort: list[PatientRecord] = []


# ═══════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════

def record_to_json_dict(record: PatientRecord) -> dict:
    return record.model_dump(mode="json")


def cohort_to_json_bytes(records: list[PatientRecord]) -> bytes:
    data = [record_to_json_dict(r) for r in records]
    return json.dumps(data, indent=2).encode("utf-8")


def render_patient_detail(record: PatientRecord):
    d = record.demographics

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Name", d.name)
    col2.metric("Age", d.age_at_generation)
    col3.metric("Gender", d.gender)
    col4.metric("Smoker", "Yes" if d.smoker else "No")

    col1, col2, col3 = st.columns(3)
    col1.metric("Blood Type", d.blood_type)
    col2.metric("Ethnicity", d.ethnicity)
    col3.metric("DOB", d.dob.isoformat())

    st.markdown("<br><h3>Primary Conditions</h3>", unsafe_allow_html=True)
    if record.primary_conditions:
        cond_df = pd.DataFrame([
            {"Code": c.code, "Description": c.description, "Category": c.category}
            for c in record.primary_conditions
        ])
        st.dataframe(cond_df, use_container_width=True, hide_index=True)
    else:
        st.info("No conditions assigned.")

    st.markdown(f"<h3>Timeline ({len(record.timeline)} visits)</h3>", unsafe_allow_html=True)
    if not record.timeline:
        st.info("No timeline generated.")
    else:
        for visit in record.timeline:
            with st.expander(f"🔹 {visit.date} — {visit.visit_type}"):
                vcol1, vcol2, vcol3, vcol4 = st.columns(4)
                if visit.systolic_bp and visit.diastolic_bp:
                    vcol1.metric("BP", f"{visit.systolic_bp}/{visit.diastolic_bp}")
                if visit.glucose_mg_dl:
                    vcol2.metric("Glucose", f"{visit.glucose_mg_dl} mg/dL")
                if visit.heart_rate:
                    vcol3.metric("HR", visit.heart_rate)
                if visit.weight_kg:
                    vcol4.metric("Weight", f"{visit.weight_kg} kg")
                st.caption(f"Referenced codes: {', '.join(visit.referenced_codes) or 'none'}")
                st.write(visit.notes)

    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button(
        label="💾 Save Patient as JSON",
        data=json.dumps(record_to_json_dict(record), indent=2).encode("utf-8"),
        file_name=f"patient_{d.patient_id}.json",
        mime="application/json",
    )


# ═══════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════

st.sidebar.title("SynthEHR-LLM")
mode = st.sidebar.radio("Mode", ["Single Patient", "Bulk Cohort", "Evaluation"])

all_codes = list_all_conditions()
code_options = [f"{c['code']} — {c['description']}" for c in all_codes]


# ═══════════════════════════════════════════════
# SINGLE PATIENT MODE
# ═══════════════════════════════════════════════

if mode == "Single Patient":
    st.title("Single Patient Generator")
    st.caption("Generate one synthetic patient with a full longitudinal timeline.")

    with st.form("single_patient_form"):
        c1, c2 = st.columns(2)
        with c1:
            min_age = st.number_input("Min age", min_value=18, max_value=100, value=30)
            gender_choice = st.selectbox("Gender", ["Random", "Male", "Female"])
            num_years = st.slider("Timeline span (years)", 1, 15, 5)
        with c2:
            max_age = st.number_input("Max age", min_value=18, max_value=100, value=80)
            target_labels = st.multiselect(
                "Target conditions (optional)",
                options=code_options,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🧬 Generate Patient")

    if submitted:
        if min_age > max_age:
            st.error("Min age cannot be greater than max age.")
        else:
            force_gender = None if gender_choice == "Random" else gender_choice
            target_codes = [label.split(" — ")[0] for label in target_labels] or None

            with st.spinner("Generating demographics, conditions, and timeline..."):
                try:
                    record = generate_patient_record(
                        min_age=min_age,
                        max_age=max_age,
                        force_gender=force_gender,
                        target_conditions=target_codes,
                        num_years=num_years,
                    )
                    st.session_state.single_patient = record
                except Exception as e:
                    st.error(f"Generation failed: {e}")

    if st.session_state.single_patient is not None:
        st.divider()
        render_patient_detail(st.session_state.single_patient)
elif mode == "Evaluation":
    render_evaluation_section(st.session_state.cohort)

# ═══════════════════════════════════════════════
# BULK COHORT MODE
# ═══════════════════════════════════════════════

else:
    st.title("Bulk Cohort Generator")
    st.caption("Generate a cohort of synthetic patients for statistical evaluation.")
    st.warning(
      "⏳ This may take a few minutes for multiple patients. Thanks for your patience."
    )

    with st.form("bulk_cohort_form"):
        c1, c2 = st.columns(2)
        with c1:
            num_patients = st.number_input("Number of patients", min_value=1, max_value=100, value=10)
            min_age = st.number_input("Min age", min_value=18, max_value=100, value=30, key="bulk_min_age")
            num_years = st.slider("Timeline span (years)", 1, 15, 5, key="bulk_years")
        with c2:
            max_age = st.number_input("Max age", min_value=18, max_value=100, value=80, key="bulk_max_age")
            gender_mix = st.selectbox("Gender mix", ["Random", "All Male", "All Female"])

        st.markdown("---")
        target_labels = st.multiselect(
            "Target conditions (optional — forces these into EVERY patient in this cohort)",
            options=code_options,
            key="bulk_target_conditions",
        )
        st.markdown("---")

        append_mode = st.checkbox("Append to existing cohort (instead of replacing it)", value=False)
        submitted = st.form_submit_button("🧬 Generate Cohort")

    # Placeholder positioned BELOW the form — progress renders here, not above it.
    progress_slot = st.empty()

    if submitted:
        if min_age > max_age:
            st.error("Min age cannot be greater than max age.")
        else:
            force_gender = None
            if gender_mix == "All Male":
                force_gender = "Male"
            elif gender_mix == "All Female":
                force_gender = "Female"

            target_codes = [label.split(" — ")[0] for label in target_labels] or None

            if not append_mode:
                st.session_state.cohort = []

            progress = progress_slot.progress(0, text="Starting generation...")
            new_records = []
            errors = 0

            for i in range(num_patients):
                progress.progress(
                    i / num_patients,
                    text=f"Generating patient {i + 1} of {num_patients}...",
                )
                try:
                    record = generate_patient_record(
                        min_age=min_age,
                        max_age=max_age,
                        force_gender=force_gender,
                        target_conditions=target_codes,
                        num_years=num_years,
                    )
                    new_records.append(record)
                except Exception as e:
                    errors += 1
                    st.toast(f"⚠ Patient {i + 1} failed: {e}", icon="⚠️")

            progress.progress(1.0, text="Done.")
            st.session_state.cohort.extend(new_records)

            if errors:
                st.warning(f"{errors} patient(s) failed to generate (likely rate limits). "
                           f"Successfully added {len(new_records)}.")
            else:
                st.success(f"Successfully generated {len(new_records)} patients.")

    st.divider()

    if not st.session_state.cohort:
        st.info("No cohort generated yet. Use the form above.")
    else:
        st.subheader(f"Cohort Summary ({len(st.session_state.cohort)} patients)")

        summary_rows = []
        for r in st.session_state.cohort:
            d = r.demographics
            summary_rows.append({
                "Patient ID": d.patient_id,
                "Name": d.name,
                "Age": d.age_at_generation,
                "Gender": d.gender,
                "Smoker": d.smoker,
                "Conditions": ", ".join(c.code for c in r.primary_conditions),
                "# Visits": len(r.timeline),
            })

        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="💾 Save Cohort as JSON",
                data=cohort_to_json_bytes(st.session_state.cohort),
                file_name=f"cohort_{date.today().isoformat()}_{len(st.session_state.cohort)}patients.json",
                mime="application/json",
            )
        with col2:
            if st.button("🗑️ Clear Cohort"):
                st.session_state.cohort = []
                st.rerun()

        st.divider()
        st.subheader("Inspect Individual Patient")
        selected_id = st.selectbox(
            "Select a patient to view full timeline",
            options=[r.demographics.patient_id for r in st.session_state.cohort],
            format_func=lambda pid: next(
                f"{r.demographics.name} ({pid})"
                for r in st.session_state.cohort if r.demographics.patient_id == pid
            ),
        )
        selected_record = next(
            r for r in st.session_state.cohort if r.demographics.patient_id == selected_id
        )
        render_patient_detail(selected_record)