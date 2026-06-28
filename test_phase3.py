# test_phase3.py
from generator import generate_patient_record

print("Generating 1 patient with 5-year timeline...\n")
record = generate_patient_record(num_years=5)

d = record.demographics
print(f"Patient: {d.name}, {d.age_at_generation}y, {d.gender}")
print(f"Conditions: {', '.join(c.code for c in record.primary_conditions)}\n")

print(f"Timeline ({len(record.timeline)} visits, chronological):\n")
for visit in record.timeline:
    print(f"--- {visit.date} | {visit.visit_type} ---")
    print(f"  Codes: {visit.referenced_codes}")
    vitals = []
    if visit.systolic_bp and visit.diastolic_bp:
        vitals.append(f"BP {visit.systolic_bp}/{visit.diastolic_bp}")
    if visit.glucose_mg_dl:
        vitals.append(f"Glucose {visit.glucose_mg_dl} mg/dL")
    if visit.heart_rate:
        vitals.append(f"HR {visit.heart_rate}")
    if visit.weight_kg:
        vitals.append(f"Weight {visit.weight_kg} kg")
    print(f"  Vitals: {', '.join(vitals) if vitals else 'none recorded'}")
    print(f"  Notes: {visit.notes}\n")

# Sanity checks
dates_sorted = [v.date for v in record.timeline]
print("Chronological order valid:", dates_sorted == sorted(dates_sorted))
print("All dates after DOB:", all(v.date > d.dob for v in record.timeline))
print("All dates <= today:", all(v.date <= __import__("datetime").date.today() for v in record.timeline))

allowed_codes = {c.code for c in record.primary_conditions}
print("All referenced codes valid:", all(
    set(v.referenced_codes).issubset(allowed_codes) for v in record.timeline
))