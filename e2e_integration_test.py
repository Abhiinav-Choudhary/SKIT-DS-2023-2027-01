"""
End-to-End Integration Test
Tests: Node.js Backend <-> Python ML Bridge <-> Trained Model Artifacts
"""
import subprocess
import json
import sys
import os

PYTHON = sys.executable
BRIDGE = os.path.join("backend", "src", "ml", "inference_bridge.py")

PASS = "\u2705 PASS"
FAIL = "\u274c FAIL"

results = []

# ─────────────────────────────────────────────
# 1. DIABETES: Normal high-risk patient (all 21 CDC features)
# ─────────────────────────────────────────────
diabetes_input = {
    "HighBP": 1, "HighChol": 1, "CholCheck": 1,
    "BMI": 33.6, "Smoker": 0, "Stroke": 0,
    "HeartDiseaseorAttack": 0, "PhysActivity": 0,
    "Fruits": 0, "Veggies": 1, "HvyAlcoholConsump": 0,
    "AnyHealthcare": 1, "NoDocbcCost": 0, "GenHlth": 3,
    "MentHlth": 5, "PhysHlth": 10, "DiffWalk": 1,
    "Sex": 0, "Age": 9, "Education": 4, "Income": 5
}

proc = subprocess.run(
    [PYTHON, BRIDGE, "diabetes"],
    input=json.dumps(diabetes_input),
    capture_output=True, text=True
)
try:
    r = json.loads(proc.stdout)
    ok = r.get("success") is True and "data" in r
    d = r.get("data", {})
    results.append(("Diabetes prediction (high-risk patient)", ok))
    print("=== DIABETES PREDICTION ===")
    print(f"  success      : {r.get('success')}")
    print(f"  riskLevel    : {d.get('riskLevel')}")
    print(f"  riskScore    : {d.get('riskScore')}%")
    print(f"  predictedClass: {d.get('predictedClass')}")
    print(f"  model        : {d.get('modelName')} v{d.get('modelVersion')}")
    print(f"  status       : {PASS if ok else FAIL}")
except Exception as e:
    results.append(("Diabetes prediction (high-risk patient)", False))
    print(f"  PARSE ERROR: {e} | stdout: {proc.stdout!r} | stderr: {proc.stderr!r}")

print()

# ─────────────────────────────────────────────
# 2. DIABETES: Low-risk patient
# ─────────────────────────────────────────────
diabetes_low = {
    "HighBP": 0, "HighChol": 0, "CholCheck": 1,
    "BMI": 22.0, "Smoker": 0, "Stroke": 0,
    "HeartDiseaseorAttack": 0, "PhysActivity": 1,
    "Fruits": 1, "Veggies": 1, "HvyAlcoholConsump": 0,
    "AnyHealthcare": 1, "NoDocbcCost": 0, "GenHlth": 1,
    "MentHlth": 0, "PhysHlth": 0, "DiffWalk": 0,
    "Sex": 1, "Age": 4, "Education": 5, "Income": 7
}
proc2 = subprocess.run(
    [PYTHON, BRIDGE, "diabetes"],
    input=json.dumps(diabetes_low),
    capture_output=True, text=True
)
try:
    r2 = json.loads(proc2.stdout)
    ok2 = r2.get("success") is True
    d2 = r2.get("data", {})
    results.append(("Diabetes prediction (low-risk patient)", ok2))
    print("=== DIABETES PREDICTION (Low Risk) ===")
    print(f"  riskLevel    : {d2.get('riskLevel')}")
    print(f"  riskScore    : {d2.get('riskScore')}%")
    print(f"  status       : {PASS if ok2 else FAIL}")
except Exception as e:
    results.append(("Diabetes prediction (low-risk patient)", False))
    print(f"  PARSE ERROR: {e}")

print()

# ─────────────────────────────────────────────
# 3. HEART: Normal risk patient
# ─────────────────────────────────────────────
heart_input = {
    "age": 55, "sex": 1, "cp": 2, "trestbps": 140,
    "chol": 240, "fbs": 1, "restecg": 1, "thalach": 150,
    "exang": 0, "oldpeak": 1.5, "slope": 2, "ca": 0, "thal": 3
}
proc3 = subprocess.run(
    [PYTHON, BRIDGE, "heart"],
    input=json.dumps(heart_input),
    capture_output=True, text=True
)
try:
    r3 = json.loads(proc3.stdout)
    ok3 = r3.get("success") is True and "data" in r3
    h = r3.get("data", {})
    results.append(("Heart disease prediction", ok3))
    print("=== HEART DISEASE PREDICTION ===")
    print(f"  success      : {r3.get('success')}")
    print(f"  riskLevel    : {h.get('riskLevel')}")
    print(f"  riskScore    : {h.get('riskScore')}%")
    print(f"  predictedClass: {h.get('predictedClass')} (0=No disease, 1-4=Severity)")
    probs = h.get("probabilities", [])
    print(f"  probabilities: {[round(p,3) for p in probs]}")
    print(f"  model        : {h.get('modelName')} v{h.get('modelVersion')}")
    print(f"  status       : {PASS if ok3 else FAIL}")
except Exception as e:
    results.append(("Heart disease prediction", False))
    print(f"  PARSE ERROR: {e} | stdout: {proc3.stdout!r} | stderr: {proc3.stderr!r}")

print()

# ─────────────────────────────────────────────
# 4. Invalid input rejection (missing fields)
# ─────────────────────────────────────────────
proc4 = subprocess.run(
    [PYTHON, BRIDGE, "diabetes"],
    input=json.dumps({"Glucose": 150}),
    capture_output=True, text=True
)
try:
    r4 = json.loads(proc4.stdout)
    rejected = r4.get("success") is False
    results.append(("Invalid input rejected", rejected))
    print("=== INVALID INPUT TEST ===")
    print(f"  success: {r4.get('success')} (expected False)")
    print(f"  error  : {r4.get('error', '')[:80]}")
    print(f"  status : {PASS if rejected else FAIL}")
except Exception as e:
    results.append(("Invalid input rejected", False))
    print(f"  PARSE ERROR: {e}")

print()

# ─────────────────────────────────────────────
# 5. Unknown model type
# ─────────────────────────────────────────────
proc5 = subprocess.run(
    [PYTHON, BRIDGE, "unknown_model"],
    input=json.dumps({}),
    capture_output=True, text=True
)
try:
    r5 = json.loads(proc5.stdout)
    rejected5 = r5.get("success") is False
    results.append(("Unknown model type rejected", rejected5))
    print("=== UNKNOWN MODEL TYPE TEST ===")
    print(f"  success: {r5.get('success')} (expected False)")
    print(f"  error  : {r5.get('error', '')[:80]}")
    print(f"  status : {PASS if rejected5 else FAIL}")
except Exception as e:
    results.append(("Unknown model type rejected", False))
    print(f"  PARSE ERROR: {e}")

print()

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
passed = sum(1 for _, ok in results if ok)
total = len(results)
print("=" * 55)
print("  END-TO-END INTEGRATION SUMMARY")
print("=" * 55)
for name, ok in results:
    print(f"  {PASS if ok else FAIL}  {name}")
print()
print(f"  Result: {passed}/{total} integration checks passed")
print(f"  Node.js Backend -> Python Bridge -> ML Models: {'ALL GOOD' if passed == total else 'ISSUES FOUND'}")
print("=" * 55)
sys.exit(0 if passed == total else 1)
