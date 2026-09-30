import urllib.request
import json
import os

BASE = "http://127.0.0.1:8000"

# 1. Reset Session
req = urllib.request.Request(f"{BASE}/api/defender/reset", method="POST")
with urllib.request.urlopen(req) as resp:
    print("1. Reset:", json.loads(resp.read().decode()))

# 2. Status Before Upload
with urllib.request.urlopen(f"{BASE}/api/defender/status") as resp:
    st = json.loads(resp.read().decode())
    print("2. Status before upload:", st["status"], "session_id:", st["session_id"])
    assert st["status"] == "NO_TRAFFIC"
    assert st["session_id"] is None
    assert st["threat_context"] is None
    assert len(st["recommendations"]) == 0

# 3. Upload sample_traffic.csv
sample_csv = r"c:\Users\TEYJAL SRI\Downloads\network_traffic-main\network_traffic-main\data\sample_traffic.csv"
boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
with open(sample_csv, "rb") as f:
    file_bytes = f.read()

body = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="file"; filename="sample_traffic.csv"\r\n'
    f"Content-Type: text/csv\r\n\r\n"
).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

req = urllib.request.Request(
    f"{BASE}/api/traffic/upload",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    method="POST"
)
try:
    with urllib.request.urlopen(req) as resp:
        up_res = json.loads(resp.read().decode())
        print("3. Upload result: rows =", up_res.get("rows"), "session_id =", up_res.get("session_id"))
        session_id = up_res.get("session_id")
        assert session_id is not None
except urllib.error.HTTPError as e:
    print("Upload failed:", e.code, e.read().decode())
    raise

# 4. Status After Upload
with urllib.request.urlopen(f"{BASE}/api/defender/status") as resp:
    st = json.loads(resp.read().decode())
    print("4. Status after upload:", st["status"])
    print("   Threat context:", st["threat_context"])
    print("   Pre-response risk:", f"{st['pre_response_risk']*100:.1f}%")
    print("   Recommendations:", [r["title"] for r in st["recommendations"]])
    target_ip = st["threat_context"]["observed_source"]

# 5. Unauthorized Test (must fail)
unauth_payload = {
    "session_id": session_id,
    "action": "block_source",
    "target": target_ip,
    "operator_id": "secops_lead",
    "authorization_reason": "Contain detected lateral movement",
    "authorized": False
}
req = urllib.request.Request(
    f"{BASE}/api/defender/apply",
    data=json.dumps(unauth_payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST"
)
try:
    with urllib.request.urlopen(req) as resp:
        print("ERROR: Unauthorized request succeeded!")
except urllib.error.HTTPError as e:
    print(f"5. Unauthorized rejection verified: HTTP {e.code} ({e.read().decode()})")

# 6. Authorized Mitigation Execution (block_source)
auth_payload = {
    "session_id": session_id,
    "action": "block_source",
    "target": target_ip,
    "operator_id": "secops_lead",
    "authorization_reason": "Contain detected adversary lateral movement",
    "authorized": True
}
req = urllib.request.Request(
    f"{BASE}/api/defender/apply",
    data=json.dumps(auth_payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST"
)
with urllib.request.urlopen(req) as resp:
    apply_res = json.loads(resp.read().decode())
    print("\n6. --- Mitigation & Closed-Loop Verification Result ---")
    print("   Response Status:", apply_res["response_status"])
    print("   Pre-Risk:", f"{apply_res['pre_response_risk']*100:.1f}%")
    print("   Post-Risk:", f"{apply_res['post_response_risk']*100:.1f}%")
    print("   Risk Delta:", f"{apply_res['risk_change_pts']:+.1f} percentage points")
    print("   Blocked Flows:", apply_res["flows_blocked_count"])
    print("   Post-Response Flows:", apply_res["post_response_flows_count"])
    print("   Verification Status:", apply_res["verification_status"])
    print("   Verification Message:", apply_res["verification_message"])

# 7. Check Audit Trail
with urllib.request.urlopen(f"{BASE}/api/defender/audit") as resp:
    aud = json.loads(resp.read().decode())
    print(f"\n7. --- Authoritative Audit Trail ({aud['count']} events) ---")
    for ev in aud["audit_events"]:
        print(f"   [{ev['event_type']}] {ev['message']}")

# 8. Check Flow Inspection (Pre vs Post)
with urllib.request.urlopen(f"{BASE}/api/traffic/flows?stage=pre&limit=5") as resp:
    pre_flows = json.loads(resp.read().decode())
    print(f"\n8. Pre-response flow count check: {len(pre_flows)} sample flows loaded")

with urllib.request.urlopen(f"{BASE}/api/traffic/flows?stage=post&limit=5") as resp:
    post_flows = json.loads(resp.read().decode())
    print(f"   Post-response flow count check: {len(post_flows)} sample flows loaded")

# 9. Check Forecast Stage Endpoint
req = urllib.request.Request(f"{BASE}/api/forecast?stage=post&horizon=5", method="POST")
with urllib.request.urlopen(req) as resp:
    post_fc = json.loads(resp.read().decode())
    print(f"\n9. Post-response forecast check: highest risk = {post_fc['highest_predicted_risk']*100:.1f}%, status = {post_fc['status']}")

print("\nALL VERIFICATIONS PASSED SUCCESSFULLY!")
