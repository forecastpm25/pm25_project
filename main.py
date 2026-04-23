import requests
from datetime import datetime, timedelta
import pytz
import firebase_admin
from firebase_admin import credentials, firestore

# =========================
# 🔐 Firebase
# =========================
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# =========================
# 🇹🇭 เวลาไทย
# =========================
thai_tz = pytz.timezone("Asia/Bangkok")

# =========================
# ⏱ กันยิงซ้ำ (1 ชั่วโมง)
# =========================
def should_run():
    doc_ref = db.collection("system").document("last_run")
    doc = doc_ref.get()

    now = datetime.now(thai_tz)

    if doc.exists:
        last_run = doc.to_dict().get("time")
        if last_run:
            last_run = last_run.astimezone(thai_tz)

            diff = now - last_run
            print(f"⏱ เวลาห่าง: {diff}")

            if diff < timedelta(hours=1):
                print("⏭️ Skip (ยังไม่ครบ 1 ชม.)")
                return False

    # update เวลา
    doc_ref.set({"time": now})
    return True

# =========================
# 🔑 API
# =========================
API_KEY = "v8JhZtLUNsQKZrmVb4f0Vz0762WaCQdlwHLgjjwa"

STATIONS = [
    {"id": "5030", "collection": "pm25_station_5030"},
    {"id": "3295", "collection": "pm25_station_3295"}
]

# =========================
# 🚀 main
# =========================
def run():
    now = datetime.now(thai_tz)
    print("🇹🇭 เวลาไทย:", now)

    # 🔥 เช็คก่อน
    if not should_run():
        return

    print("🚀 RUNNING JOB...")

    # 🔥 format เวลา
    formatted_time = now.strftime("%d/%m/%Y %H:%M:%S")
    doc_id_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    for s in STATIONS:
        try:
            url = f"https://open-api.cmuccdc.org/api/dustboy/station/{s['id']}?apikey={API_KEY}"
            res = requests.get(url)

            if res.status_code != 200:
                print(f"❌ API ERROR ({s['id']}):", res.status_code)
                continue

            data = res.json()

            if isinstance(data, dict):
                data = [data]

            for i, item in enumerate(data):
                doc_id = f"{doc_id_time}_{i}"

                db.collection(s["collection"]).document(doc_id).set({
                    "station_id": s["id"],
                    "pm25": item.get("pm25"),
                    "temp": item.get("temp"),
                    "humid": item.get("humid"),

                    # 👇 อ่านง่าย
                    "timestamp": formatted_time,

                    # 👇 เอาไว้ query / sort
                    "timestamp_raw": now
                })

            print(f"✅ บันทึกสถานี {s['id']}")

        except Exception as e:
            print(f"🔥 ERROR station {s['id']}:", e)

# =========================
# ▶️ run
# =========================
if __name__ == "__main__":
    run()