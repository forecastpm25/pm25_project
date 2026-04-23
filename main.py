import requests
from datetime import datetime
import pytz
import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()
def should_run():
    doc_ref = db.collection("system").document("last_run")
    doc = doc_ref.get()

    now = datetime.utcnow()

    if doc.exists:
        last_run = doc.to_dict().get("time")
        if last_run:
            last_run = last_run.replace(tzinfo=None)
            if now - last_run < timedelta(hours=1):
                print("⏭️ Skip (ยังไม่ครบ 1 ชม.)")
                return False

    # update เวลา
    doc_ref.set({"time": now})
    return True

API_KEY = "v8JhZtLUNsQKZrmVb4f0Vz0762WaCQdlwHLgjjwa"

STATIONS = [
    {"id": "5030", "collection": "pm25_station_5030"},
    {"id": "3295", "collection": "pm25_station_3295"}
]

def run():
    now = datetime.now(pytz.timezone("Asia/Bangkok"))

    for s in STATIONS:
        url = f"https://open-api.cmuccdc.org/api/dustboy/station/{s['id']}?apikey={API_KEY}"
        res = requests.get(url)
        data = res.json()

        if isinstance(data, dict):
            data = [data]

        for i, item in enumerate(data):
            db.collection(s["collection"]).document(f"{int(now.timestamp())}_{i}").set({
                "pm25": item.get("pm25"),
                "temp": item.get("temp"),
                "humid": item.get("humid"),
                "timestamp": now.isoformat()
            })

if __name__ == "__main__":
    run()