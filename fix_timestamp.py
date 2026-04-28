import pandas as pd
import pytz
import firebase_admin
from firebase_admin import credentials, firestore

# =========================
# 🔥 INIT FIREBASE
# =========================
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# =========================
# 🔥 CONFIG
# =========================
COLLECTIONS = [
    "pm25_station_5030",
    "pm25_station_3295"
]

tz = pytz.timezone("Asia/Bangkok")

# =========================
# 🔥 PARSE (รองรับทุก format)
# =========================
def parse_datetime(x):
    dt = pd.to_datetime(x, errors="coerce")
    if pd.isna(dt):
        return None
    return dt

# =========================
# 🔥 CONVERT
# =========================
for col in COLLECTIONS:
    print(f"\n🔄 Processing {col}")

    docs = db.collection(col).stream()

    count = 0
    skip = 0

    for doc in docs:
        data = doc.to_dict()

        if "timestamp" not in data:
            skip += 1
            continue

        dt = parse_datetime(data["timestamp"])

        if dt is None:
            print(f"❌ skip invalid: {data['timestamp']}")
            skip += 1
            continue

        # 🔥 ใส่ timezone ถ้ายังไม่มี
        if dt.tzinfo is None:
            dt = tz.localize(dt)
        else:
            dt = dt.astimezone(tz)

        # 🔥 format ใหม่ (ตามที่คุณต้องการ)
        new_ts = dt.strftime("%d/%m/%Y %H:%M:%S")

        db.collection(col).document(doc.id).update({
            "timestamp": new_ts
        })

        count += 1

    print(f"✅ updated: {count}")
    print(f"⚠️ skipped: {skip}")

print("\n🎉 DONE")