import pandas as pd
import joblib
import numpy as np

# טעינת המודל והנתונים
model = joblib.load('models/att_model.pkl')
le = joblib.load('models/att_label_encoder.pkl')
features = joblib.load('models/att_columns.pkl')
test = pd.read_csv('data/attribution/radcom_att_test.csv')

# שחזור הפיצ'רים (אותה לוגיקה כמו ב-Sniper)
def get_features(df):
    df['is_udp'] = df['Protocol'].apply(lambda x: 1 if str(x).lower() == 'udp' else 0)

    if 'max_fwd_packet_size' in df.columns:
        df['max_size'] = df['max_fwd_packet_size']
    else:
        handshake_cols = [c for c in df.columns if 'first_packet_sizes' in c]
        df['max_size'] = df[handshake_cols].max(axis=1)

    # Add symmetry_index as in att_model.py
    down_bytes = df['bwd_packets_length'] + 1
    up_bytes = df['fwd_packets_length'] + 1
    df['symmetry_index'] = np.abs(np.log(down_bytes / up_bytes))

    if 'max_fwd_inter_arrival_time' in df.columns:
        df['silence_ratio'] = df['max_fwd_inter_arrival_time'] / (df['mean_fwd_inter_arrival_time'] + 1e-6)

    df['bytes_ratio'] = df['bwd_packets_length'] / (df['fwd_packets_length'] + 1e-6)

    handshake_cols = [c for c in df.columns if 'first_packet_sizes' in c]
    df[handshake_cols[:15]] = df[handshake_cols[:15]].fillna(0)

    return df

# הכנת הנתונים
test_eng = get_features(test.copy())
X_test = test_eng[features].fillna(0)
y_true = test['attribution']

# ביצוע חיזוי
preds = model.predict(X_test)
preds_label = le.inverse_transform(preds)

# מציאת הטעויות ב-Audio
print("\n🕵️‍♂️ AUDIO FAILURE ANALYSIS:\n")
for i in range(len(y_true)):
    if y_true.iloc[i] == 'real_time_audio' and preds_label[i] != 'real_time_audio':
        print(f"❌ Mistake #{i}: Real is Audio -> Predicted as {preds_label[i]}")
        print(f"   - Max Size: {test_eng.iloc[i]['max_size']} (Video usually > 1000)")
        print(f"   - Bytes Ratio (Down/Up): {test_eng.iloc[i]['bytes_ratio']:.2f} (Audio ~1.0, Video >> 1.0)")
        print(f"   - Protocol: {test.iloc[i]['Protocol']}")
        print("-" * 40)