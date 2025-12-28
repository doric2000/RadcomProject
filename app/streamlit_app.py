import streamlit as st
import pandas as pd
import requests
import time
import io

# Backend API URL (defined in docker-compose)
API_URL = "http://backend:8000"

st.set_page_config(page_title="RADCOM Cyber Classifier", layout="wide")

# Custom CSS for professional aesthetics (no emoji style)
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #0e1117;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #f0f2f6;
        font-weight: 500;
        letter-spacing: -0.5px;
    }
    
    /* Buttons - Clean Professional Look */
    .stButton > button {
        background-color: #2563eb;
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 6px;
        font-weight: 500;
        transition: background-color 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #1d4ed8;
    }
    
    /* Dataframes */
    .stDataFrame {
        border: 1px solid #374151;
        border-radius: 4px;
    }
    
    /* Status Messages */
    .stSuccess, .stInfo, .stError {
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# 1. Header Section (Professional List Layout)
st.title("RADCOM COMPETITION - Cyber Classifier Challenge")
st.caption("Network Traffic Classification System")
st.caption("By Baruh Ifraimov and Dor Cohen")
st.caption("Version 1.0")
st.caption("2025")

st.divider()

# 2. Configuration Section
st.subheader("Configuration")
task_type = st.selectbox(
    "Select Classification Model", 
    ["app", "att"],
    format_func=lambda x: "Application Classification" if x == "app" else "Attribution Classification"
)

# 3. Upload Section
st.subheader("Data Upload")
uploaded_file = st.file_uploader("Select CSV File", type=["csv"])

if uploaded_file is not None:
    # Read file
    original_df = pd.read_csv(uploaded_file)
    
    # 3.1 Data Preview
    st.subheader("Data Preview")
    st.dataframe(original_df.head(), use_container_width=True)
    
    # 4. Action Section
    st.subheader("Actions")
    st.write(f"Loaded {len(original_df)} rows")
    
    if st.button("Run Prediction"):
        uploaded_file.seek(0)
        
        computation_success = False
        predictions = []
        
        with st.status("Processing...", expanded=True) as status_box:
            try:
                # A. Send request to backend
                st.write("Sending data to inference engine...")
                response = requests.post(
                    f"{API_URL}/predict/{task_type}", 
                    files={"file": uploaded_file}
                )
                
                if response.status_code == 200:
                    task_id = response.json().get("task_id")
                    st.write(f"Task ID: {task_id}")
                    st.write("Computing predictions...")
                    
                    # B. Polling - check if task finished
                    while True:
                        result_response = requests.get(f"{API_URL}/result/{task_id}")
                        result_data = result_response.json()
                        status = result_data.get("status")
                        
                        if status == "SUCCESS":
                            status_box.update(label="Complete", state="complete", expanded=False)
                            predictions = result_data["result"]
                            computation_success = True
                            break
                        
                        elif status == "FAILURE":
                            status_box.update(label="Failed", state="error")
                            st.error(f"Task failed: {result_data.get('error')}")
                            break
                        
                        else:
                            time.sleep(1) # wait 1 second before next check
                            
                else:
                    status_box.update(label="Error", state="error")
                    st.error(f"API Error: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                status_box.update(label="Connection Failed", state="error")
                st.error("Cannot connect to Backend API. Is Docker running?")
        
        # Display results OUTSIDE the status box so they remain visible/expanded
        if computation_success:
            # C. Combine Data
            results_df = original_df.copy()
            # Insert Prediction as first column (Renamed from PREDICTION to Prediction)
            results_df.insert(0, "Prediction", predictions)
            
            st.divider()
            st.subheader("Results")
            
            # Display metric
            st.info(f"Analysis complete. Total samples classified: {len(results_df)}")
            
            # D. Display Table
            # Ensure Prediction is the first column in the list
            cols = ["Prediction"] + [c for c in results_df.columns if c != "Prediction"]
            st.dataframe(results_df[cols], use_container_width=True, height=600)
            
            # E. Download
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results CSV",
                data=csv,
                file_name=f"radcom_{task_type}_predictions.csv",
                mime="text/csv"
            )