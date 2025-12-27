import streamlit as st
import pandas as pd
import requests
import time
import io

# Backend API URL (defined in docker-compose)
API_URL = "http://backend:8000"

st.set_page_config(page_title="REDCOM Cyber Classifier", layout="wide")

st.title("[SECURITY] REDCOM Competition")
st.markdown("### Powered by Machine Learning & Docker Microservices")

# 1. Select task type
task_type = st.selectbox("Select Challenge:", ["att", "app"])

# 2. Upload file
uploaded_file = st.file_uploader("Upload Validation CSV", type=["csv"])

if uploaded_file is not None:
    # Read file for preview
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:", df.head())
    
    if st.button("[RUN] Run Prediction"):
        # Prepare the file to send
        # Reset file pointer to the start
        uploaded_file.seek(0)
        files = {"file": uploaded_file.getvalue()}
        
        with st.spinner(f'Sending data to Worker Queue ({task_type})...'):
            try:
                # A. Send request to backend
                response = requests.post(
                    f"{API_URL}/predict/{task_type}", 
                    files={"file": uploaded_file}
                )
                
                if response.status_code == 200:
                    task_id = response.json()["task_id"]
                    st.success(f"Task submitted! ID: {task_id}")
                    
                    # B. Polling - check if task finished
                    status_placeholder = st.empty()
                    while True:
                        result_response = requests.get(f"{API_URL}/result/{task_id}")
                        result_data = result_response.json()
                        status = result_data["status"]
                        
                        if status == "SUCCESS":
                            status_placeholder.success("Processing Complete!")
                            
                            # Receive predictions
                            predictions = result_data["result"]
                            
                            # Add predictions to original dataframe
                            # (Note: if order preserved — Celery keeps order — this is fine)
                            df['prediction'] = predictions
                            
                            st.write("### Results Table")
                            st.dataframe(df)
                            
                            # C. Download the results file
                            csv = df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Results CSV",
                                data=csv,
                                file_name=f"redcom_{task_type}_predictions.csv",
                                mime="text/csv",
                            )
                            break
                        
                        elif status == "FAILURE":
                            st.error("Task failed inside the worker.")
                            st.error(result_data)
                            break
                        
                        else:
                            status_placeholder.info(f"Status: {status}... Waiting for worker...")
                            time.sleep(2) # wait 2 seconds before next check
                            
                else:
                    st.error(f"Error submitting task: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to Backend API. Is Docker running?")