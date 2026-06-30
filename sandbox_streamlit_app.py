import streamlit as st
import pandas as pd
import os
# Import the function from your existing rank.py
# (Make sure rank.py is structured to be imported or run as a function)
from runtime.rank import run_ranking_pipeline 

st.title("🏆 Hackathon Candidate Ranking Sandbox")
st.write("This sandbox uses precomputed Groq LLM reasons and embeddings to instantly rank candidates.")

# Path to your precomputed JSON
artifacts_json = "outputs/top_300_with_reasons.json"

if os.path.exists(artifacts_json):
    st.success("✅ Precomputed artifacts loaded successfully!")
    
    if st.button("Run Heuristic Ranking Engine"):
        with st.spinner("Applying heuristics..."):
            # Trigger your rank.py logic
            # Ensure your function outputs the final dataframe or saves the CSV
            df_final = run_ranking_pipeline(artifacts_json) 
            
            st.write("### Final Ranked Candidates")
            st.dataframe(df_final)
            
            # Allow judges to download the final CSV
            csv = df_final.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Final CSV",
                data=csv,
                file_name="final_ranked_candidates.csv",
                mime="text/csv"
            )
else:
    st.error("❌ Missing top_300_with_reasons.json in the artifacts/ folder.")