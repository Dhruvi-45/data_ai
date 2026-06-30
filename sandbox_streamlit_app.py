# import streamlit as st
# import pandas as pd
# import os
# import sys
# # Import the function from your existing rank.py
# # (Make sure rank.py is structured to be imported or run as a function)
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# # CORRECT IMPORT: Import the main execution function 'run'
# from runtime.rank import run as run_ranking_pipeline
# st.title("🏆 Hackathon Candidate Ranking Sandbox")
# st.write("This sandbox uses precomputed Groq LLM reasons and embeddings to instantly rank candidates.")

# # Path to your precomputed JSON
# artifacts_json = ".outputs/top_300_with_reasons.json"

# if os.path.exists(artifacts_json):
#     st.success("✅ Precomputed artifacts loaded successfully!")
    
#     if st.button("Run Heuristic Ranking Engine"):
#         with st.spinner("Applying heuristics..."):
#             # Trigger your rank.py logic
#             # Ensure your function outputs the final dataframe or saves the CSV
#             df_final = run_ranking_pipeline(artifacts_json) 
            
#             st.write("### Final Ranked Candidates")
#             st.dataframe(df_final)
            
#             # Allow judges to download the final CSV
#             csv = df_final.to_csv(index=False).encode('utf-8')
#             st.download_button(
#                 label="📥 Download Final CSV",
#                 data=csv,
#                 file_name="final_ranked_candidates.csv",
#                 mime="text/csv"
#             )
# else:
#     st.error("❌ Missing top_300_with_reasons.json in the artifacts/ folder.")











import streamlit as st
import pandas as pd
import os
import sys

# Force Python to look in the current directory for packages
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

# Import the main execution function 'run' from runtime/rank.py
from runtime.rank import run as run_ranking_pipeline

st.title("🏆 Hackathon Candidate Ranking Sandbox")
st.write("This sandbox uses precomputed Groq LLM reasons and embeddings to instantly rank candidates.")

# FIX: Build the path starting directly from SCRIPT_DIR (which is /mount/src/data_ai)
# This perfectly maps to /mount/src/data_ai/outputs/top300_with_reasons.json
artifacts_json = os.path.join(SCRIPT_DIR, "outputs", "top300_with_reasons.json")

if os.path.exists(artifacts_json):
    st.success("✅ Precomputed artifacts loaded successfully!")
    
    if st.button("Run Heuristic Ranking Engine"):
        with st.spinner("Applying heuristics..."):
            try:
                # Call the ranking logic function from rank.py
                ranked_data = run_ranking_pipeline() 
                
                st.write("### Final Ranked Candidates")
                
                # Convert the returned list of dicts into a clean Pandas DataFrame
                df_final = pd.DataFrame(ranked_data)
                st.dataframe(df_final)
                
                # Download button configuration
                csv = df_final.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Final CSV",
                    data=csv,
                    file_name="final_ranked_candidates.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Execution Error: {e}")
else:
    st.error(f"❌ Missing artifact file at target path: {artifacts_json}")