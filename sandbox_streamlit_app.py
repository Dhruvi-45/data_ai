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

# FIX 1: Corrected filename (removed underscore from top300) 
# and built absolute path relative to the repository layout
artifacts_json = os.path.join(os.path.dirname(SCRIPT_DIR), "outputs", "top300_with_reasons.json")

if os.path.exists(artifacts_json):
    st.success("✅ Precomputed artifacts loaded successfully!")
    
    if st.button("Run Heuristic Ranking Engine"):
        with st.spinner("Applying heuristics..."):
            try:
                # FIX 2: Call the function without arguments, since 'run()' in rank.py 
                # takes no parameters and reads directly from the JSON.
                ranked_data = run_ranking_pipeline() 
                
                st.write("### Final Ranked Candidates")
                
                # Convert the returned list of dicts into a clean Pandas DataFrame for Streamlit
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