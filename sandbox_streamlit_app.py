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

# Path mapping to /mount/src/data_ai/outputs/top300_with_reasons.json
artifacts_json = os.path.join(SCRIPT_DIR, "outputs", "top300_with_reasons.json")

if os.path.exists(artifacts_json):
    st.success("✅ Precomputed artifacts loaded successfully!")
    
    if st.button("Run Heuristic Ranking Engine"):
        with st.spinner("Applying heuristics..."):
            try:
                # Call the ranking logic function from rank.py
                ranked_data = run_ranking_pipeline() 
                
                st.write("### Final Ranked Candidates")
                
                # 1. Convert raw list of dicts to a temporary DataFrame
                df_raw = pd.DataFrame(ranked_data)
                
                # 2. Build the strict 4-column layout required by the hackathon regulations
                df_final = pd.DataFrame()
                df_final["rank"] = range(1, len(df_raw) + 1)
                df_final["candidate_id"] = df_raw["candidate_id"]
                df_final["score"] = df_raw["composite"].round(4)
                df_final["reasoning"] = df_raw["llm_reason"]
                
                # 3. Set the rank column as the visual index row tracking margin
                df_final.set_index("rank", inplace=True)
                
                # Render the filtered table cleanly across the screen width
                st.dataframe(df_final, use_container_width=True)
                
                # Download button configuration (keeps the rank index explicitly out of column bodies)
                csv = df_final.to_csv(index=True).encode('utf-8')
                st.download_button(
                    label="📥 Download Final CSV",
                    data=csv,
                    file_name="top100_final.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Execution Error: {e}")
else:
    st.error(f"❌ Missing artifact file at target path: {artifacts_json}")






# import streamlit as st
# import pandas as pd
# import os
# import sys

# # Force Python to look in the current directory for packages
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# if SCRIPT_DIR not in sys.path:
#     sys.path.append(SCRIPT_DIR)

# # Import the main execution function 'run' from runtime/rank.py
# from runtime.rank import run as run_ranking_pipeline

# st.title("🏆 Hackathon Candidate Ranking Sandbox")
# st.write("This sandbox uses precomputed Groq LLM reasons and embeddings to instantly rank candidates.")

# # FIX: Build the path starting directly from SCRIPT_DIR (which is /mount/src/data_ai)
# # This perfectly maps to /mount/src/data_ai/outputs/top300_with_reasons.json
# artifacts_json = os.path.join(SCRIPT_DIR, "outputs", "top300_with_reasons.json")

# if os.path.exists(artifacts_json):
#     st.success("✅ Precomputed artifacts loaded successfully!")
    
#     if st.button("Run Heuristic Ranking Engine"):
#         with st.spinner("Applying heuristics..."):
#             try:
#                 # Call the ranking logic function from rank.py
#                 ranked_data = run_ranking_pipeline() 
                
#                 st.write("### Final Ranked Candidates")
                
#                 # Convert the returned list of dicts into a clean Pandas DataFrame
#                 df_final = pd.DataFrame(ranked_data)
#                 df_final.index = df_final.index + 1
#                 st.dataframe(df_final)
                
#                 # Download button configuration
#                 csv = df_final.to_csv(index=False).encode('utf-8')
#                 st.download_button(
#                     label="📥 Download Final CSV",
#                     data=csv,
#                     file_name="final_ranked_candidates.csv",
#                     mime="text/csv"
#                 )
#             except Exception as e:
#                 st.error(f"Execution Error: {e}")
# else:
#     st.error(f"❌ Missing artifact file at target path: {artifacts_json}")