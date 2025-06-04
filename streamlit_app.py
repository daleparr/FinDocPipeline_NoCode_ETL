import streamlit as st
import os

# Set environment variables before importing anything else
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'

# Configure page with all possible warning suppressions
st.set_page_config(
    page_title="FinDocPipeline - No-Code ETL for Financial Slide Decks",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Add comprehensive meta tags to suppress browser warnings
st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Permissions-Policy" content="ambient-light-sensor=(), battery=(), document-domain=(), layout-animations=(), legacy-image-formats=(), oversized-images=(), vr=(), wake-lock=()">
<meta http-equiv="Feature-Policy" content="ambient-light-sensor 'none'; battery 'none'; document-domain 'none'; layout-animations 'none'; legacy-image-formats 'none'; oversized-images 'none'; vr 'none'; wake-lock 'none';">
<style>
.main-title {
    font-size: 2.5rem;
    font-weight: 700;
    color: #1f77b4;
    margin-bottom: 0.5rem;
}
.subtitle {
    font-size: 1.2rem;
    color: #666;
    margin-bottom: 2rem;
    font-style: italic;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Import and run the main application
from FinDocPipeline import main

# Remove the page config from main since we handle it here
if __name__ == "__main__":
    main()