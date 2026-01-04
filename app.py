# app.py
import streamlit as st
from Streamlit_Rendering import repo
from Streamlit_Rendering import functions as fn  # <-- function.py를 쓰는 경우

st.set_page_config(page_title="Explainable News Recommender", layout="wide")

repo.init_db()

defaults = {
    "admin_mode": False,
    "selected_article_id": None,
    "search_query": "",
    "search_executed": False,
    "user_id": "default_user",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state["admin_mode"]:
    fn.render_admin_page()
elif st.session_state["selected_article_id"] is not None:
    fn.render_detail_page(st.session_state["selected_article_id"])
elif st.session_state["search_executed"]:
    fn.render_search_results_page(st.session_state["search_query"])
else:
    fn.render_main_page()
