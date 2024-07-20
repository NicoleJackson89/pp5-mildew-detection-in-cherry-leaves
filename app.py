import streamlit as st
from app_pages.multipage import MultiPage

# loads the pages scripts
from app_pages.page_summary import page_summary_body

app = MultiPage(app_name="Powdery Mildew Detection - Cherry Leaves")  # Creates an instance of the app

# app pages are added here using .add_page()
app.add_page("Quick Project Summary", page_summary_body)

app.run()  # Run the app