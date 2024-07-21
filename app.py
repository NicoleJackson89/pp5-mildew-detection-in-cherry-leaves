import streamlit as st
from app_pages.multipage import MultiPage

# loads the pages scripts
from app_pages.page_summary import page_summary_body
from app_pages.page_cherry_leaves_visualizer import page_cherry_leaves_visualizer_body
from app_pages.page_powdery_mildew_detection import page_powdery_mildew_detection_body

app = MultiPage(app_name="Mildew Detection - Cherry Leaves")  # Creates an instance of the app

# app pages are added here using .add_page()
app.add_page("Quick Project Summary", page_summary_body)
app.add_page("Cherry Leaves Visualiser", page_cherry_leaves_visualizer_body)
app.add_page("Powdery Mildew detection", page_powdery_mildew_detection_body)

app.run()  # Run the app