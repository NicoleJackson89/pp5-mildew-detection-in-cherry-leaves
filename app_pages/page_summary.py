import streamlit as st
import matplotlib.pyplot as plt


def page_summary_body():

    st.write("### Quick Project Summary")

    st.info(
        f"**General Information**\n\n"
        f" Powdery mildew caused by Podosphaera clandestina affects cherry trees. "
        f" It appears as a white or grayish powder on leaves, causing curling, yellowing, and premature leaf drop.\n\n"
        f" The disease thrives in warm, dry conditions with poor air circulation.\n\n"
        f" Management includes improving air flow through pruning, planting resistant varieties, and using "
        f" fungicides at the first sign of symptoms. Severe cases can weaken trees and reduce fruit quality.\n\n"
        f" Visual criteria for infected leaves include:\n\n"
        f" * Light-green, circular lesions on either leaf surface\n"
        f" * Subtle white cotton-like growth on infected areas and fruits.")

    st.success(
        f"**Project Dataset**\n\n"
        f" The available [kaggle](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves) dataset contains "
        f" 4208 images consisting of healthy and infected leaves individually photographed (2104 images were used).\n\n"
        f" The model has achieved an accuracy of 99% \n\n"
        f" The data was split as follows: \n\n"
        f"* The training set - 70% of the data,\n"
        f"* The validation set - 10% of the data,\n"
        f"* The test set - 20% of the data.\n"
        f" \n")

    st.warning(
        f"**Project business requirements**\n\n"
        f" 1) A study analysis to differentiate visually between a healthy and powdery mildew cherry leaves.\n\n"
        f" 2) An accurate prediction of whether a given leaf is infected with powdery mildew or not.\n\n"
        f" 3) Download a report detailing the predictions for the examined leaves.")

    st.write(
        f" For additional information, please visit the "
        f" [Project README file](https://github.com/NicoleJackson89/pp5-mildew-detection-in-cherry-leaves#readme).")