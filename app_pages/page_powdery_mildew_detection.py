import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

from src.data_management import download_dataframe_as_csv
from src.machine_learning.predictive_analysis import (
                                                    load_model_and_predict,
                                                    resize_input_img,
                                                    plot_pred_probs
                                                    )

def page_powdery_mildew_detection_body():
    st.info(
        f"* Upload pictures of cherry leaves to determine if they are affected by powdery"
        f" mildew and download a report of the analysis."
        )

    st.success(
        f"* You can download a set of healthy or powdery mildew-infected leaves for a live prediction "
        f"[here](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves)."
        )

    st.write("---")

    imgs_buffer = st.file_uploader('Upload an image/s of a cherry leaf',
                                        type='jpeg',accept_multiple_files=True)
   
    if imgs_buffer is not None:
        df_report = pd.DataFrame([])
        for image in imgs_buffer:

            img_pil = (Image.open(image))
            st.info(f"Cherry Leaf Sample: **{image.name}**")
            img_array = np.array(img_pil)
            st.image(img_pil, caption=f"Image Size: {img_array.shape[1]}px width x {img_array.shape[0]}px height")

            version = 'v2'
            resized_img = resize_input_img(img=img_pil, version=version)
            pred_proba, pred_class = load_model_and_predict(resized_img, version=version)
            plot_pred_probs(pred_proba, pred_class)

            df_report = df_report.append({"Name":image.name, 'Result': pred_class },
                                        ignore_index=True)
        
        if not df_report.empty:
            st.success("Analysis Report")
            st.table(df_report)
            st.markdown(download_dataframe_as_csv(df_report), unsafe_allow_html=True)


