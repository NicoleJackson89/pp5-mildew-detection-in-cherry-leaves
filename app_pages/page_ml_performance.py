import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
from src.machine_learning.evaluate_clf import load_test_evaluation


def page_ml_performance_metrics():
    version = 'v4'

    st.write("### Label Frequencies on Train, Validation and Test Sets")

    st.info(
        f" The cherry leaves dataset was divided into three subsets:\n\n"
        f" * The training set comprises 1,472 images, representing 70% of the entire dataset. This data is used to "
        f" train the model, enabling it to generalize and make predictions on new, unseen data.\n\n"
        f" * The validation set comprises 210 images, representing 10% of the entire dataset. Assists in enhancing the "
        f" model's performance by refining it after each epoch, which is a full pass of the training set through the model.\n\n"
        f" * The test set comprises 422 images, representing 20% of the entire dataset. Provides information about the model's "
        f" final accuracy after the training phase is completed. This evaluation uses a batch of data that the model has never seen before.")

    labels_distribution = plt.imread(f"outputs/{version}/label_dist.png")
    st.image(labels_distribution, caption='Labels Distribution on Train, Validation and Test Sets')
    st.write("---")


    st.write("### Model History")

    st.info(
        f" Accuracy measures how closely the model's predictions (accuracy) match the true data (val_acc).\n"
        f" A good model that performs well on unseen data demonstrates its ability to generalize and avoid overfitting to the training dataset.\n\n"
        f" The loss is the total of errors made for each example in the training (loss) or validation (val_loss) sets.\n"
        f" The loss value indicates how poorly or well a model performs after each optimization iteration.")

    col1, col2 = st.beta_columns(2)
    with col1: 
        model_acc = plt.imread(f"outputs/{version}/model_training_accuracy.png")
        st.image(model_acc, caption='Model Training Accuracy')
    with col2:
        model_loss = plt.imread(f"outputs/{version}/model_training_losses.png")
        st.image(model_loss, caption='Model Training Losses')
    st.write("---")

    st.write("### Generalised Performance on Test Set")

    st.dataframe(pd.DataFrame(load_test_evaluation(version), index=['Loss', 'Accuracy']))
    
    