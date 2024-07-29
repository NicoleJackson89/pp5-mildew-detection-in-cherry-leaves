# Mildew Detection in Cherry Leaves

![Mildew Detection in Cherry Leaves Am I Responsive Image](./assets/readme_imgs/am-i-responsive.png)

The goal of this project is to visually differentiate between healthy cherry leaves and those affected by powdery mildew. The project is available for live viewing on the [Streamlit Dashboard](https://pp5-mildew-cherry-leaf-ecc22555c3b8.herokuapp.com/), where users can read more about the project and upload new images to test the model's performance.

The dashboard displays the results of the data analysis, a description and evaluation of the project's hypotheses, and detailed performance metrics of the machine learning model.

The project includes a series of Jupyter Notebooks that create a pipeline for data import, cleaning, visualization, and the development and evaluation of a deep learning model.


## Table of Contents
1. [Dataset Content](#dataset-content)
2. [Business Requirements](#business-requirements)
3. [Hypothesis and Validation](#hypothesis-and-how-to-validate)
4. [The rationale to map the business requirements](#the-rationale-to-map-the-business-requirements)

## Dataset Content

- The dataset used for this project is supplied by Code Institute and sourced from [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves).
- The dataset contains +4 thousand images taken from the client's crop fields where 50% of these images were utilised for the model training, validating and testing prcess. The images show healthy cherry leaves and cherry leaves that have powdery mildew, a fungal disease that affects many plant species. The cherry plantation crop is one of the finest products in their portfolio, and the company is concerned about supplying the market with a compromised quality product.

## Business Requirements

The cherry plantation crop from Farmy & Foods is facing a challenge where their cherry plantations have been presenting powdery mildew. Currently, the process is manual verification if a given cherry tree contains powdery mildew. An employee spends around 30 minutes in each tree, taking a few samples of tree leaves and verifying visually if the leaf tree is healthy or has powdery mildew. If there is powdery mildew, the employee applies a specific compound to kill the fungus. The time spent applying this compound is 1 minute. The company has thousands of cherry trees located on multiple farms across the country. As a result, this manual process is not scalable due to the time spent in the manual process inspection.

To save time in this process, the IT team suggested an ML system that detects instantly, using a leaf tree image, if it is healthy or has powdery mildew. A similar manual process is in place for other crops for detecting pests, and if this initiative is successful, there is a realistic chance to replicate this project for all other crops. The dataset is a collection of cherry leaf images provided by Farmy & Foods, taken from their crops.

1. The client is interested in conducting a study to visually differentiate a healthy cherry leaf from one with powdery mildew.
2. The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.
3. The client is interested in having the option to download a prediction report for the examined leaves.
4. It was agreed with the client to attain an accuracy rate of 97%.

## Hypothesis and how to validate?

Hypothesis: The machine learning model can accurately predict with the use of images whether a cherry leaf is healthy or contains powdery mildew based on its features

Validation: By following a systematic approach that includes data preparation, model training, and evaluation of healthy cherry leaves images and those that contain powdery mildew. 

Cherry leaves containing powdery mildew can be distinguished from healthy leaves by their appearance. This is verified by creating an average image study and an image montage to determine the differences in the appearance of both contaminated leaves and healthy leaves.
Contaminated leaves and healthy leaves can be determined with a 97% accuracy, this will be verified by evaluating the model on the test dataset, which achieves 99% accuracy%.


## The rationale to map the business requirements

### Business requirements:
- Differentiate Healthy and Powdery mildew Cherry Leaves: The main goal is to correctly identify whether cherry leaves are healthy or affected by powdery mildew.
- High Accuracy: Meet the client's requirement by ensuring the model achieves an accuracy of at least 97%.
- Prediction Report: Provide an option for clients to download prediction reports of the examined leaves.

### Epics:
1. Information Gathering and Data Collection: 
    - The importation of the cherry leaf image dataset from Kaggle.
2. Data Visualization, Cleaning, and Preparation:
    - Data cleaning to identify and correct errors or inconsistencies in the dataset to improve its quality, data preparation to transform raw data into a format more suitable for analysis and visualisation to graphically represent the data to uncover patterns, trends, and insights.
3. Model Training, Optimization, and Validation:
    - To teach the model to recognise patterns from the training set, improve the model's performance by tuning its parameters and adjusting its algorithms and validate by testing the model on new data to ensure it generalises the new data well and performs accurately on the unseen data.
4. Dashboard Planning, Designing, and Development:
    - To ensure that the dashboard is useful, user-friendly, and effective in communicating data insights to facilitate informed decision-making.
5. Dashboard Deployment and Release: 
    - To ensure that the dashboard is easily accessible and functional for the end-user to utilise for decision-making.

### User Stories/Tasks:
Information Gathering and Data Collection: 
 - User Story: As a data analyst/developer, I can gather all relevant data about cherry leaves, including images of both healthy and powdery mildew leaves, so that I can use this dataset for analysis and model training.
    - Task: Gather and download quality images from Kaggle, label and organize them, and document the process.  

Data Visualization, Cleaning, and Preparation:
 - User Story: As a data analyst/developer, I can clean the images collected in the dataset by removing or correcting errors, duplicates or irrelevant images so that I can have a good high quality dataset that imporves the perfomance of the model.
 - User Story: As a data analyst/developer, I need to prepare the dataset for the model training process so that I can make sure it is in the correct format and structure for the best results.
 - User Story: As a data analyst/developer, I need to visualise the data so that I can visually understand the dataset.
     - Task: Check the dataset for non-image files & reomove, correct any labels and remove duplicates.
     - Task: Remove excess images (50%), create sub-folders (train, test & validation) and split the data into relevant folders with the correct ratios.
     - Task: Plot the class distribution and display and save sample & average images, create an image montage on the two labels.

Model Training, Optimization, and Validation:
 - User Story: As a data analyst/developer, I need to train a deep learning on the newly prepared dataset so that I can accurately differentiate between healthy and powdery mildew cherry leaves.
 - User Story: As a data analyst/developer, I need to optimise the model so that I can get at least the minimum required accuracy and performance.
 - User Story: As a data analyst/developer, I need to validate the model so that I can make sure it generalises on unseen data.
     - Task: Create and train a CNN with multiple layers.
     - Task: Optimize hyperparameters to achieve a minimum accuracy of 97%, adjust batch size, epoches.
     - Task: Evaluate the model's performance using the test data.

Dashboard Planning, Designing, and Development:
 - User Story: As a data analyst/developer, I need to plan the dashboard features so that I can make sure it meets the client's needs and business requirments.
 - User Story: As a data analyst/developer, I need to design a user friendly dashboard so that a user can easily navigate and understand the features.
 - User Story: As a data analyst/developer, I need to develop the dashboard inline with the design specs so that it works as it is meant to.
     - Task: Implement features as agreed, test and debug.
     - Task: Gather and understand requirments to meet the client's needs.
     - Task: Must be visually appealing, user friendly, responsive and accissible of varius devices.

Dashboard Deployment and Release: 
 - User Story: As a data analyst/developer, I need to prepare & deploy the dashboard so that  users can access and make use of it no problems or setbacks. 
     - Task: Review and optimise code, prepare, configure the production environments & deploy.

## ML Business Case

- In the previous bullet, you potentially visualised an ML task to answer a business requirement. You should frame the business case using the method we covered in the course.

## Dashboard Design

- List all dashboard pages and their content, either blocks of information or widgets, like buttons, checkboxes, images, or any other items, that your dashboard library supports.
- Finally, during the project development, you may revisit your dashboard plan to update a given feature (for example, at the beginning of the project, you were confident you would use a given plot to display an insight, but later, you chose another plot type).

## Unfixed Bugs

- You will need to mention unfixed bugs and why they were unfixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a significant variable for consideration, paucity of time and difficulty understanding implementation is not a valid reason to leave bugs unfixed.

## Deployment

### Heroku

- The App live link is: `https://YOUR_APP_NAME.herokuapp.com/`
- Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
- The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.
6. If the slug size is too large, then add large files not required for the app to the .slugignore file.

## Main Data Analysis and Machine Learning Libraries

- Here, you should list the libraries used in the project and provide an example(s) of how you used these libraries.

## Credits

- In this section, you need to reference where you got your content, media and from where you got extra help. It is common practice to use code from other repositories and tutorials. However, it is necessary to be very specific about these sources to avoid plagiarism.
- You can break the credits section up into Content and Media, depending on what you have included in your project.

### Content

- The text for the Home page was taken from Wikipedia Article A.
- Instructions on how to implement form validation on the Sign-Up page were taken from [Specific YouTube Tutorial](https://www.youtube.com/).
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/).

### Media

- The photos used on the home and sign-up page are from This Open-Source site.
- The images used for the gallery page were taken from this other open-source site.

## Acknowledgements (optional)

- Thank the people who provided support throughout this project.
