"""This module implements a Streamlit app
which runs different Recommender Systems
that suggest new AI courses to a user
that pre-selects some topics.

This module uses the backend.py file,
where most of the machine learning
functionalities are implemented.

Author: Mikel Sagardia
Date: 2023-02-07
"""
import streamlit as st
import pandas as pd
import time
import backend as backend

from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import GridUpdateMode, DataReturnMode

# Basic webpage setup
st.set_page_config(
   page_title="Course Recommender System",
   layout="wide",
   initial_sidebar_state="expanded",
)

##
## Functions 
##

# Load datasets
@st.cache
def load_ratings():
    """Load ratings dataframe: user, course, rating (2/3)."""
    return backend.load_ratings()

@st.cache
def load_course_sims():
    """Load course similarities dataframe: course vs. course."""
    return backend.load_course_sims()

@st.cache
def load_courses():
    """Load courses dataframe: course, title, description."""
    return backend.load_courses()

@st.cache
def load_bow():
    """Load course bags-of-words (BoW) descriptors:
    course index and name, token, bow-count."""
    return backend.load_bow()

def init_recommender_app():
    """Initialization: It loads all dataframes to cache
    and builds and interactive AgGrid table for `course_processed.csv`
    from which user input is taken and used
    to generate a response dataframe.
    
    Inputs:
        None
    Outputs:
        results: pd.DataFrame
            Data frame with the selections by the user.
    """
    # Load all dataframes
    with st.spinner('Loading datasets...'):
        #ratings_df = load_ratings()
        #sim_df = load_course_sims()
        #course_bow_df = load_bow()
        course_df = load_courses()
    
    st.success('Datasets loaded successfully...')
    st.markdown("""---""")
    st.subheader("Select courses that you have audited or completed: ")

    # Streamlit AgGrid
    # https://pypi.org/project/streamlit-aggrid/
    # This component enables creating interactive tables in which:
    # - The user can select rows
    # - There are Filters and Column operations available:
    #   - Filters: filter cell values from different columns
    #   - Column operations: group by values, aggregate functions

    # GridOptionsBuilder: Define options of interactive table
    gb = GridOptionsBuilder.from_dataframe(course_df)
    gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    gb.configure_side_bar()
    grid_options = gb.build()

    # Create an AgGrid: an interactive table from which we get a response
    # with the rows selected by the user
    response = AgGrid(
        course_df,
        gridOptions=grid_options,
        enable_enterprise_modules=True,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load=False,
    )
    # response is a dictionary with these key:
    # response["data"]: entire CSV in a string
    # response["selected_rows"]: list of selected rows; each element is a dictionary
    #   with the column names as keys
    #   and row cell contents as values

    # Use user response to build a filtered dataframe
    # The column names should be related to the original columns
    # in df, e.g., we can use the same column names
    results = pd.DataFrame(response["selected_rows"],
                           columns=['COURSE_ID', 'TITLE', 'DESCRIPTION'])
    results = results[['COURSE_ID', 'TITLE']]
    st.subheader("Your courses: ")
    st.table(results)
    
    return results

def train(model_name, params):
    """Train function for
    the selected model + hyperparameters + courses.
    
    Inputs:
        model_name: str
            Model from the selectbox.
        params: dict
            Hyperparameters.
    Returns:
        None.
    """
    try:
        assert model_name in backend.MODELS
        with st.spinner('Training...'):
            time.sleep(0.5)
            backend.train(model_name)
        st.success('Done!')
    except AssertionError as err:
        print("Model name must be in the drop down.") # we should use the logger
        raise err

def predict(model_name, user_ids, params):
    """Predict function for
    the trained model.
    
    Inputs:
        model_name: str
            Model from the selectbox.
        user_ids: list
            User ids.
        params: dict
            Hyperparameters.
    Returns:
        res: pd.DataFrame
            Predicted/suggested courses.
    """
    res = None
    # Start making predictions based on model name, test user ids, and parameters
    try:
        assert model_name in backend.MODELS
        with st.spinner('Generating course recommendations: '):
            time.sleep(0.5)
            res = backend.predict(model_name, user_ids, params)
        st.success('Recommendations generated!')
        return res
    except AssertionError as err:
        print("Model name must be in the drop down.") # we should use the logger
        raise err
        
##
## User Interface (UI) 
##

# The user interface the following parts:
# - Sidebar
#   1. Model selection
#   2. Hyper-parameter tuning
#   3. Training button + action
#   4. Prediction button + action
# - Main body
#   1. Aggrid interactive table with course dataframe
#   2. Table with selected courses
#   3. Table with predictions

# Sidebar
st.sidebar.title('Personalized Learning Recommender')
# Initialize the app
# Here the UI elements 1 & 2 from the main body are created
selected_courses_df = init_recommender_app()

# Model selection selectbox: Element 1 from sidebar
st.sidebar.subheader('1. Select recommendation models')
model_selection = st.sidebar.selectbox(
    "Select model:",
    backend.MODELS
)

# Hyper-parameters for each model: Element 2 from sidebar
# MODELS
# 0: "Course Similarity"
# 1: "User Profile"
# 2: "Clustering"
# 3: "Clustering with PCA"
# 4: "KNN"
# 5: "NMF"
# 6: "Neural Network"
# 7: "Regression with Embedding Features"
# 8: "Classification with Embedding Features"
params = {}
st.sidebar.subheader('2. Tune Hyper-parameters: ')
# Course similarity model
if model_selection == backend.MODELS[0]:
    # Add a slide bar for selecting top courses
    top_courses = st.sidebar.slider('Top courses',
                                    min_value=0, max_value=100,
                                    value=10, step=1)
    # Add a slide bar for choosing similarity threshold
    course_sim_threshold = st.sidebar.slider('Course Similarity Threshold %',
                                             min_value=0, max_value=100,
                                             value=50, step=10)
    params['top_courses'] = top_courses
    params['sim_threshold'] = course_sim_threshold
# User profile model
elif model_selection == backend.MODELS[1]:
    profile_sim_threshold = st.sidebar.slider('User Profile Similarity Threshold %',
                                              min_value=0, max_value=100,
                                              value=50, step=10)
# Clustering model
elif model_selection == backend.MODELS[2]:
    cluster_no = st.sidebar.slider('Number of Clusters',
                                   min_value=0, max_value=50,
                                   value=20, step=1)
# TODO: Add hyper-parameters for other models
else:
    pass

# Training: Element 3 from sidebar
st.sidebar.subheader('3. Training: ')
training_button = st.sidebar.button("Train Model")
training_text = st.sidebar.text('')
# Start training process
if training_button:
    train(model_selection, params)

# Prediction
# Element 4 from sidebar
# Element 3 from main body (prediction results)
st.sidebar.subheader('4. Prediction')
# Start prediction process
pred_button = st.sidebar.button("Recommend New Courses")
if pred_button and selected_courses_df.shape[0] > 0:
    # Create a new id for current user session
    new_id = backend.add_new_ratings(selected_courses_df['COURSE_ID'].values)
    user_ids = [new_id]
    res_df = predict(model_selection, user_ids, params)
    res_df = res_df[['COURSE_ID', 'SCORE']]
    course_df = load_courses()
    res_df = pd.merge(res_df, course_df, on=["COURSE_ID"]).drop('COURSE_ID', axis=1)
    st.table(res_df)
