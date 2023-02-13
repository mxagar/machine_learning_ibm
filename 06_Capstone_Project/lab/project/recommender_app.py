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

# Using ANNs with Tensorflow exceeds the memory
# limits of the Heroku slug. Thus, you can deactivate
# models 6-7-8, which use Tensorflow, when the app
# is deployed. For local runs, you can allow ANNs.
ALLOW_ANN = True

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

@st.cache
def load_course_genres():
    """Load course genre table:
    course index, title, 14 binary genre features."""
    return backend.load_course_genres()

@st.cache
def load_user_profiles():
    """Load user profiles table:
    user id, 14 binary genre features."""
    return backend.load_user_profiles()

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

# Cache notes:
# https://docs.streamlit.io/library/advanced-features/caching
@st.cache(suppress_st_warning=True)
def train(model_name, params):
    """Train function for
    the selected model + hyperparameters + courses.
    
    Inputs:
        model_name: str
            Model from the selectbox.
        params: dict
            Hyperparameters.
    Returns:
        training_artifacts: dict
            Any model-specific artifacts generated during training:
            pipelines, dataframes, etc.
    """
    training_artifacts = None
    try:
        assert model_name in backend.MODELS
        model_index = backend.get_model_index(model_selection)
        do_train = False
        if model_index > 5: # Neural Networks & Co.
            if ALLOW_ANN:
                do_train = True            
        else:
            do_train = True
        if do_train:  
            with st.spinner('Training...'):
                time.sleep(0.5)
                training_artifacts = backend.train(model_name, params)
            st.success('Done!')
        else:
            st.write("Sorry, the Neural Networks model is not active at the moment\
                due to the slug memory quota on Heroku. \
                If you clone the repository, \
                you can try it on your local machine, though.")
        return training_artifacts
    except AssertionError as err:
        print("Model name must be in the drop down.") # we should use the logger
        raise err

def predict(model_name, params, training_artifacts):
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
        model_index = backend.get_model_index(model_selection)
        do_predict = False
        if model_index > 5: # Neural Networks & Co.
            if ALLOW_ANN:
                do_predict = True            
        else:
            do_predict = True
        if do_predict:
            with st.spinner('Generating course recommendations: '):
                time.sleep(0.5)
                # FIXME: new_id is also contained in params to trigger the cache miss for train()
                # That makes user_ids redundant, however, necessary if we want to
                # - maintain a lower API for predict()
                # - be agnostic of some contents in params within the lower API predict
                # Fix that! 
                new_id = params["new_id"]
                user_ids = [new_id]
                res, descr = backend.predict(model_name, user_ids, params, training_artifacts)
            st.success('Recommendations generated!')
            st.write(descr)
        else:
            st.write("Sorry, the Neural Networks model is not active at the moment\
                due to the slug memory quota on Heroku. \
                If you clone the repository, \
                you can try it on your local machine, though.")
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
params = {}
st.sidebar.subheader('2. Tune Hyper-parameters: ')
top_courses = st.sidebar.slider('Top courses',
                                min_value=1, max_value=100,
                                value=10, step=1)
params['top_courses'] = top_courses
# Model-dependent options
if model_selection == backend.MODELS[0]: # 0: "Course Similarity"
    course_sim_threshold = st.sidebar.slider('Course similarity threshold %',
                                             min_value=0, max_value=100,
                                             value=50, step=10)
    params['sim_threshold'] = course_sim_threshold
elif model_selection == backend.MODELS[1]: # 1: "User Profile"
    profile_threshold = st.sidebar.slider('Course topic alignment score',
                                          min_value=0, max_value=100,
                                          value=1, step=1)
    params['profile_threshold'] = profile_threshold
elif model_selection == backend.MODELS[2]: # 2: "Clustering"
    num_clusters = st.sidebar.slider('Number of clusters',
                                   min_value=1, max_value=30,
                                   value=11, step=1)
    params['num_clusters'] = num_clusters
    params['pca_variance'] = 1.0
elif model_selection == backend.MODELS[3]: # 3: "Clustering with PCA"
    num_clusters = st.sidebar.slider('Number of clusters',
                                   min_value=1, max_value=30,
                                   value=11, step=1)
    pca_variance = st.sidebar.slider('Genre variance coverage (PCA)',
                                   min_value=0, max_value=100,
                                   value=90, step=5)
    params['num_clusters'] = num_clusters
    params['pca_variance'] = pca_variance / 100.0
elif model_selection == backend.MODELS[4]: # 4: "KNN"
    pass
elif model_selection == backend.MODELS[5]: # 5: "NMF"
    num_components = st.sidebar.slider('Number of latent components (discovered topics)',
                                   min_value=1, max_value=30,
                                   value=15, step=1)
    params['num_components'] = num_components
elif model_selection == backend.MODELS[6] \
    or model_selection == backend.MODELS[7]\
    or model_selection == backend.MODELS[8]: # 6: "Neural Network"
    num_components = st.sidebar.slider('Number of latent components (embedding size)',
                                   min_value=1, max_value=30,
                                   value=16, step=1)
    num_epochs = st.sidebar.slider('Number of epochs',
                                   min_value=1, max_value=10,
                                   value=1, step=1)
    params['num_components'] = num_components
    params['num_epochs'] = num_epochs
    
    # Check sub-options
    if model_selection == backend.MODELS[7]: # 7: "Regression with Embedding Features"
        pass
    elif model_selection == backend.MODELS[8]: # 8: "Classification with Embedding Features"
        pass

# Training: Element 3 from sidebar
st.sidebar.subheader('3. Training: ')
training_button = False
training_button = st.sidebar.button("Train Model")
training_text = st.sidebar.text('')
# Initialize global training return variables
training_artifacts = None
model_index = backend.get_model_index(model_selection)
#new_id = None
# Start training process
if training_button and model_index < 6:
    #FIXME
    # For all models on ANN embeddings, we need to have the new user id (the interacting user)
    # in the training dataset to create embeddings for them.
    # Therefore, we train here only for the rest of the models
    training_artifacts = train(model_selection, params)

# Prediction
# Element 4 from sidebar
# Element 3 from main body (prediction results)
st.sidebar.subheader('4. Prediction')
# Start prediction process
pred_button = st.sidebar.button("Recommend New Courses")
if pred_button and selected_courses_df.shape[0] > 0:
    if model_index < 6 and not training_artifacts:
        # Since train() is cached, we don't really recompute everything
        # All models which are not based on ANN embeddings don't have
        # the new user entries in the ratings dataset yet - we need to add them now.
        training_artifacts = train(model_selection, params)
    # Create a new id for current user session
    # We create a new entry in the ratings.csv for the interactive user
    # who has selected the courses in the UI
    new_id = backend.add_new_ratings(selected_courses_df['COURSE_ID'].values)
    params["new_id"] = new_id
    if model_index > 5:
        #FIXME
        # For all models on ANN embeddings, we need to have the new user id (the interacting user)
        # in the training dataset to create embeddings for them.
        training_artifacts = train(model_selection, params)
    if new_id:
        res_df = predict(model_selection, params, training_artifacts)
        res_df = res_df[['COURSE_ID', 'SCORE']]
        course_df = load_courses()
        res_df = pd.merge(res_df, course_df, on=["COURSE_ID"]).drop('COURSE_ID', axis=1)
        st.table(res_df)
