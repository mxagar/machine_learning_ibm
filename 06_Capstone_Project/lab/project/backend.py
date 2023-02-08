"""This module is the backend of a Streamlit app
which runs different Recommender Systems
that suggest new AI courses to a user
that pre-selects some topics.

In a more production-like environment
the backend would use a library/package
where models definition & training is implemented.
However, here, for the sake of simplicity,
everything is packed in the backend.

Author: Mikel Sagardia
Date: 2023-02-07
"""

from os.path import isfile
import pandas as pd
import numpy as np

MODELS = ("Course Similarity",
          "User Profile",
          "Clustering",
          "Clustering with PCA",
          "KNN",
          "NMF",
          "Neural Network",
          "Regression with Embedding Features",
          "Classification with Embedding Features")
DATA_ROOT = "data"
FILEPATH_RATINGS = DATA_ROOT+"/ratings.csv"
FILEPATH_COURSE_SIMS = DATA_ROOT+"/sim.csv"
FILEPATH_COURSES = DATA_ROOT+"/course_processed.csv"
FILEPATH_BOWS = DATA_ROOT+"/courses_bows.csv"
FILEPATH_COURSE_GENRES = DATA_ROOT+"/course_genre.csv"
FILEPATH_USER_PROFILES = DATA_ROOT+"/user_profile.csv"

def load_ratings():
    """Load ratings dataframe: user, course, rating (2/3)."""
    return pd.read_csv(FILEPATH_RATINGS)

def load_course_sims():
    """Load course similarities dataframe: course vs. course."""
    return pd.read_csv(FILEPATH_COURSE_SIMS)

def load_courses():
    """Load courses dataframe: course, title, description."""
    df = pd.read_csv(FILEPATH_COURSES)
    df['TITLE'] = df['TITLE'].str.title()
    return df

def load_bow():
    """Load course bags-of-words (BoW) descriptors:
    course index and name, token, bow-count."""
    return pd.read_csv(FILEPATH_BOWS)

def load_course_genres():
    """Load course genre table:
    course index, title, 14 binary genre features."""
    return pd.read_csv(FILEPATH_COURSE_GENRES)

def load_user_profiles(get_df=True):
    """Load user profiles table:
    user id, 14 binary genre features.
    First, it is checked, whether the file exists.
    If not, it is created and persisted.
    We can get the dataset or not, depending on
    the value of the get_df flag.
    """
    if not isfile(FILEPATH_USER_PROFILES):
        course_genres_df = load_course_genres()
        ratings_df = load_ratings()
        build_user_profiles(course_genres_df,
                            ratings_df,
                            FILEPATH_USER_PROFILES)
    if get_df:
        return pd.read_csv(FILEPATH_USER_PROFILES)
    else:
        return None
        
def add_new_ratings(new_courses):
    """The ratings.csv table is extended with the choices
    of the new interactive user. All selected courses
    are rated with 3.0. This function is called after train()
    but before predict().

    Inputs:
        new_courses: list
            List of course ids (str).
    Outputs:
        new_id: int
            Index of the new user profile; None if no courses provided.
    """
    res_dict = {}
    new_id = None
    if len(new_courses) > 0:
        # Create a new user id, max id + 1
        ratings_df = load_ratings()
        new_id = ratings_df['user'].max() + 1
        users = [new_id] * len(new_courses)
        ratings = [3.0] * len(new_courses)
        res_dict['user'] = users
        res_dict['item'] = new_courses
        res_dict['rating'] = ratings
        new_df = pd.DataFrame(res_dict)
        updated_ratings = pd.concat([ratings_df, new_df])
        updated_ratings.to_csv(DATA_ROOT+"/ratings.csv", index=False)
        
    return new_id

# Create course id to index and index to id mappings
def get_doc_dicts():
    """Compute course_id <-> course_index dictionaries.
    
    Inputs:
        None
    Outputs:
        idx_id_dict: dict
            Key: course index, int; value: course id, str.
        id_idx_dict: dict
            Key: course id, str; value: course index, int.
    """
    bow_df = load_bow()
    grouped_df = bow_df.groupby(['doc_index', 'doc_id']).max().reset_index(drop=False)
    idx_id_dict = grouped_df[['doc_id']].to_dict()['doc_id']
    id_idx_dict = {v: k for k, v in idx_id_dict.items()}
    del grouped_df

    return idx_id_dict, id_idx_dict

def course_similarity_recommendations(idx_id_dict, 
                                      id_idx_dict,
                                      enrolled_course_ids, 
                                      sim_matrix):
    """Use the course similarity matrix computed from course text BoWs
    to get a dictionary of similar courses for a given course list.
    The result dictionary contains a key for each unselected course
    and an associated similarity value.
    
    Inputs:
        idx_id_dict: dict
            Key: course index, int; value: course id, str.
        id_idx_dict: dict
            Key: course id, str; value: course index, int.
        enrolled_course_ids: list
            List of selected courses, i.e., user enrolled courses.
        sim_matrix: numpy.array
            Similarity matrix between courses.
    Outputs:
        res: dict
            Key: course id, str; value: similarity.
    """
    all_courses = set(idx_id_dict.values())
    unselected_course_ids = all_courses.difference(enrolled_course_ids)
    # Create a dictionary to store your recommendation results
    res = {}
    # First find all enrolled courses for user
    for enrolled_course in enrolled_course_ids:
        for unselected_course in unselected_course_ids:
            if enrolled_course in id_idx_dict and unselected_course in id_idx_dict:
                idx1 = id_idx_dict[enrolled_course]
                idx2 = id_idx_dict[unselected_course]
                sim = sim_matrix[idx1][idx2]
                if unselected_course not in res:
                    res[unselected_course] = sim
                else:
                    if sim >= res[unselected_course]:
                        res[unselected_course] = sim
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}

    return res

def user_profile_recommendations(idx_id_dict, 
                                 enrolled_course_ids,
                                 course_genres_df):
    """Given a list of courses in which a user has enrolled,
    build a user profile based on the genres of those courses
    and suggest courses aligned in the genre/topic space.

    Inputs:
        idx_id_dict: dict
            Key: course index, int; value: course id, str.
        enrolled_course_ids: list
            List of selected courses, i.e., user enrolled courses.
        sim_matrix: pd.DataFrame
            Data frame with binary genre features for each course.
    Outputs:
        res: dict
            Key: course id, str; value: score.
    """
    # Sets of attended/unattended courses
    all_courses = set(idx_id_dict.values())
    unselected_course_ids = all_courses.difference(enrolled_course_ids)
    # Build profile
    user_profile = np.zeros((1,14))
    standard_rating = 3.0
    for enrolled_course in enrolled_course_ids:
        course_descriptor = course_genres_df[course_genres_df.COURSE_ID == enrolled_course].iloc[:,2:].values
        user_profile += standard_rating*course_descriptor 
    # Get course score
    # FIXME: this could be one matrix multiplication
    res = {}
    for unselected_course in unselected_course_ids:
        score = 0.0
        course_descriptor = course_genres_df[course_genres_df.COURSE_ID == unselected_course].iloc[:,2:].values
        score = np.dot(course_descriptor, user_profile.T)[0,0]
        if unselected_course not in res:
            res[unselected_course] = score
        else:
            if score >= res[unselected_course]:
                res[unselected_course] = score    
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}

    return res

def build_user_profiles(course_genres_df,
                        ratings_df,
                        filepath):
    """Given the course genre descriptors
    (i.e., the one-hot encoded topic vectors)
    and the ratings given by each user,
    build and persist a data frame in which each user
    has aggregated topic/genre values.

    Inputs:
        course_genres_df: pd.DataFrame
            Data frame with course genre descriptors.
        ratings_df: pd.DataFrame
            Data frame with user ratings.
        filepath: str
            File path of the persisted user profiles dataframe.
    Outputs:
        None.
    """
    user_ids = sorted(list(ratings_df.user.unique()))
    num_genres = course_genres_df.shape[1]-2
    user_matrix = np.zeros((len(user_ids), num_genres))
    # For each user, get their course ratings
    # and sum all one-hot encoded course descriptors scaled by the ratings
    for i, user in enumerate(user_ids):
        user_df = ratings_df.loc[ratings_df.user==user, :]
        user_profile = np.zeros((1,num_genres))
        user_courses = user_df["item"].to_list()
        for course in user_courses:
            rating = user_df.loc[user_df.item==course, "rating"].values[0]
            user_profile += rating*course_genres_df[course_genres_df.COURSE_ID==course].iloc[:, 2:].values
        user_matrix[i] = user_profile
    # Pack everything in a dataframe and persist
    user_profiles_df = pd.DataFrame(data=user_matrix, columns=course_genres_df.columns[2:])
    user_id_df = pd.DataFrame(data=user_ids, columns=['user'], dtype=int)
    user_profiles_df = pd.concat([user_id_df, user_profiles_df], axis=1)
    user_profiles_df.to_csv(filepath, index=False)

def cluster_users(user_profiles_df,
                  pca_variance,
                  num_clusters,
                  filepath):
    #user_profiles_df = load_user_profiles(get_df=True)
    pass

def train(model_name, params):
    """Train the selected model."""
    if model_name == MODELS[0]: # 0: "Course Similarity"
        # Nothing to train here
        pass
    elif model_name == MODELS[1]: # 1: "User Profile"
        # Nothing to train here
        pass
    elif model_name == MODELS[2]: # 2: "Clustering"
        # Build user profiles and persist (if not present)
        user_profiles_df = load_user_profiles(get_df=True)
        # Perform profile clustering and persist in same file
        cluster_users(user_profiles_df=user_profiles_df, 
                      pca_variance=1.0,
                      num_clusters=params["params"],
                      filepath=FILEPATH_USER_PROFILES)
    elif model_name == MODELS[3]: # 3: "Clustering with PCA"
        pass
    elif model_name == MODELS[4]: # 4: "KNN"
        pass
    elif model_name == MODELS[5]: # 5: "NMF"
        pass
    elif model_name == MODELS[6]: # 6: "Neural Network"
        pass
    elif model_name == MODELS[7]: # 7: "Regression with Embedding Features"
        pass
    elif model_name == MODELS[8]: # 8: "Classification with Embedding Features"
        pass

# Prediction
def predict(model_name, user_ids, params):
    """Predict with the trained model."""
    users = []
    courses = []
    scores = []
    res_dict = {}
    score_threshold = -1.0
    for user_id in user_ids: # usually, we'll have a unique user id
        if model_name == MODELS[0]: # 0: "Course Similarity"
            # Extract params
            sim_threshold = 0.2
            if "sim_threshold" in params:
                sim_threshold = params["sim_threshold"] / 100.0
            score_threshold = sim_threshold
            # Generated/load data
            idx_id_dict, id_idx_dict = get_doc_dicts()
            sim_matrix = load_course_sims().to_numpy()
            ratings_df = load_ratings()       
            # Predict
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()
            res = course_similarity_recommendations(idx_id_dict,
                                                    id_idx_dict,
                                                    enrolled_course_ids,
                                                    sim_matrix)
        elif model_name == MODELS[1]: # 1: "User Profile"
            # Extract params
            profile_threshold = 0.0
            if "profile_threshold" in params:
                profile_threshold = params["profile_threshold"]
            score_threshold = profile_threshold
            # Generate/load data
            course_genres_df = load_course_genres()
            idx_id_dict, _ = get_doc_dicts()
            ratings_df = load_ratings()
            # Predict
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()
            res = user_profile_recommendations(idx_id_dict, 
                                               enrolled_course_ids,
                                               course_genres_df)
        elif model_name == MODELS[2] or model_name == MODELS[3] : # 2: "Clustering", 3: "Clustering with PCA"
            if model_name == MODELS[3]:
                pass
            pass
        elif model_name == MODELS[4]: # 4: "KNN"
            pass
        elif model_name == MODELS[5]: # 5: "NMF"
            pass
        elif model_name == MODELS[6]: # 6: "Neural Network"
            pass
        elif model_name == MODELS[7]: # 7: "Regression with Embedding Features"
            pass
        elif model_name == MODELS[8]: # 8: "Classification with Embedding Features"
            pass

    # Filter results depending on score
    for key, score in res.items():
        if score >= score_threshold:
            users.append(user_id)
            courses.append(key)
            scores.append(score)

    # Create dataframe with results
    res_dict['USER'] = users
    res_dict['COURSE_ID'] = courses
    res_dict['SCORE'] = scores
    res_df = pd.DataFrame(res_dict, columns=['USER', 'COURSE_ID', 'SCORE'])
    res_df = res_df.drop_duplicates(subset=['COURSE_ID']).reset_index(drop=True)
    
    # Restrict number of results, if required
    if "top_courses" in params:
        top_courses = params["top_courses"]
        if res_df.shape[0] > top_courses and top_courses > 0:
            # Sort according to score
            res_df.sort_values(by='SCORE', ascending=False, inplace=True)
            # Select top_courses
            res_df = res_df.reset_index(drop=True).iloc[:top_courses, :]

    return res_df
