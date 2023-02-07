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

import pandas as pd

DATA_ROOT = "data"
MODELS = ("Course Similarity",
          "User Profile",
          "Clustering",
          "Clustering with PCA",
          "KNN",
          "NMF",
          "Neural Network",
          "Regression with Embedding Features",
          "Classification with Embedding Features")

def load_ratings():
    """Load ratings dataframe: user, course, rating (2/3)."""
    return pd.read_csv(DATA_ROOT+"/ratings.csv")

def load_course_sims():
    """Load course similarities dataframe: course vs. course."""
    return pd.read_csv(DATA_ROOT+"/sim.csv")

def load_courses():
    """Load courses dataframe: course, title, description."""
    df = pd.read_csv(DATA_ROOT+"/course_processed.csv")
    df['TITLE'] = df['TITLE'].str.title()
    return df

def load_bow():
    """Load course bags-of-words (BoW) descriptors:
    course index and name, token, bow-count."""
    return pd.read_csv(DATA_ROOT+"/courses_bows.csv")

def add_new_ratings(new_courses):
    res_dict = {}
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
        updated_ratings.to_csv("ratings.csv", index=False)
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
        for unselect_course in unselected_course_ids:
            if enrolled_course in id_idx_dict and unselect_course in id_idx_dict:
                idx1 = id_idx_dict[enrolled_course]
                idx2 = id_idx_dict[unselect_course]
                sim = sim_matrix[idx1][idx2]
                if unselect_course not in res:
                    res[unselect_course] = sim
                else:
                    if sim >= res[unselect_course]:
                        res[unselect_course] = sim
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}

    return res

def train(model_name, params):
    """Train the selected model."""
    if model_name == MODELS[0]: # 0: "Course Similarity"
        pass
    elif model_name == MODELS[1]: # 1: "User Profile"
        pass
    elif model_name == MODELS[2]: # 2: "Clustering"
        pass
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
    sim_threshold = 0.6
    if "sim_threshold" in params:
        sim_threshold = params["sim_threshold"] / 100.0
    idx_id_dict, id_idx_dict = get_doc_dicts()
    sim_matrix = load_course_sims().to_numpy()
    users = []
    courses = []
    scores = []
    res_dict = {}

    for user_id in user_ids:
        # Course Similarity model
        if model_name == MODELS[0]: # 0: "Course Similarity"
            ratings_df = load_ratings()
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()
            res = course_similarity_recommendations(idx_id_dict,
                                                    id_idx_dict,
                                                    enrolled_course_ids,
                                                    sim_matrix)
            for key, score in res.items():
                if score >= sim_threshold:
                    users.append(user_id)
                    courses.append(key)
                    scores.append(score)
        # TODO: Add prediction model code here
        elif model_name == MODELS[1]: # 1: "User Profile"
            pass
        elif model_name == MODELS[2]: # 2: "Clustering"
            pass
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

    res_dict['USER'] = users
    res_dict['COURSE_ID'] = courses
    res_dict['SCORE'] = scores
    res_df = pd.DataFrame(res_dict, columns=['USER', 'COURSE_ID', 'SCORE'])

    return res_df
