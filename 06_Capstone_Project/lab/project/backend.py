"""This module is the backend of a Streamlit app
which runs different Recommender Systems
that suggest new AI courses to a user
that pre-selects some topics.

WARNING: In a more production-like environment
the backend would use a library/package
where models definition & training is implemented.
However, here, for the sake of simplicity,
everything is packed in the backend.

Also, note that there are several issues
marked with a FIXME tag.

Author: Mikel Sagardia
Date: 2023-02-07
"""

from os.path import isfile
import pandas as pd
import numpy as np

from scipy.spatial.distance import cosine

from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.losses import MeanSquaredError

MODELS = ("1. Course Similarity",
          "2. User Profile",
          "3. Clustering",
          "4. Clustering with PCA",
          "5. KNN",
          "6. NMF",
          "7. Neural Network",
          "8. Regression with Embedding Features",
          "9. Classification with Embedding Features")
DATA_ROOT = "data"
FILEPATH_RATINGS = DATA_ROOT+"/ratings.csv"
FILEPATH_COURSE_SIMS = DATA_ROOT+"/sim.csv"
FILEPATH_COURSES = DATA_ROOT+"/course_processed.csv"
FILEPATH_BOWS = DATA_ROOT+"/courses_bows.csv"
FILEPATH_COURSE_GENRES = DATA_ROOT+"/course_genre.csv"
FILEPATH_USER_PROFILES = DATA_ROOT+"/user_profile.csv"
RANDOM_SEED = 123
NUM_GENRES = 14
MODEL_DESCRIPTIONS = (
    "Course similarities are built from course text descriptions using Bags-of-Words (BoW). \
        A similarity value is the projection of a course descriptor vector in the form of a BoW \
        on another, i.e., the cosine similarity between both. Given the selected courses, \
        the set of courses with the highest similarity value are found.",
    "Courses have a genre descriptor vector which encodes all the topics covered by them. \
        User profiles can be built by summing the user course descriptors scaled by the ratings \
        given by the user. Then, for a target user profile, the unselected courses that are \
        most aligned with it can be found using the cosine similarity (i.e., dot product) \
        between the profile and the courses. Finally, the courses with the highest scores are provided.",
    "Courses have a genre descriptor vector which encodes all the topics covered by them. \
        User profiles can be built by summing the user course descriptors scaled by the ratings \
        given by the users. Then, those users can be clustered according to their profile. \
        This approach provides with the courses most popular within the user cluster.",
    "This approach is the same as the simple clustering approach, but the user profile descriptors \
        are transformed to their principal components and only a subset of them is taken, \
        enough to cover a percentage of the total variance, selected by the user.",
    "Given the ratings dataframe, course columns are treated as course descriptors, i.e., \
        each course is defined by all the ratings provided by the users. With that, a \
        course similarity matrix is built using the cosine similarity. Then, for the set of \
        selected courses, the most similar ones are suggested.",
    "Non-Negative Matrix Factorization is performed: given the ratings dataset which contains \
        the rating of each user for each course (sparse notation), the matrix is factorized \
        as the multiplication of two lower rank matrices. That lower rank is the size of \
        a latent space which represents discovered inherent features). With the factorization, \
        the ratings of unselected courses are predicted by multiplying the lower rank matrices, \
        which yields the approximate but complete user-course rating table.",
    "An Artificial Neural Network (ANN) which maps users and courses to ratings is defined and trained. \
        If the user is in the training set, the ratings for unselected courses can be predicted. \
        However, the most interesting part of this approach consists in extracting the user and course \
        embeddings from the ANN for later use. An embedding vector is a continuous N-dimensional \
        representation of a discrete object (e.g., a user).",
    "The user and item embeddings extracted from the ANN are used to build a linear regression model \
        which predicts the rating given the embedding of a user and a course.",
    "The user and item embeddings extracted from the ANN are used to build a random forest \
        classification model which predicts the rating given the embedding of a user and a course."
)

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

def get_model_index(model_name):
    """Get model index value given its name.

    Inputs:
        model_name: str
            Name of the model, contained in the MODELS tuple.

    Outputs:
        index: int
            Index of the model name in the MODELS tuple.
    """
    index = None
    for i in range(len(MODELS)):
        if model_name == MODELS[i]:
            index = i
            break
    return index

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

class RecommenderNet(keras.Model):
    
    def __init__(self, num_users, num_items, embedding_size=16, **kwargs):
        """Constructor.
           :param int num_users: number of users
           :param int num_items: number of items
           :param int embedding_size: the size of embedding vector
        """
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        
        # Define a user_embedding vector
        # Input dimension is the num_users
        # Output dimension is the embedding size
        self.user_embedding_layer = layers.Embedding(
            input_dim=num_users,
            output_dim=embedding_size,
            name='user_embedding_layer',
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        # Define a user bias layer
        self.user_bias = layers.Embedding(
            input_dim=num_users,
            output_dim=1,
            name="user_bias")
        
        # Define an item_embedding vector
        # Input dimension is the num_items
        # Output dimension is the embedding size
        self.item_embedding_layer = layers.Embedding(
            input_dim=num_items,
            output_dim=embedding_size,
            name='item_embedding_layer',
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        # Define an item bias layer
        self.item_bias = layers.Embedding(
            input_dim=num_items,
            output_dim=1,
            name="item_bias")
        
    def call(self, inputs):
        """Method to be called during model fitting.
           
           :param inputs: user and item one-hot vectors
        """
        # Compute the user embedding vector
        user_vector = self.user_embedding_layer(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        item_vector = self.item_embedding_layer(inputs[:, 1])
        item_bias = self.item_bias(inputs[:, 1])
        dot_user_item = tf.tensordot(user_vector, item_vector, 2)
        # Add all the components (including bias)
        x = dot_user_item + user_bias + item_bias
        # Sigmoid output layer to output the probability
        return tf.nn.relu(x)

def encode_ratings(raw_data):
    """Encode user-item ratings for the ANN training.

    Inputs:
        raw_data: pd.DataFrame
            Table with user-course/item ratings
    Outputs:
        encoded_data: pd.DataFrame
            Encoded table in which user & item ids are mapped to integers.
        user_idx2id_dict: dict
            User mappings.
        course_idx2id_dict: dict
            Course/item mappings
    """
    
    encoded_data = raw_data.copy()
    
    # Mapping user ids to indices
    user_list = encoded_data["user"].unique().tolist()
    user_id2idx_dict = {x: i for i, x in enumerate(user_list)}
    user_idx2id_dict = {i: x for i, x in enumerate(user_list)}
    
    # Mapping course ids to indices
    course_list = encoded_data["item"].unique().tolist()
    course_id2idx_dict = {x: i for i, x in enumerate(course_list)}
    course_idx2id_dict = {i: x for i, x in enumerate(course_list)}

    # Convert original user ids to idx
    encoded_data["user"] = encoded_data["user"].map(user_id2idx_dict)
    # Convert original course ids to idx
    encoded_data["item"] = encoded_data["item"].map(course_id2idx_dict)
    # Convert rating to int
    encoded_data["rating"] = encoded_data["rating"].values.astype("int")

    return encoded_data, user_idx2id_dict, course_idx2id_dict

def generate_train_test_datasets_ann(dataset, scale=True):
    """Scale and split dataset for training ANNs.

    Args:
        dataset: pd.DataFrame
            Table with features (user & item) and target (rating).
        scale: bool
            Whether to scale target or not. Defaults to True.

    Returns:
        x_train, x_val, x_test, y_train, y_val, y_test: np.array
            Scaled and splitted features and target.
    """
    min_rating = min(dataset["rating"])
    max_rating = max(dataset["rating"])

    dataset = dataset.sample(frac=1, random_state=42)
    x = dataset[["user", "item"]].values
    if scale:
        y = dataset["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
    else:
        y = dataset["rating"].values

    # Assuming training on 80% of the data and validating on 10%, and testing 10%
    train_indices = int(0.8 * dataset.shape[0])
    test_indices = int(0.9 * dataset.shape[0])

    x_train, x_val, x_test, y_train, y_val, y_test = (
        x[:train_indices],
        x[train_indices:test_indices],
        x[test_indices:],
        y[:train_indices],
        y[train_indices:test_indices],
        y[test_indices:],
    )
    return x_train, x_val, x_test, y_train, y_val, y_test

def train_ann(ratings_df, embedding_size, epochs):
    """Instantiate and train ANN model.

    Inputs:
        ratings_df: pd.DataFrame
            Dataframe with user-time-ratings.
        embedding_size: int
            Size of the latent embedding components.
        epochs: int
            Number of epochs.

    Outputs:
        res_dict: dict
            Dictionary with training artifacts.
        model: class RecommenderNet
            Keras ANN, trained.
    """
    res_dict = dict()
    
    # Encode ratings table to integers
    encoded_data, user_idx2id_dict, course_idx2id_dict = encode_ratings(ratings_df)
    # Scale dataset and split it to train/val/test
    X = generate_train_test_datasets_ann(encoded_data)
    x_train, x_val, x_test, y_train, y_val, y_test = X
    # Get size for ANN
    num_users = len(ratings_df['user'].unique())
    num_items = len(ratings_df['item'].unique())
    # Instantiate ANN
    model = RecommenderNet(num_users, num_items, embedding_size)
    model.compile(optimizer=Adam(learning_rate = .003),
                    loss=MeanSquaredError(), 
                    metrics=[RootMeanSquaredError()])
    
    # Train ANN
    #train_me = False
    train_me = True
    if train_me:
        run_hist = model.fit(x_train,
                            y_train,
                            validation_data=(x_val, y_val),
                            epochs=epochs,
                            shuffle=True)
    
    # Evaluate trained ANN
    rmse = model.evaluate(x_test,y_test,verbose=0)

    # Extract embeddings
    # Create a dataframe of the user features
    user_latent_features = model.get_layer('user_embedding_layer').get_weights()[0]
    user_columns = [f"UFeature{i}" for i in range(user_latent_features.shape[1])]
    user_embeddings_df = pd.DataFrame(data=user_latent_features, columns = user_columns)
    user_embeddings_df['user'] = user_embeddings_df.index
    # Shift column 'user' to first position
    first_column = user_embeddings_df.pop('user')
    user_embeddings_df.insert(0, 'user', first_column)
    # Decode user ids
    user_embeddings_df['user'] = user_embeddings_df['user'].replace(user_idx2id_dict)
    # Create a dataframe of the item features
    item_latent_features = model.get_layer('item_embedding_layer').get_weights()[0]
    item_columns = [f"CFeature{i}" for i in range(item_latent_features.shape[1])]
    item_embeddings_df = pd.DataFrame(data=item_latent_features, columns = item_columns)
    item_embeddings_df['item'] = item_embeddings_df.index
    # Shift column 'item' to first position
    first_column = item_embeddings_df.pop('item')
    item_embeddings_df.insert(0, 'item', first_column)
    # Decode user ids
    item_embeddings_df['item'] = item_embeddings_df['item'].replace(course_idx2id_dict)

    # Pack all results
    #res_dict["model"] = model
    res_dict["rmse"] = rmse
    res_dict["user_idx2id_dict"] = user_idx2id_dict
    res_dict["course_idx2id_dict"] = course_idx2id_dict
    res_dict["user_embeddings_df"] = user_embeddings_df
    res_dict["item_embeddings_df"] = item_embeddings_df

    return res_dict, model

def predict_ann_values(model,
                       unselected_course_ids,
                       new_id,
                       training_artifacts):
    """Given the trained model and new user & selected courses,
    predict ratings for the unselected courses.

    Inputs:
        model: class RecommenderNet
            Keras ANN, trained.
        unselected_course_ids: list
            List of unselected courses.
        new_id: int
            Index of the new user.
        training_artifacts: dict
            Training artifacts.
    Outputs:
        result: dict
            Rating prediction for each unselected course.
    """
    result = {}
    # Get and modify mapping dictionaries
    course_idx2id_dict = training_artifacts["course_idx2id_dict"]
    user_idx2id_dict = training_artifacts["user_idx2id_dict"]
    course_id2idx_dict = {v:k for k,v in course_idx2id_dict.items()}    
    user_id2idx_dict = {v:k for k,v in user_idx2id_dict.items()}    
    # Create dataframe with user data
    courses = list(unselected_course_ids)
    users = [new_id]*len(courses)
    data_dict = {"user": users, "item": courses}
    data_df = pd.DataFrame(data_dict, columns=["user", "item"])
    # Encode data
    data_df['item'] = data_df['item'].map(course_id2idx_dict)
    data_df['user'] = data_df['user'].map(user_id2idx_dict)
    data_df = data_df.dropna()
    # Extract data matrix
    x = data_df[["user", "item"]].values
    # Predict
    pred = model.predict(x)
    # Pack and decode
    data_df["ratings"] = pred.ravel()
    data_df['item'] = data_df['item'].map(course_idx2id_dict)
    data_df['user'] = data_df['user'].map(user_idx2id_dict)
    courses = data_df['item'].to_list()
    ratings = data_df["ratings"].to_list()
    result = {courses[i]: ratings[i] for i in range(len(courses))}
    
    return result

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

def create_user_profile(enrolled_course_ids,
                        course_genres_df):
    """Given a list of courses in which a user has enrolled,
    build a user profile based on the genres of those courses.

    Inputs:
        enrolled_course_ids: list
            List of selected courses, i.e., user enrolled courses.
        sim_matrix: pd.DataFrame
            Data frame with binary genre features for each course.
    Outputs:
        user_profile: numpy.array (1, NUM_GENRES=14)
            Array with genre weights associated to the user.
    """
    # Build profile
    user_profile = np.zeros((1,NUM_GENRES))
    standard_rating = 3.0
    for enrolled_course in enrolled_course_ids:
        course_descriptor = course_genres_df[course_genres_df.COURSE_ID == enrolled_course].iloc[:,2:].values
        user_profile += standard_rating*course_descriptor 

    return user_profile

def compute_user_profile_recommendations(user_profile,
                                         idx_id_dict, 
                                         enrolled_course_ids,
                                         course_genres_df):
    """Given a list of courses in which a user has enrolled,
    build a user profile based on the genres of those courses
    and suggest courses aligned in the genre/topic space.

    Inputs:
        user_profile: numpy.array (1, NUM_GENRES=14)
            Array with genre weights associated to the user.
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
                  num_clusters):
    
    """Cluster user according to their profile
    using K-Means. Apply PCA to the features (topics/genres)
    if specified.

    Inputs:
        user_profiles_df: pd.DataFrame
            Dataframe with generated user profiles.
        pca_variance: float
            Explained variance ratio if PCA is applied.
            PCA is applied only if < 1.0.
        num_clusters: int
            Number of clusters to find.
    Outputs:
        res_dict: dict
            Dictionary with training artifacts, incl. model.
    """
    res_dict = dict()
    # FIXME: I no longer store/return PCA components,
    # so it's better to use Pipeline.fit() even with such a small pipeline...
    # Extract features, user ids, etc.
    feature_names = [f for f in list(user_profiles_df.columns) if f != 'user']
    user_ids = user_profiles_df.loc[:, user_profiles_df.columns == 'user']
    # Scale
    features = user_profiles_df[feature_names]
    res_dict['feature_names'] = feature_names
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    # PCA
    n_components = len(feature_names)
    if pca_variance < 1.0:
        n_components = pca_variance
    pca = PCA(n_components=n_components)
    features_scaled_pca = pca.fit_transform(features_scaled)
    # K-Means Clustering
    kmeans = KMeans(n_clusters=num_clusters,
                init='k-means++',
                random_state=RANDOM_SEED)
    kmeans.fit(features_scaled_pca)
    # Extract cluster labels
    cluster_labels = kmeans.labels_
    # Assemble user-cluster dataframe
    labels_df = pd.DataFrame(cluster_labels)
    cluster_df = pd.merge(user_ids, labels_df, left_index=True, right_index=True)
    cluster_df.columns = ['user', 'cluster']
    res_dict['cluster_df'] = cluster_df

    # Pack transformers + model into a pipeline    
    pipe = Pipeline([("scaler", scaler),
                     ("pcs", pca),
                     ("kmeans", kmeans)])
    res_dict['pipe'] = pipe
    
    return res_dict

def predict_user_clusters(user_profiles_df,
                          training_artifacts):
    """Predict the cluster of the user profiles passed.

    Inputs:
        user_profiles_df: pd.DataFrame
            User profiles.
        training_artifact: dict
            Clustering artifacts, incl. model.
    Outputs:
        clusters: np.array
            Predicted user clusters.
    """
    # Unpack training artifacts
    feature_names = training_artifacts['feature_names']
    pipe = training_artifacts['pipe']
    # Run pipeline
    feature_names = training_artifacts['feature_names']
    clusters = pipe.predict(user_profiles_df[feature_names])

    return clusters

def compute_user_cluster_recommendations(cluster, 
                                         ratings_df,
                                         training_artifacts):
    """For a given cluster, computed the most common courses.

    Inputs:
        cluster: int
            Cluster id.
        ratings_df: pandas.DataFrame
            All user-course ratings.
        training_artifacts: dict
            List of selected courses, i.e., user enrolled courses.
    Outputs:
        res: dict
            Key: course id, str; value: score (=num enrollments).
    """
    # Initialize return dict as empty
    res = {}
    # Get cluster labels per user
    cluster_df = training_artifacts['cluster_df']
    # Join ratings (user-course) with user cluster labels
    ratings_labelled_df = pd.merge(ratings_df, cluster_df, left_on='user', right_on='user')
    # Aggregate/group by cluster and count enrollments for each course 
    courses_cluster = ratings_labelled_df[['item', 'cluster']]
    courses_cluster['count'] = [1] * len(courses_cluster)
    courses_cluster = courses_cluster.groupby(['cluster','item']).agg(enrollments = ('count','sum')).reset_index()
    # Take the rows with the required cluster values
    # and sort them according to the number of enrollments
    recommended_courses = (courses_cluster
                               .loc[courses_cluster.cluster==cluster,
                                    ["item", "enrollments"]]
                               .sort_values(by="enrollments",
                                            ascending=False)
                          )
    # Extract courses & enrollments (=scores) and pack them in a dictionary
    courses = list(recommended_courses.item)
    scores = list(recommended_courses.enrollments)
    res = {courses[i]:scores[i] for i in range(len(courses))}

    return res

def compute_course_user_similarities(rating_sparse_df):
    """Build course descriptor vectors taking the ratings
    provided by each user. Then, compute the cosine similarity between the
    course vectors.

    Args:
        rating_sparse_df: pd.DataFrame
            Sparse dataframe of user-course ratings.

    Returns:
        item_sim_df: pd.DataFrame
            Table wth item similarity values, matrix.
    """
    # Extract item list
    item_list = rating_sparse_df.columns[1:]
    item_list_df = pd.DataFrame(data=item_list, columns = ['item'])
    # Empts similarity matrix
    item_sim = np.zeros((len(item_list), len(item_list)))
    # Compare items pairwise
    for i, this_item in enumerate(item_list):
        this_item_ratings = rating_sparse_df[this_item].values
        for j, other_item in enumerate(item_list):
            other_item_ratings = rating_sparse_df[other_item].values
            similarity = 1 - cosine(this_item_ratings, other_item_ratings)
            # FIXME: matrix is symmetric, only half needs to be computed!
            item_sim[i,j] = similarity
    # Assemble similarity dataframe
    item_sim_df = pd.DataFrame(data=item_sim, columns=item_list)
    item_sim_df = pd.concat([item_list_df, item_sim_df], axis=1)
    # index, item, AI011EN, BC...
    # course_list = item_sim_df.columns[1:]
    # course_list = item_sim_df.item.values
    
    return item_sim_df

def compute_knn_courses(enrolled_course_ids,
                        idx_id_dict,
                        training_artifacts):
    """Given a list of enrolled courses 
    and a course similarity matrix (contained in the training artifacts),
    go through the non-selected courses and build a dictionary
    with the largest similarity score for each non-selected course.

    Inputs:
        enrolled_course_ids: list
        idx_id_dict: dict
        training_artifacts: dict

    Outputs:
        res: dict
    """
    # Initialize return dict as empty
    res = {}
    # Get course similarity matrix
    course_sim_df = training_artifacts["course_sim_df"]
    # Sets of attended/unattended courses
    all_courses = set(idx_id_dict.values())
    unselected_course_ids = all_courses.difference(enrolled_course_ids)
    # Get course score
    # FIXME: this could be one matrix multiplication
    for selected_course in enrolled_course_ids:
        # Get all similarities
        course_sim_row = course_sim_df.loc[course_sim_df.item==selected_course]
        for unselected_course in unselected_course_ids:
            score = 0
            if unselected_course in course_sim_row.columns:
                score = course_sim_row[unselected_course].values[0]
            if unselected_course not in res:
                res[unselected_course] = score
            else:
                if score >= res[unselected_course]:
                    res[unselected_course] = score    
        res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}

    return res

def preprocess_embeddings(ratings_df,
                          user_embeddings_df,
                          item_embeddings_df):
    """Join the ANN embeddings to the ratings dataframe
    and generate the features.

    Inputs:
        ratings_df: pd.DataFrame
        user_embeddings_df: pd.DataFrame
        item_embeddings_df: pd.DataFrame

    Outputs:
        X, y: np.array
    """
    # Merge user embedding features
    user_emb_merged = pd.merge(ratings_df, 
                                user_embeddings_df, 
                                how='left', 
                                left_on='user', 
                                right_on='user').fillna(0)
    # Merge course embedding features
    merged_df = pd.merge(user_emb_merged, 
                            item_embeddings_df, 
                            how='left', 
                            left_on='item', 
                            right_on='item').fillna(0)

    # Sum embedding features and create new dataset
    u_feautres = [f"UFeature{i}" for i in range(16)]
    c_features = [f"CFeature{i}" for i in range(16)]

    user_embeddings = merged_df[u_feautres]
    course_embeddings = merged_df[c_features]
    ratings = merged_df['rating']

    # Aggregate the two feature columns using element-wise add
    embedding_dataset = user_embeddings + course_embeddings.values
    embedding_dataset.columns = [f"Feature{i}" for i in range(16)]
    embedding_dataset['rating'] = ratings
    
    # Extract features and target        
    X = embedding_dataset.iloc[:, :-1]
    y = embedding_dataset.iloc[:, -1]

    return X, y

def create_embeddings_frame(user_id, user_embeddings_df, item_embeddings_df):
    """Given a new series of ratings, create a
    dataframe which contains the corresponding embeddings
    for each user-item row.

    Inputs:
        user_id: int
        user_embeddings_df: pd.DataFrame
        item_embeddings_df: pd.DataFrame

    Outputs:
        X: np.array
        unselected_course_ids: list
    """
    # Generate/load data
    ratings_df = load_ratings()
    idx_id_dict, _ = get_doc_dicts()
    user_ratings = ratings_df[ratings_df['user'] == user_id]
    enrolled_course_ids = user_ratings['item'].to_list() 
    all_courses = set(idx_id_dict.values())
    unselected_course_ids = list(all_courses.difference(enrolled_course_ids))            
    # Create dataframe with courses for which we predict ratings
    ratings_pred_df = pd.DataFrame(unselected_course_ids, columns=['item'])
    ratings_pred_df['user'] = [user_id]*len(unselected_course_ids)
    ratings_pred_df['rating'] = -1
    # Preprocess ratings dataframe
    X, _ = preprocess_embeddings(ratings_pred_df,
                        user_embeddings_df,
                        item_embeddings_df)

    return X, unselected_course_ids

def train(model_name, params):
    """Train the selected model.
    
    Each model has a dedicated case and produces a specific
    training_artifacts.
    
    Inputs:
        model_name: str
            Model name as in MODELS.
        params: dict
            Parameters collected in the UI.
    Outputs:
        training_artifacts: dict
            Training artifacts, sometimes the model/inference pipeline is included.
    """
    training_artifacts = dict()
    training_artifacts["model_name"] = model_name
    if model_name == MODELS[0]: # 0: "Course Similarity"
        # Nothing to train here
        pass
    elif model_name == MODELS[1]: # 1: "User Profile"
        # Nothing to train here
        pass
    elif model_name == MODELS[2] or model_name == MODELS[3]: # 2: "Clustering", 3: "Clustering with PCA"
        # Build user profiles and persist (if not present)
        user_profiles_df = load_user_profiles(get_df=True)
        pca_variance = params["pca_variance"]
        # Perform profile clustering and persist in same file
        res_dict = cluster_users(user_profiles_df=user_profiles_df, 
                                 pca_variance=pca_variance,
                                 num_clusters=params["num_clusters"])
        # Extend training_artifacts with the new created elements from res_dict
        training_artifacts.update(res_dict)
    elif model_name == MODELS[4]: # 4: "KNN"
        # Compute sparse ratings matrix
        ratings_df = load_ratings()
        ratings_sparse_df = (ratings_df.pivot(index='user',
                                             columns='item',
                                             values='rating')
                                       .fillna(0)
                                       .reset_index()
                                       .rename_axis(index=None,
                                                    columns=None))
        # Compute course similarity matrix based on users
        course_sim_df = compute_course_user_similarities(ratings_sparse_df)
        # Pack results to training_artifact
        training_artifacts["course_sim_df"] = course_sim_df
    elif model_name == MODELS[5]: # 5: "NMF"
        # Compute sparse ratings matrix
        ratings_df = load_ratings()
        ratings_sparse_df = (ratings_df.pivot(index='user',
                                             columns='item',
                                             values='rating')
                                       .fillna(0)
                                       .reset_index()
                                       .rename_axis(index=None,
                                                    columns=None))
        # Fit NMF model
        num_components = params["num_components"]
        nmf = NMF(n_components=num_components,
                  init='random',
                  random_state=RANDOM_SEED)
        nmf = nmf.fit(ratings_sparse_df.iloc[:,1:]) # (n_samples, n_components)
        H = nmf.components_ # (n_components, n_features)
        # W = nmf.transform(X)
        # X_hat = W@H
        # H (components) are constant, W (transformed X) changes every time
        # Pack results to training_artifact
        training_artifacts["components"] = H
        training_artifacts["nmf"] = nmf
    elif model_name == MODELS[6]\
        or model_name == MODELS[7]\
        or model_name == MODELS[8]: # 6: "Neural Network"
        # Load ratings dataset
        ratings_df = load_ratings()
        # Extract user parameters
        num_components = params['num_components']
        num_epochs = params['num_epochs']
        # Train ANN
        res_dict, model = train_ann(ratings_df, num_components, num_epochs)
        # Extend training_artifacts with the new created elements from res_dict
        training_artifacts.update(res_dict)
        # FIXME:
        # Tensorflow models cannot be passed in a dict,
        # because they're not hashable.
        # A quick and dirty solution is to compute the prediction here...
        #training_artifacts["model"] = model
        new_id = params["new_id"]
        idx_id_dict, _ = get_doc_dicts()
        all_courses = set(idx_id_dict.values())
        user_ratings = ratings_df[ratings_df['user'] == new_id]
        enrolled_course_ids = user_ratings['item'].to_list()
        unselected_course_ids = all_courses.difference(enrolled_course_ids)        
        result = predict_ann_values(model,
                                    unselected_course_ids,
                                    new_id,
                                    training_artifacts)
        training_artifacts["result"] = result
        # Prepare inputs for sub-options: regression & classification with embeddings
        user_embeddings_df = training_artifacts["user_embeddings_df"]
        item_embeddings_df = training_artifacts["item_embeddings_df"]        
        X, y = preprocess_embeddings(ratings_df,
                                    user_embeddings_df,
                                    item_embeddings_df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, # predictive variables
            y, # target
            test_size=0.1, # portion of dataset to allocate to test set
            random_state=RANDOM_SEED # we are setting the seed here, ALWAYS DO IT!
        )
        # Run sub-options
        if model_name == MODELS[7]: # 7: "Regression with Embedding Features"
            # Define and train model
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            pred = lr.predict(X_test)
            rmse = mean_squared_error(y_test, pred, squared=False)
            # Pack results
            training_artifacts["lr_model"] = lr
            training_artifacts["rmse_lr"] = rmse
        elif model_name == MODELS[8]: # 8: "Classification with Embedding Features"
            # Encode labels
            label_encoder = LabelEncoder()
            y_train_ = label_encoder.fit_transform(y_train.values.ravel())
            y_test_ = label_encoder.transform(y_test.values.ravel())
            # Define and train model
            rf = RandomForestClassifier(random_state=RANDOM_SEED,
                                        max_depth=20,
                                        min_samples_split=2,
                                        n_estimators=100)
            rf.fit(X_train, y_train_)
            pred = rf.predict(X_test)
            _, _, f1, _ = precision_recall_fscore_support(y_test_, pred)
            # Pack results
            training_artifacts["rf_model"] = rf
            training_artifacts["le_rf"] = label_encoder
            training_artifacts["f1_rf"] = np.mean(f1)
    
    return training_artifacts

def predict(model_name, user_ids, params, training_artifacts):
    """Predict with the trained model.
    
    Each model has its dedicated part.
    
    Inputs:
        model_name: str
            Model name as in MODELS.
        user_ids: int
            New user id.
        params: dict
            Parameters collected from the UI.
        training_artifacts: dict
            Training objects/artifacts, sometimes the model/inference pipeline is included.
    Outputs:
        res_df: dict
            Results packed in a dictionary which contains pairs
            course:score
        score_description: str
            String which describes the score.
    """
    users = []
    courses = []
    scores = []
    res_dict = dict()
    score_threshold = -1.0
    score_description = ""
    try:
        assert "model_name" in training_artifacts
    except AssertionError as err:
        print("You need to train the model before predicting!")
        raise(err)
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
            score_description = "Note: the score is the cosine similarity\
                                 between the selected and the recommended\
                                 courses."
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
            # Create user profile vector: (1,14)
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()
            user_profile = create_user_profile(enrolled_course_ids,
                                               course_genres_df)
            # Predict
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()
            res = compute_user_profile_recommendations(user_profile,
                                                       idx_id_dict, 
                                                       enrolled_course_ids,
                                                       course_genres_df)
            score_description = "Note: the score is the alignment (dot product)\
                                 between the user profile built with the selected\
                                 courses and the recommended ones."
        elif model_name == MODELS[2] or model_name == MODELS[3] : # 2: "Clustering", 3: "Clustering with PCA"
            # Generate/load data
            course_genres_df = load_course_genres()
            idx_id_dict, _ = get_doc_dicts()
            ratings_df = load_ratings()
            # Create user profile vector: (1,14)
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()
            user_profile = create_user_profile(enrolled_course_ids,
                                               course_genres_df)
            user_profile_df = pd.DataFrame(data=user_profile,
                                           columns=course_genres_df.columns[2:])
            # Get user cluster
            cluster = predict_user_clusters(user_profile_df,
                                            training_artifacts)[0]
            # Compute recommendations based on user cluster
            res = compute_user_cluster_recommendations(cluster, 
                                         ratings_df,
                                         training_artifacts)
            score_description = "Note: the score is the number of enrollments\
                of each recommended course, which belongs to the user\
                cluster of the interacting user."
        elif model_name == MODELS[4]: # 4: "KNN"
            # Generate/load data
            #course_genres_df = load_course_genres()
            idx_id_dict, _ = get_doc_dicts()
            ratings_df = load_ratings()
            # Get selected courses by user
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()
            # Compute k nearest neighbors to those users with the similarity matrix
            res = compute_knn_courses(enrolled_course_ids,
                                      idx_id_dict,
                                      training_artifacts)
            score_description = "Note: the score is the cosine similarity\
                of the suggested course with respect to one\
                of the selected courses."
        elif model_name == MODELS[5]: # 5: "NMF"
            # Generate/load data
            #course_genres_df = load_course_genres()
            #idx_id_dict, _ = get_doc_dicts()
            ratings_df = load_ratings()
            # Create sparse version
            #FIXME: We should not make sparse the complete frame, but only the user rows!
            ratings_sparse_df = (ratings_df.pivot(index='user',
                                        columns='item',
                                        values='rating')
                                .fillna(0)
                                .reset_index()
                                .rename_axis(index=None,
                                            columns=None))
            # Get user rows
            user_ratings_sparse = ratings_sparse_df[ratings_sparse_df['user'] == user_id]
            #print(user_ratings_sparse)
            # Transform user ratings to latent feature space
            # H (components) are constant, W (transformed X) changes every time
            H = training_artifacts["components"]
            nmf = training_artifacts["nmf"]            
            W = nmf.transform(user_ratings_sparse.iloc[:, 1:])            
            X_hat = W@H
            items = list(user_ratings_sparse.columns[1:])
            ratings = list(X_hat.ravel())
            res = {items[i]:ratings[i] for i in range(len(items))}                
            # Pack results to training_artifact
            score_description = "Note: the score is the rating predicted\
                by the Non-Negative Matrix Factorization (NMF) model."
        elif model_name == MODELS[6]: # 6: "Neural Network"
            res = training_artifacts["result"]
            score_description = "Note: the score is the rating predicted\
                by the neural network model."
        elif model_name == MODELS[7]: # 7: "Regression with Embedding Features"
            # Extract model
            lr = training_artifacts["lr_model"]
            # Generate/load data
            user_embeddings_df = training_artifacts["user_embeddings_df"]
            item_embeddings_df = training_artifacts["item_embeddings_df"]        
            X, unselected_course_ids = create_embeddings_frame(user_id,
                                                               user_embeddings_df,
                                                               item_embeddings_df)
            # Predict dataframe
            pred = lr.predict(X)
            # Pack results
            ratings = pred.ravel()
            res = {unselected_course_ids[i]:ratings[i] for i in range(len(unselected_course_ids))}                
            score_description = "Note: the score is the rating predicted\
                by the regression model which works\
                with the embeddings created by a neural network."
        elif model_name == MODELS[8]: # 8: "Classification with Embedding Features"
            # Extract model
            rf = training_artifacts["rf_model"]
            label_encoder = training_artifacts["le_rf"]
            # Generate/load data
            user_embeddings_df = training_artifacts["user_embeddings_df"]
            item_embeddings_df = training_artifacts["item_embeddings_df"]        
            X, unselected_course_ids = create_embeddings_frame(user_id,
                                                               user_embeddings_df,
                                                               item_embeddings_df)
            # Predict dataframe and process output
            pred = rf.predict(X)
            y_pred = label_encoder.inverse_transform(pred)
            prob = rf.predict_proba(X)
            ratings = y_pred*np.max(prob,axis=1)
            # Pack results
            res = {unselected_course_ids[i]:ratings[i] for i in range(len(unselected_course_ids))}                
            score_description = "Note: the score is the rating x probability predicted\
                by the classification model which works\
                with the embeddings created by a neural network."

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

    return res_df, score_description
