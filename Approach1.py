import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import requests
import json
import openai

# Function to load datasets
def load_datasets():
    path_genome = "/content/drive/MyDrive/DataSet_recom/genome-tags.csv"
    df_genome = pd.read_csv(path_genome)

    df_genome_score = pd.read_csv('/content/drive/MyDrive/DataSet_recom/genome-scores.csv')

    path_rating = "/content/drive/MyDrive/DataSet_recom/ratings.csv"
    df_rating = pd.read_csv(path_rating)

    path_tags = "/content/drive/MyDrive/DataSet_recom/tags.csv"
    df_tags = pd.read_csv(path_tags)

    path_links = "/content/drive/MyDrive/DataSet_recom/links.csv"
    df_links = pd.read_csv(path_links)

    path_movies = "/content/drive/MyDrive/DataSet_recom/movies.csv"
    movies_df = pd.read_csv(path_movies)
    
    return df_genome, df_genome_score, df_rating, df_tags, df_links, movies_df

# Function to preprocess the movies dataset
def preprocess_movies(movies_df, df_links):
    movies_df.drop(movies_df[movies_df['genres'] == "(no genres listed)"].index, inplace=True)
    new_movies_df = movies_df.merge(df_links, on='movieId')
    new_movies_df['genres'] = new_movies_df['genres'].str.replace('|', ' ')
    new_df_1k = new_movies_df.sample(1000)
    return new_df_1k

# Function to perform TF-IDF vectorization and K-means clustering
def cluster_movies(df, num_clusters=30):
    df['text'] = df['title'] + ' ' + df['genres']
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['text'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.toarray())
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    return df

# Function to select one movie from each cluster
def select_movies_from_clusters(df):
    return df.groupby('cluster').head(1)

# Function to generate movie recommendations using OpenAI API
def get_recommendations(selected_movies, openai_api_key):
    URL = "https://api.openai.com/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {openai_api_key}"}
    big_df = pd.DataFrame()

    for index, row in selected_movies.iterrows():
        movie_title = row['title']
        genres = row['genres']
        message = f"predict five movies that are similar to this movie '{movie_title}' and genre '{genres}' check similarity on the basis of genres, theme of movie, and other factors. The output is in the given JSON format: {{'recommended_movies': ['Interstellar', 'The Matrix', 'In Time', 'The Dark Knight', 'Eternal Sunshine of the Spotless Mind']}}."
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": message}],
            "temperature": 1.0,
            "top_p": 1.0,
            "n": 5,
            "stream": False,
            "presence_penalty": 0,
            "frequency_penalty": 0,
        }
        response = requests.post(URL, headers=headers, json=payload, stream=False)
        try:
            predicted_movies = response.json()['choices'][0]['message']['content']
        except:
            predicted_movies = "[]"
        json_data = json.loads(predicted_movies)
        given_movies = [movie_title] * len(json_data['recommended_movies'])
        predicted_movies = json_data['recommended_movies']
        data = {'Given Movie': given_movies, 'Predicted Movie': predicted_movies}
        df_final = pd.DataFrame(data)
        big_df = big_df.append(df_final, ignore_index=True)
    
    return big_df

# Function to check if the predicted movie is present in the dataset
def check_present(movie_to_check, new_movies_df):
    matching_rows = new_movies_df[new_movies_df['title'].str.contains(movie_to_check)]
    return 1 if not matching_rows.empty else 0

# Function to map predicted movies to their respective IDs
def map_title_to_id(movie_to_check, new_movies_df):
    if movie_to_check == "None":
        return "None"
    matching_rows = new_movies_df[new_movies_df['title'].str.contains(movie_to_check)]
    return matching_rows['movieId'].values[0] if not matching_rows.empty else None

# Main function
def main():
    openai_api_key = "YOUR_OPENAI_API_KEY"  # Replace with your actual API key

    # Load datasets
    df_genome, df_genome_score, df_rating, df_tags, df_links, movies_df = load_datasets()

    # Preprocess movies
    new_movies_df = preprocess_movies(movies_df, df_links)
    df = new_movies_df[['movieId', 'title', 'genres']]

    # Cluster movies
    df = cluster_movies(df)

    # Select movies from clusters
    selected_movies = select_movies_from_clusters(df)

    # Get recommendations from OpenAI API
    big_df = get_recommendations(selected_movies, openai_api_key)

    # Pivot the DataFrame
    pivot_df = big_df.pivot_table(index='Given Movie', columns=big_df.groupby('Given Movie').cumcount().add(1), values='Predicted Movie', aggfunc='first').reset_index()
    pivot_df.columns.name = None
    pivot_df.columns = ['Given Movie'] + [f'Predicted Movie {i}' for i in range(1, 6)]

    # Apply check_present function
    pivot_df['Predicted Movie 1'] = pivot_df['Predicted Movie 1'].apply(lambda x: x if check_present(x, new_movies_df) else None)
    pivot_df['Predicted Movie 2'] = pivot_df['Predicted Movie 2'].apply(lambda x: x if check_present(x, new_movies_df) else None)
    pivot_df['Predicted Movie 3'] = pivot_df['Predicted Movie 3'].apply(lambda x: x if check_present(x, new_movies_df) else None)
    pivot_df['Predicted Movie 4'] = pivot_df['Predicted Movie 4'].apply(lambda x: x if check_present(x, new_movies_df) else None)
    pivot_df['Predicted Movie 5'] = pivot_df['Predicted Movie 5'].apply(lambda x: x if check_present(x, new_movies_df) else None)

    # Merge two DataFrames
    merged_df = selected_movies.merge(pivot_df, left_on='title', right_on='Given Movie')
    merged_df.drop(columns=['genres', 'title', 'cluster'], inplace=True)
    merged_df1 = merged_df.merge(df_rating, on='movieId')

    # Save the result to CSV
    merged_df1.to_csv('output1.csv', index=False)

    # Drop timestamp column
    merged_df1.drop(columns=['timestamp'], inplace=True)

    # Group data by movieId
    grouped_data = merged_df1.groupby('movieId').agg({
        'Given Movie': 'first',
        'Predicted Movie 1': 'first',
        'Predicted Movie 2': 'first',
        'Predicted Movie 3': 'first',
        'Predicted Movie 4': 'first',
        'Predicted Movie 5': 'first',
        'userId': lambda x: list(x),
        'rating': lambda x: list(x)
    }).reset_index()

    # Convert predicted movie columns to strings
    for i in range(1, 6):
        grouped_data[f'Predicted Movie {i}'] = grouped_data[f'Predicted Movie {i}'].astype(str)

    # Map predicted movies to their respective IDs and drop columns
    for i in range(1, 6):
        grouped_data[f'MovieId{i}'] = grouped_data[f'Predicted Movie {i}'].apply(lambda x: map_title_to_id(x, new_movies_df))
        grouped_data.drop(columns=[f'Predicted Movie {i}'], inplace=True)

    final_percentage_list = []

    relevant_ratings = df_rating[df_rating['movieId'].isin(grouped_data[[f'MovieId{i}' for i in range(1, 6)]].values.flatten())]

    # Calculate percentage of users with ratings within threshold
    for index, row in grouped_data.iterrows():
        cnt1 = cnt2 = 0
        user_ratings = zip(row["userId"], row["rating"])

        for user_id, rating in user_ratings:
            for i in range(1, 6):
                movie_id = row[f"MovieId{i}"]
                movie_rating = relevant_ratings[(relevant_ratings['userId'] == user_id) & (relevant_ratings['movieId'] == movie_id)]
                if not movie_rating.empty:
                    cnt2 += 1
                    if abs(rating - movie_rating['rating'].values[0]) <= 0.5:
                        cnt1 += 1

        percentage_within_threshold = (cnt1 / cnt2) * 100 if cnt2 != 0 else 0
        final_percentage_list.append(percentage_within_threshold)
        print(index, percentage_within_threshold)

    print("Final percentage for all 30 movies:", final_percentage_list)

if __name__ == "__main__":
    main()
