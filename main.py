#CS7050 Final Project
#Austin Klein

import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#pathways must be changed if the data is stored in a different location
R_Path = "data/ratings.csv" #ratings pathway
I_Path = "data/movies.csv"    #movie title pathway
Clusters = 20
Top_Choice = 10

class MovieRecommender:
    def __init__(self, ratings_path=R_Path, items_path=I_Path, n_clusters=Clusters):
        self.ratings_path = ratings_path
        self.items_path = items_path
        self.n_clusters = n_clusters
        self.kmeans = None
        self.scaler = None
        self.user_item = None
        self.movie_titles = None
        self.user_cluster = None
        self.train_done = False

    def load_data(self):
        if not os.path.exists(self.ratings_path) or not os.path.exists(self.items_path):
            raise FileNotFoundError("Data files not found. Make sure ratings.csv and movies.csv are in data/ folder.")

        # reads the ratings data from the exel file
        ratings = pd.read_csv(self.ratings_path, sep=',', encoding='utf-8')
        ratings = ratings.rename(columns={'userId':'user_id','movieId':'movie_id'})

        # reads the movie title data from the excel file
        items = pd.read_csv(self.items_path, sep=',', encoding='utf-8', engine='python')
        items = items.rename(columns={'movieId':'movie_id','title':'title'})

        user_item = ratings.pivot(index='user_id', columns='movie_id', values='rating')

        # saves data
        self.raw_ratings = ratings
        self.movie_titles = items.set_index('movie_id')['title']
        self.user_item = user_item
        return True

    def preprocess_and_train(self):
        if self.user_item is None:
            self.load_data()

        user_item_filled = self.user_item.copy()
        row_means = user_item_filled.mean(axis=1)
        global_mean = self.raw_ratings['rating'].mean()
        user_item_filled = user_item_filled.apply(lambda row: row.fillna(row.mean() if not np.isnan(row.mean()) else global_mean), axis=1)

        self.movie_ids = list(user_item_filled.columns)

        scaler = StandardScaler()
        X = scaler.fit_transform(user_item_filled.values)

        # k means clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        kmeans.fit(X)

        # save work
        self.scaler = scaler
        self.kmeans = kmeans
        self.user_item_filled = user_item_filled
        self.user_cluster = pd.Series(kmeans.labels_, index=user_item_filled.index)
        self.train_done = True

    def recommend_for_liked_movies(self, liked_movie_ids, top_n=Top_Choice):
        if not self.train_done:
            raise RuntimeError("Model not trained. Call preprocess_and_train() first.")

        # creates a profile for the user to get movie recommendations
        global_mean = self.raw_ratings['rating'].mean()
        user_vec = pd.Series(index=self.movie_ids, dtype=float)
        user_vec[:] = np.nan
        for m in liked_movie_ids:
            if m in user_vec.index:
                user_vec.loc[m] = 5.0  #if they like the movie it gets a high rating
        user_vec = user_vec.fillna(global_mean)

        # scale and then predict created cluster
        X_user = self.scaler.transform([user_vec.values])
        cluster = self.kmeans.predict(X_user)[0]

        users_in_cluster = self.user_cluster[self.user_cluster == cluster].index

        # gets the average of the movie ratings for that cluster
        cluster_ratings = self.user_item_filled.loc[users_in_cluster]
        mean_ratings = cluster_ratings.mean(axis=0)

        # makes sure there are no movies that were already liked
        for m in liked_movie_ids:
            if m in mean_ratings.index:
                mean_ratings.loc[m] = -np.inf

        # reccomends the requested number of movies
        top_movie_ids = mean_ratings.sort_values(ascending=False).head(top_n).index.tolist()
        return top_movie_ids, cluster

#all the gui for the user to be able to pick the movies and get recommendations
class RecommenderGUI:
    def __init__(self, root):
        self.root = root
        root.title('MovieLens KMeans Recommender (100k)')
        self.model = MovieRecommender()
        self.liked_movie_ids = []

        frm = ttk.Frame(root, padding=10)
        frm.grid(row=0, column=0, sticky='nsew')

        # controls for the clustering, training, and progress
        top_ctrl = ttk.Frame(frm)
        top_ctrl.grid(row=0, column=0, sticky='w')
        ttk.Label(top_ctrl, text='# clusters:').grid(row=0, column=0, padx=2)
        self.clusters_var = tk.IntVar(value=Top_Choice)
        ttk.Entry(top_ctrl, textvariable=self.clusters_var, width=6).grid(row=0, column=1, padx=2)
        self.train_btn = ttk.Button(top_ctrl, text='Train Model', command=self.train_model)
        self.train_btn.grid(row=0, column=2, padx=6)
        self.status = ttk.Label(top_ctrl, text='Not trained', foreground='red')
        self.status.grid(row=0, column=3, padx=10)

        # setion where you can search through the movies and add the ones you like
        search_frame = ttk.Frame(frm)
        search_frame.grid(row=1, column=0, pady=(10,0), sticky='w')
        ttk.Label(search_frame, text='Search movie:').grid(row=0, column=0, padx=2)
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=50)
        self.search_entry.grid(row=0, column=1, padx=4)
        self.search_btn = ttk.Button(search_frame, text='Search', command=self.search_movies)
        self.search_btn.grid(row=0, column=2, padx=4)
        self.add_btn = ttk.Button(search_frame, text='Add liked', command=self.add_liked)
        self.add_btn.grid(row=0, column=3, padx=4)

        # search results 
        self.results_listbox = tk.Listbox(frm, height=8, width=80)
        self.results_listbox.grid(row=2, column=0, pady=6)

        # liked movies list
        ttk.Label(frm, text='Movies you liked:').grid(row=3, column=0, sticky='w')
        self.liked_listbox = tk.Listbox(frm, height=6, width=80)
        self.liked_listbox.grid(row=4, column=0, pady=6)
        self.remove_liked_btn = ttk.Button(frm, text='Remove selected liked', command=self.remove_liked)
        self.remove_liked_btn.grid(row=5, column=0, sticky='w')

        # controls for reccomending movies
        rec_frame = ttk.Frame(frm)
        rec_frame.grid(row=6, column=0, pady=(10,0), sticky='w')
        ttk.Label(rec_frame, text='Top N:').grid(row=0, column=0)
        self.topn_var = tk.IntVar(value=Top_Choice)
        ttk.Entry(rec_frame, textvariable=self.topn_var, width=6).grid(row=0, column=1, padx=4)
        self.recommend_btn = ttk.Button(rec_frame, text='Recommend', command=self.recommend)
        self.recommend_btn.grid(row=0, column=2, padx=6)

        # recommendations list
        ttk.Label(frm, text='Recommendations:').grid(row=7, column=0, sticky='w')
        self.recs_listbox = tk.Listbox(frm, height=10, width=80)
        self.recs_listbox.grid(row=8, column=0, pady=6)

    def train_model(self):
        try:
            n = int(self.clusters_var.get())
            if n <= 0:
                raise ValueError
            self.model.n_clusters = n
        except Exception:
            messagebox.showerror('Error', 'Number of clusters must be a positive integer')
            return

        self.status.config(text='Training...', foreground='orange')
        self.train_btn.config(state='disabled')
        t = threading.Thread(target=self._train_background, daemon=True)
        t.start()

    def _train_background(self):
        try:
            self.model.load_data()
            self.model.preprocess_and_train()
            self.status.config(text='Trained', foreground='green')
            self.update_results_listbox()
        except Exception as e:
            self.status.config(text='Error', foreground='red')
            messagebox.showerror('Training error', str(e))
        finally:
            self.train_btn.config(state='normal')

    def update_results_listbox(self, query=None):
        self.results_listbox.delete(0, tk.END)
        if self.model.movie_titles is None:
            return
        titles = self.model.movie_titles
        if query:
            matches = titles[titles.str.contains(query, case=False, na=False)]
        else:
            matches = titles
        for mid, title in matches.head(200).items():
            self.results_listbox.insert(tk.END, f"{mid}: {title}")

    def search_movies(self):
        q = self.search_var.get().strip()
        if not q:
            self.update_results_listbox()
            return
        self.update_results_listbox(query=q)

    def add_liked(self):
        sel = self.results_listbox.curselection()
        if not sel:
            messagebox.showinfo('Info', 'Please select a movie from the search results to add')
            return
        text = self.results_listbox.get(sel[0])
        mid = int(text.split(':', 1)[0])
        if mid in self.liked_movie_ids:
            messagebox.showinfo('Info', 'Already added')
            return
        self.liked_movie_ids.append(mid)
        title = self.model.movie_titles.get(mid, 'Unknown')
        self.liked_listbox.insert(tk.END, f"{mid}: {title}")

    def remove_liked(self):
        sel = self.liked_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        self.liked_listbox.delete(idx)
        del self.liked_movie_ids[idx]

    def recommend(self):
        if not self.model.train_done:
            messagebox.showerror('Error', 'Model is not trained yet. Click Train Model first.')
            return
        if len(self.liked_movie_ids) == 0:
            messagebox.showinfo('Info', 'Add at least one movie you liked')
            return
        top_n = int(self.topn_var.get())
        try:
            rec_ids, cluster = self.model.recommend_for_liked_movies(self.liked_movie_ids, top_n)
        except Exception as e:
            messagebox.showerror('Error', str(e))
            return
        self.recs_listbox.delete(0, tk.END)
        self.recs_listbox.insert(tk.END, f"Predicted cluster: {cluster}")
        for rid in rec_ids:
            title = self.model.movie_titles.get(rid, 'Unknown')
            self.recs_listbox.insert(tk.END, f"{rid}: {title}")


if __name__ == '__main__':
    root = tk.Tk()
    gui = RecommenderGUI(root)
    root.mainloop()
