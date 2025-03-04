import os
import csv
import sys
import re
import pandas as pd
import numpy as np
from surprise import Dataset
from surprise import Reader
from collections import defaultdict

class MusicData:
    def __init__(self):
        self.mediaID_to_name = {}
        self.name_to_mediaID = {}
        self.ratingsPath = './Dataset/train.csv'
        self._df = None
        self._genres = None
        self._artists = None
        self._years = None
        self._durations = None
        
    def _load_df(self):
        if self._df is None:
            try:
                self._df = pd.read_csv(self.ratingsPath)
                # Convert media_id to string for consistent handling
                self._df['media_id'] = self._df['media_id'].astype(int).astype(str)
            except Exception as e:
                print(f"Error loading data: {e}")
                self._df = pd.DataFrame()
        return self._df
        
    def loadMusicData(self):
        print(f"Loading music data from: {self.ratingsPath}")
        
        try:
            # Load the listening data
            df = self._load_df()
            print(f"Loaded {len(df)} listening events.")
            
            # Convert boolean is_listened to numeric ratings
            df['rating'] = df['is_listened'].astype(int)
            
            # Convert timestamp to unix timestamp
            df['timestamp'] = pd.to_datetime(df['ts_listen']).astype(np.int64) // 10**9
            
            # Create a mapping for media IDs
            unique_media = df[['media_id']].drop_duplicates()
            for _, row in unique_media.iterrows():
                media_id = str(row['media_id'])  # Already converted to string in _load_df
                self.mediaID_to_name[media_id] = f"Media {media_id}"
                self.name_to_mediaID[f"Media {media_id}"] = media_id
            
            # Create ratings dataframe in the format Surprise expects
            ratings_df = pd.DataFrame({
                'user_id': df['user_id'].astype(int).astype(str),
                'media_id': df['media_id'],  # Already string
                'rating': df['rating'],
                'timestamp': df['timestamp']
            })
            
            # Create a Surprise reader object
            reader = Reader(rating_scale=(0, 1))
            
            # Create the Surprise dataset
            return Dataset.load_from_df(ratings_df[['user_id', 'media_id', 'rating']], reader)
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def getYears(self):
        if self._years is None:
            self._years = defaultdict(int)
            try:
                df = self._load_df()
                if 'release_date' in df.columns:
                    # Convert release_date to year
                    df['year'] = pd.to_datetime(df['release_date']).dt.year
                    
                    # Group by media_id and take first year
                    media_years = df[['media_id', 'year']].drop_duplicates('media_id')
                    
                    for _, row in media_years.iterrows():
                        self._years[row['media_id']] = int(row['year'])
                        
            except Exception as e:
                print(f"Error getting years: {e}")
        return self._years
    
    def getDurations(self):
        if self._durations is None:
            self._durations = defaultdict(float)
            try:
                df = self._load_df()
                if 'media_duration' in df.columns:
                    # Group by media_id and take first duration
                    media_durations = df[['media_id', 'media_duration']].drop_duplicates('media_id')
                    
                    for _, row in media_durations.iterrows():
                        self._durations[row['media_id']] = float(row['media_duration'])
                        
            except Exception as e:
                print(f"Error getting durations: {e}")
        return self._durations
    
    def getGenres(self):
        if self._genres is None:
            self._genres = defaultdict(list)
            try:
                df = self._load_df()
                if 'genre_id' in df.columns:
                    # Get unique genres
                    unique_genres = sorted(df['genre_id'].unique())
                    n_genres = len(unique_genres)
                    
                    # Create genre mapping
                    genre_to_idx = {genre: idx for idx, genre in enumerate(unique_genres)}
                    
                    # Create one-hot encoded vectors for each media item
                    media_genres = df[['media_id', 'genre_id']].drop_duplicates('media_id')
                    for _, row in media_genres.iterrows():
                        # Create one-hot vector
                        genre_vector = [0] * n_genres
                        genre_vector[genre_to_idx[row['genre_id']]] = 1
                        self._genres[row['media_id']] = genre_vector
                        
            except Exception as e:
                print(f"Error getting genres: {e}")
        return self._genres
    
    def getArtists(self):
        if self._artists is None:
            self._artists = defaultdict(str)
            try:
                df = self._load_df()
                if 'artist_id' in df.columns:
                    # Create a mapping of media_id to artist_id
                    media_artists = df[['media_id', 'artist_id']].drop_duplicates('media_id')
                    
                    for _, row in media_artists.iterrows():
                        self._artists[row['media_id']] = str(int(row['artist_id']))
                        
            except Exception as e:
                print(f"Error getting artists: {e}")
        return self._artists
    
    def getPopularityRanks(self):
        try:
            df = self._load_df()
            
            # Count listens per media item
            listen_counts = df[df['is_listened']]['media_id'].value_counts()
            
            # Convert to rankings
            rankings = defaultdict(int)
            rank = 1
            for media_id, _ in listen_counts.items():
                rankings[media_id] = rank  # media_id already string
                rank += 1
            return rankings
            
        except Exception as e:
            print(f"Error getting popularity ranks: {e}")
            return defaultdict(int)
    
    def getTrackName(self, mediaID):
        if mediaID in self.mediaID_to_name:
            return self.mediaID_to_name[mediaID]
        return f"Media ID: {mediaID}"
        
    def getTrackID(self, trackName):
        if trackName in self.name_to_mediaID:
            return self.name_to_mediaID[trackName]
        return "0"

    def getGenresForTrack(self, media_id):
        try:
            df = self._load_df()
            if 'genre_id' in df.columns and 'media_id' in df.columns:
                # Get genre for this media item
                media_genres = df[df['media_id'] == int(media_id)]['genre_id'].unique()
                return list(media_genres)
        except Exception as e:
            print(f"Error getting genres for track {media_id}: {e}")
        return []
    
    def getYearForTrack(self, media_id):
        try:
            df = self._load_df()
            if 'release_date' in df.columns and 'media_id' in df.columns:
                # Get year for this media item
                media_data = df[df['media_id'] == int(media_id)]
                if not media_data.empty:
                    year = pd.to_datetime(media_data['release_date'].iloc[0]).year
                    return int(year)
        except Exception as e:
            print(f"Error getting year for track {media_id}: {e}")
        return None
    
    def getArtistForTrack(self, media_id):
        try:
            df = self._load_df()
            if 'artist_id' in df.columns and 'media_id' in df.columns:
                # Get artist for this media item
                media_data = df[df['media_id'] == int(media_id)]
                if not media_data.empty:
                    return str(int(media_data['artist_id'].iloc[0]))
        except Exception as e:
            print(f"Error getting artist for track {media_id}: {e}")
        return None
    
    def getDurationForTrack(self, media_id):
        try:
            df = self._load_df()
            if 'media_duration' in df.columns and 'media_id' in df.columns:
                # Get duration for this media item
                media_data = df[df['media_id'] == int(media_id)]
                if not media_data.empty:
                    return float(media_data['media_duration'].iloc[0])
        except Exception as e:
            print(f"Error getting duration for track {media_id}: {e}")
        return None 