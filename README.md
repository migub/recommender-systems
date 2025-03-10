# content based collaborative filtering recommender-systems 

# Music Recommendation System

## Project Overview
This part of the project implements a sophisticated music recommendation system developed for the DSG17 Online Phase competition on Kaggle. The system combines multiple recommendation approaches to provide accurate music suggestions while addressing both competition-specific requirements and real-world production needs.

## Features
- **Content-Based Filtering**: Utilizes track features including:
  - Genre analysis
  - Artist similarity
  - Release year
  - Track duration
  - Popularity metrics

- **Enhanced SVD Implementation**:
  - Hyperparameter tuning
  - Cross-validation
  - Optimized performance metrics

- **Production-Ready Features**:
  - Memory-efficient processing
  - Batch processing capabilities
  - Caching system
  - Comprehensive error handling
  - Detailed logging

## System Architecture

### Core Components
1. **ContentBasedRecommender.py**
   - Main recommendation engine
   - Feature extraction and processing
   - Similarity computation
   - Rating prediction

2. **OptimizedContentBasedRecommender.py**
   - Enhanced version with memory optimization
   - Batch processing implementation
   - Production-ready features

3. **Data Processing**
   - Efficient data loading
   - Feature extraction
   - Preprocessing utilities

## Technical Specifications

### Feature Weights
```python
weights = {
    'genre': 8.0,     # Genre importance
    'artist': 6.0,    # Artist similarity
    'year': 3.0,      # Temporal relevance
    'duration': 1.5,  # Duration impact
    'popularity': 5.0  # Popularity influence
}
```

### Performance Metrics
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- Precision and Recall
- F1 Score
- Coverage Ratio
- Popularity Bias

## Installation

### Prerequisites
```bash
python >= 3.7
numpy
pandas
scikit-learn
surprise
tqdm
joblib
```

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd music-recommendation-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your data:
   - Place your training data in `train.csv`
   - Ensure the data has the required columns:
     - user_id
     - media_id
     - is_listened
     - genre_id
     - artist_id
     - release_date
     - media_duration

## Usage

### Basic Usage
```python
from ContentBasedRecommender import MusicData, ContentBasedRecommender

# Load data
music_data = MusicData(filepath='train.csv')
data = music_data.load_music_data()

# Initialize and train recommender
recommender = ContentBasedRecommender(music_data)
recommender.fit(trainset)

# Get recommendations
recommendations = recommender.get_top_n_recommendations(user_id, n=10)
```


# Train and evaluate
recommender.fit(trainset)
metrics = recommender.evaluate_recommendations(testset, rankings)
```

## Performance Optimization

### Memory Efficiency
- Batch processing for large datasets
- Efficient matrix operations
- Memory-aware feature computation
- Caching system for frequent operations

### Speed Optimization
- Parallel processing capabilities
- Optimized similarity computations
- Efficient data structures
- Cached popularity scores

## Evaluation Results

The system achieves competitive performance metrics:
- High precision in top-N recommendations
- Effective cold-start handling
- Good balance between popularity and diversity
- Scalable performance on large datasets

## Competition vs Production

### Competition Mode
- Optimized for accuracy metrics
- Enhanced feature weights
- Aggressive similarity thresholds
- Competition-specific parameters

### Production Mode
- Balanced recommendation approach
- Enhanced diversity
- Scalability features
- Real-time processing capabilities

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments
- Kaggle DSG17 Online Phase competition
- Surprise library contributors
- Scientific Python community

## Contact
For any questions or feedback, please contact jiaqi.yu@stud.hslu.ch
