# Movie Recommendation System

A hybrid recommendation system built with LightFM that combines collaborative filtering 
and content-based features to predict movie preferences. Trained on the MovieLens dataset 
with 100,000+ ratings from 1,700+ users.

## Overview

This system uses the WARP (Weighted Approximate-Rank Pairwise) loss function to optimize 
for ranking quality in recommendation scenarios. The hybrid approach addresses the cold-start 
problem by leveraging both user-item interactions and movie metadata.

## Features

- Hybrid recommendation combining collaborative filtering + content-based features
- Systematic comparison of 3 loss functions (WARP, BPR, Logistic)
- Regularization to prevent overfitting on sparse data (95%+ sparsity)
- Sub-second inference for real-time recommendations
- Multi-threaded training for improved performance

## Technologies Used

- **Python**
- **LightFM** - Hybrid recommendation library
- **NumPy** - Numerical operations and array handling
- **SciPy** - Sparse matrix operations
- **MovieLens Dataset** - 1,700+ users, 100,000+ ratings

## Installation
```bash
pip install lightfm numpy scipy
```

## Usage
```bash
python movieRecommender.py
```

## Model Performance Comparison

| Loss Function | Test Precision@10 | Test AUC | Notes |
|---------------|------------------|----------|-------|
| **WARP** | 6.39% | **0.909** | Best overall ranking quality |
| BPR | 4.62% | 0.825 | Weakest performance |
| Logistic | **6.95%** | 0.867 | Best top-10 precision |

### Model Selection Analysis

**Selected: WARP**

While Logistic achieved slightly higher precision@10 (6.95% vs 6.39%), WARP was 
selected based on superior AUC score (0.909 vs 0.867). Here's why:

- **AUC measures overall ranking quality** across all items, not just top-10
- **WARP optimizes for ranking** through pairwise comparisons, making it more 
  robust for recommendation scenarios where rank matters more than absolute scores
- **Better generalization** - WARP's 10% improvement over BPR (0.909 vs 0.825 AUC) 
  demonstrates it learns more meaningful patterns in sparse data

**Business Context**: For a production system, WARP would be preferred because users 
browse through more than just the top 10 recommendations. Strong AUC ensures quality 
rankings throughout the entire catalog.

## Implementation Details

### Model Architecture
- **Hybrid approach**: Collaborative filtering + content-based features
- **Loss function**: WARP (Weighted Approximate-Rank Pairwise)
- **Latent components**: 20 (reduced to prevent overfitting)
- **Regularization**: L2 penalties (alpha=0.001) on user/item embeddings
- **Training epochs**: 10 (optimized to balance performance and overfitting)

### Data Processing
- **Dataset**: MovieLens 100K
- **Train-test split**: 80/20 with random state for reproducibility
- **Sparsity**: 95%+ (typical of recommendation systems)
- **Minimum rating**: 4.0 (focusing on positive interactions)

### Evaluation Metrics
- **Precision@10**: Measures accuracy of top-10 recommendations
- **AUC (Area Under Curve)**: Measures overall ranking quality across entire catalog
- **Train-test validation**: Ensures model generalizes to unseen interactions

## Key Findings

1. **WARP outperforms alternatives** - 10% AUC improvement over BPR
2. **Overfitting is common** - Regularization essential for sparse recommendation data
3. **Metric tradeoffs exist** - Different loss functions optimize for different objectives
4. **Hybrid approach helps** - Content features reduce cold-start problem

## Future Improvements

- [ ] Implement cross-validation for more robust evaluation
- [ ] Add item features (genres, directors, actors) for better content-based recommendations
- [ ] Experiment with deep learning models (Neural Collaborative Filtering)
- [ ] Deploy as REST API with Flask for real-time predictions
- [ ] Add A/B testing framework for production deployment

## Sample Output
```
User 3
     Known positives:
        Star Wars (1977)
        Empire Strikes Back (1980)
        Return of the Jedi (1983)
     Recommended:
        Raiders of the Lost Ark (1981)
        Indiana Jones and the Last Crusade (1989)
        Back to the Future (1985)
```

## References

- [LightFM Documentation](https://making.lyst.com/lightfm/docs/home.html)
- [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
- [WARP Loss Paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37180.pdf)

## License

MIT

## Author

Nicholas Chang - [GitHub](https://github.com/nchxng) | [LinkedIn](https://linkedin.com/in/nicholas8chang)
