import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

# Fetch MovieLens dataset with movies rated 4.0 or higher
data = fetch_movielens(min_rating=4.0)

# Create and configure the model using the WARP loss function
model = LightFM(loss='warp')

# Train the model
model.fit(data['train'], epochs=30, num_threads=2)

def sampleRecommendation(model, data, userIds):
    nUsers, nItems = data['train'].shape

    # Generate recommendations for each user
    for userId in userIds:
        knownPositives = data['item_labels'][data['train'].tocsr()[userId].indices]
        scores = model.predict(userId, np.arange(nItems))
        topItems = data['item_labels'][np.argsort(-scores)]

        print(f"User {userId}")
        print("     Known Positives:")
        for x in knownPositives[:3]:
            print(f"         {x}")

        print("     Recommended:")
        for x in topItems[:3]:
            print(f"         {x}")

# Example recommendations for selected users
sampleRecommendation(model, data, [3, 25, 450])
