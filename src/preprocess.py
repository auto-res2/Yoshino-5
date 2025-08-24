import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def inject_noise(y, noise_level=0.1):
    """Inject noise by flipping a percentage of binary labels."""
    y_noisy = y.copy()
    n_noisy = int(len(y) * noise_level)
    indices = np.random.choice(len(y), n_noisy, replace=False)
    y_noisy[indices] = 1 - y_noisy[indices]
    return y_noisy

def generate_synthetic_data(n_samples=1000, n_features=20, n_informative=15, 
                          n_redundant=5, test_size=0.2, random_state=42):
    """Generate synthetic classification dataset for experiments."""
    print("Generating synthetic dataset...")
    X, y = make_classification(
        n_samples=n_samples, 
        n_features=n_features, 
        n_informative=n_informative,
        n_redundant=n_redundant, 
        random_state=random_state
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Dataset generated: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    return X_train, X_test, y_train, y_test

def preprocess_text_data():
    """Generate sample text data for knowledge cards experiment."""
    texts = [
        "Stocks soar as market sentiment improves",
        "Economic downturn predicted by experts", 
        "New regulations affect financial markets",
        "Technology sector shows strong growth",
        "Interest rates remain stable this quarter"
    ]
    print(f"Generated {len(texts)} text samples for knowledge cards experiment")
    return texts
