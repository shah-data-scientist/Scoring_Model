"""
Sampling strategies for handling class imbalance in credit scoring.

Provides different methods to handle the imbalanced target variable:
- Balanced class weights (no resampling)
- SMOTE (oversampling)
- Random undersampling
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def apply_balanced_weights(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """
    No resampling - just return data as-is.
    Model will use class_weight='balanced' parameter.

    Args:
        X: Feature DataFrame
        y: Target Series

    Returns:
        Tuple of (X, y, metadata dict)
    """
    metadata = {
        'method': 'balanced',
        'original_samples': len(X),
        'resampled_samples': len(X),
        'minority_class_count': (y == 1).sum(),
        'majority_class_count': (y == 0).sum(),
        'imbalance_ratio': (y == 0).sum() / (y == 1).sum()
    }

    print(f"Balanced weights strategy:")
    print(f"  Total samples: {len(X):,}")
    print(f"  Class 0 (majority): {metadata['majority_class_count']:,}")
    print(f"  Class 1 (minority): {metadata['minority_class_count']:,}")
    print(f"  Imbalance ratio: {metadata['imbalance_ratio']:.2f}:1")

    return X, y, metadata


def apply_smote(X: pd.DataFrame, y: pd.Series,
                random_state: int = 42,
                sampling_strategy: str = 'auto',
                k_neighbors: int = 5) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique).

    Args:
        X: Feature DataFrame
        y: Target Series
        random_state: Random seed
        sampling_strategy: 'auto' or float (desired ratio minority/majority)
        k_neighbors: Number of nearest neighbors for SMOTE

    Returns:
        Tuple of (X_resampled, y_resampled, metadata dict)
    """
    print(f"Applying SMOTE oversampling...")
    print(f"  Original samples: {len(X):,}")
    print(f"  Original class 0: {(y == 0).sum():,}")
    print(f"  Original class 1: {(y == 1).sum():,}")

    # Initialize SMOTE
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        k_neighbors=k_neighbors
        # n_jobs parameter not supported in this version
    )

    # Apply SMOTE
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Convert back to DataFrame/Series
    X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    y_resampled = pd.Series(y_resampled, name=y.name)

    metadata = {
        'method': 'smote',
        'original_samples': len(X),
        'resampled_samples': len(X_resampled),
        'minority_class_count': (y_resampled == 1).sum(),
        'majority_class_count': (y_resampled == 0).sum(),
        'imbalance_ratio': (y_resampled == 0).sum() / (y_resampled == 1).sum(),
        'synthetic_samples_created': len(X_resampled) - len(X),
        'k_neighbors': k_neighbors
    }

    print(f"  Resampled samples: {len(X_resampled):,}")
    print(f"  Resampled class 0: {metadata['majority_class_count']:,}")
    print(f"  Resampled class 1: {metadata['minority_class_count']:,}")
    print(f"  Synthetic samples created: {metadata['synthetic_samples_created']:,}")
    print(f"  New imbalance ratio: {metadata['imbalance_ratio']:.2f}:1")

    return X_resampled, y_resampled, metadata


def apply_undersampling(X: pd.DataFrame, y: pd.Series,
                       random_state: int = 42,
                       sampling_strategy: str = 'auto') -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """
    Apply random undersampling to majority class.

    Args:
        X: Feature DataFrame
        y: Target Series
        random_state: Random seed
        sampling_strategy: 'auto' or float (desired ratio minority/majority)

    Returns:
        Tuple of (X_resampled, y_resampled, metadata dict)
    """
    print(f"Applying random undersampling...")
    print(f"  Original samples: {len(X):,}")
    print(f"  Original class 0: {(y == 0).sum():,}")
    print(f"  Original class 1: {(y == 1).sum():,}")

    # Initialize RandomUnderSampler
    rus = RandomUnderSampler(
        sampling_strategy=sampling_strategy,
        random_state=random_state
    )

    # Apply undersampling
    X_resampled, y_resampled = rus.fit_resample(X, y)

    # Convert back to DataFrame/Series
    X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    y_resampled = pd.Series(y_resampled, name=y.name)

    metadata = {
        'method': 'undersample',
        'original_samples': len(X),
        'resampled_samples': len(X_resampled),
        'minority_class_count': (y_resampled == 1).sum(),
        'majority_class_count': (y_resampled == 0).sum(),
        'imbalance_ratio': (y_resampled == 0).sum() / (y_resampled == 1).sum(),
        'samples_removed': len(X) - len(X_resampled)
    }

    print(f"  Resampled samples: {len(X_resampled):,}")
    print(f"  Resampled class 0: {metadata['majority_class_count']:,}")
    print(f"  Resampled class 1: {metadata['minority_class_count']:,}")
    print(f"  Samples removed: {metadata['samples_removed']:,}")
    print(f"  New imbalance ratio: {metadata['imbalance_ratio']:.2f}:1")

    return X_resampled, y_resampled, metadata


from imblearn.pipeline import Pipeline

def apply_smote_undersample(X: pd.DataFrame, y: pd.Series,
                           random_state: int = 42,
                           over_strategy: float = 0.1, 
                           under_strategy: float = 0.5) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """
    Apply SMOTE followed by Random Undersampling.
    
    Strategy:
    1. SMOTE oversamples minority class to 'over_strategy' ratio of majority class.
    2. RandomUnderSampler undersamples majority class to 'under_strategy' ratio (target/majority).
    
    Args:
        X: Feature DataFrame
        y: Target Series
        random_state: Random seed
        over_strategy: Ratio of minority/majority after SMOTE (e.g. 0.1 to increase minority slightly)
        under_strategy: Ratio of minority/majority after Undersampling (e.g. 0.5 for 2:1 ratio)

    Returns:
        Tuple of (X_resampled, y_resampled, metadata dict)
    """
    print(f"Applying SMOTE + Undersampling...")
    print(f"  Original samples: {len(X):,}")
    print(f"  Original class 0: {(y == 0).sum():,}")
    print(f"  Original class 1: {(y == 1).sum():,}")

    # Pipeline: SMOTE -> RandomUnderSampler
    # Note: sampling_strategy in SMOTE controls the ratio of minority to majority AFTER resampling
    # We want to first boost minority (e.g. to 0.3), then trim majority to match (e.g. to 0.5)
    
    # Default behavior if not specified: 
    # SMOTE to 0.5 (minority becomes half of majority)
    # Then Undersample to 1.0 (majority becomes equal to minority)
    
    over = SMOTE(sampling_strategy=0.5, random_state=random_state)
    under = RandomUnderSampler(sampling_strategy=1.0, random_state=random_state)
    
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)

    X_resampled, y_resampled = pipeline.fit_resample(X, y)

    # Convert back to DataFrame/Series
    X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    y_resampled = pd.Series(y_resampled, name=y.name)

    metadata = {
        'method': 'smote_undersample',
        'original_samples': len(X),
        'resampled_samples': len(X_resampled),
        'minority_class_count': (y_resampled == 1).sum(),
        'majority_class_count': (y_resampled == 0).sum(),
        'imbalance_ratio': (y_resampled == 0).sum() / (y_resampled == 1).sum()
    }

    print(f"  Resampled samples: {len(X_resampled):,}")
    print(f"  Resampled class 0: {metadata['majority_class_count']:,}")
    print(f"  Resampled class 1: {metadata['minority_class_count']:,}")
    print(f"  New imbalance ratio: {metadata['imbalance_ratio']:.2f}:1")

    return X_resampled, y_resampled, metadata


def get_sampling_strategy(strategy_name: str, X: pd.DataFrame, y: pd.Series,
                          random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """
    Apply the specified sampling strategy.

    Args:
        strategy_name: 'balanced', 'smote', 'undersample', or 'smote_undersample'
        X: Feature DataFrame
        y: Target Series
        random_state: Random seed

    Returns:
        Tuple of (X_resampled, y_resampled, metadata dict)

    Raises:
        ValueError: If strategy_name is not recognized
    """
    strategy_name = strategy_name.lower()

    if strategy_name == 'balanced':
        return apply_balanced_weights(X, y)
    elif strategy_name == 'smote':
        return apply_smote(X, y, random_state=random_state)
    elif strategy_name == 'undersample':
        return apply_undersampling(X, y, random_state=random_state)
    elif strategy_name == 'smote_undersample':
        return apply_smote_undersample(X, y, random_state=random_state)
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy_name}. "
                        f"Choose from: balanced, smote, undersample, smote_undersample")


# Strategy descriptions for documentation
STRATEGY_DESCRIPTIONS = {
    'balanced': {
        'name': 'Balanced Class Weights',
        'description': 'No resampling. Uses class_weight="balanced" in model.',
        'pros': ['Fast', 'No data modification', 'Works with all sklearn models'],
        'cons': ['Does not create new samples', 'May not work as well as resampling']
    },
    'smote': {
        'name': 'SMOTE Oversampling',
        'description': 'Creates synthetic minority class samples using k-NN.',
        'pros': ['Proven effective', 'Maintains all original data', 'Reduces overfitting'],
        'cons': ['Computationally expensive', 'May create unrealistic samples', 'Increases dataset size']
    },
    'undersample': {
        'name': 'Random Undersampling',
        'description': 'Randomly removes majority class samples.',
        'pros': ['Fast', 'Reduces dataset size', 'Simple to understand'],
        'cons': ['Loses information', 'May remove important samples', 'Reduces training data']
    },
    'smote_undersample': {
        'name': 'SMOTE + Undersampling',
        'description': 'Hybrid approach: SMOTE to increase minority, then undersample majority.',
        'pros': ['Balances data without massive size increase', 'Cleans decision boundary'],
        'cons': ['Complex to tune', 'Computationally intensive']
    }
}


def print_strategy_info(strategy_name: str):
    """Print information about a sampling strategy."""
    if strategy_name in STRATEGY_DESCRIPTIONS:
        info = STRATEGY_DESCRIPTIONS[strategy_name]
        print(f"\n{info['name']}")
        print("=" * 80)
        print(f"Description: {info['description']}")
        print("\nPros:")
        for pro in info['pros']:
            print(f"  + {pro}")
        print("\nCons:")
        for con in info['cons']:
            print(f"  - {con}")
    else:
        print(f"Unknown strategy: {strategy_name}")


if __name__ == "__main__":
    # Example usage
    print("Sampling Strategies Module")
    print("=" * 80)

    # Create sample imbalanced data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    # Create imbalanced dataset (90% class 0, 10% class 1)
    X = pd.DataFrame(
        np.random.rand(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1]), name='target')

    print(f"\nOriginal dataset:")
    print(f"  Samples: {len(X):,}")
    print(f"  Class 0: {(y == 0).sum():,}")
    print(f"  Class 1: {(y == 1).sum():,}")
    print(f"  Imbalance ratio: {(y == 0).sum() / (y == 1).sum():.2f}:1")

    # Test each strategy
    for strategy in ['balanced', 'smote', 'undersample']:
        print(f"\n{'-' * 80}")
        print_strategy_info(strategy)
        print(f"\n{'-' * 80}")

        X_resampled, y_resampled, metadata = get_sampling_strategy(strategy, X, y)

        print(f"\nResult:")
        print(f"  Samples: {len(X_resampled):,}")
        print(f"  Class 0: {(y_resampled == 0).sum():,}")
        print(f"  Class 1: {(y_resampled == 1).sum():,}")
