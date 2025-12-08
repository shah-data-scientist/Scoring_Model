"""
Tests for validation module.

Run with: poetry run pytest tests/test_validation.py -v
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.validation import (
    validate_file_exists,
    validate_dataframe_schema,
    validate_id_column,
    validate_target_column,
    validate_prediction_probabilities,
    validate_feature_names_match,
    DataValidationError,
    SchemaValidationError
)


class TestFileValidation:
    """Tests for file validation."""

    def test_validate_file_exists_valid(self, tmp_path):
        """Test that valid file passes validation."""
        test_file = tmp_path / "test.csv"
        test_file.write_text("col1,col2\n1,2\n")

        # Should not raise
        validate_file_exists(test_file, "Test file")

    def test_validate_file_exists_missing(self, tmp_path):
        """Test that missing file raises FileNotFoundError."""
        test_file = tmp_path / "missing.csv"

        with pytest.raises(FileNotFoundError, match="Test file not found"):
            validate_file_exists(test_file, "Test file")

    def test_validate_file_exists_directory(self, tmp_path):
        """Test that directory raises ValueError."""
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()

        with pytest.raises(ValueError, match="not a file"):
            validate_file_exists(test_dir, "Test file")


class TestDataFrameSchemaValidation:
    """Tests for DataFrame schema validation."""

    def test_validate_schema_valid(self):
        """Test that DataFrame with all required columns passes."""
        df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        required = ['col1', 'col2']

        # Should not raise
        validate_dataframe_schema(df, required)

    def test_validate_schema_missing_columns(self):
        """Test that missing required columns raises SchemaValidationError."""
        df = pd.DataFrame({'col1': [1, 2]})
        required = ['col1', 'col2', 'col3']

        with pytest.raises(SchemaValidationError, match="missing required columns"):
            validate_dataframe_schema(df, required)

    def test_validate_schema_empty_dataframe(self):
        """Test that empty DataFrame raises DataValidationError."""
        df = pd.DataFrame()
        required = ['col1']

        with pytest.raises(DataValidationError, match="empty or None"):
            validate_dataframe_schema(df, required)

    def test_validate_schema_with_optional(self):
        """Test validation with optional columns."""
        df = pd.DataFrame({'col1': [1, 2]})
        required = ['col1']
        optional = ['col2', 'col3']

        # Should not raise (optional columns missing is OK)
        validate_dataframe_schema(df, required, optional)


class TestIDColumnValidation:
    """Tests for ID column validation."""

    def test_validate_id_column_valid(self):
        """Test that valid ID column passes."""
        df = pd.DataFrame({'SK_ID_CURR': [1, 2, 3]})

        # Should not raise
        validate_id_column(df)

    def test_validate_id_column_missing(self):
        """Test that missing ID column raises SchemaValidationError."""
        df = pd.DataFrame({'other_col': [1, 2, 3]})

        with pytest.raises(SchemaValidationError, match="missing ID column"):
            validate_id_column(df)

    def test_validate_id_column_duplicates(self):
        """Test that duplicate IDs raise DataValidationError."""
        df = pd.DataFrame({'SK_ID_CURR': [1, 2, 2, 3]})

        with pytest.raises(DataValidationError, match="duplicate IDs"):
            validate_id_column(df)

    def test_validate_id_column_nulls(self):
        """Test that null IDs raise DataValidationError."""
        df = pd.DataFrame({'SK_ID_CURR': [1, 2, None, 3]})

        with pytest.raises(DataValidationError, match="null values"):
            validate_id_column(df)

    def test_validate_id_column_allow_duplicates(self):
        """Test that duplicates pass when allowed."""
        df = pd.DataFrame({'SK_ID_CURR': [1, 2, 2, 3]})

        # Should not raise when duplicates are allowed
        validate_id_column(df, allow_duplicates=True)


class TestTargetColumnValidation:
    """Tests for target column validation."""

    def test_validate_target_valid_binary(self):
        """Test that valid binary target passes."""
        df = pd.DataFrame({'TARGET': [0, 1, 0, 1, 1]})

        # Should not raise
        validate_target_column(df, expected_values=[0, 1])

    def test_validate_target_missing(self):
        """Test that missing target column raises SchemaValidationError."""
        df = pd.DataFrame({'other_col': [1, 2, 3]})

        with pytest.raises(SchemaValidationError, match="missing target column"):
            validate_target_column(df)

    def test_validate_target_unexpected_values(self):
        """Test that unexpected values raise DataValidationError."""
        df = pd.DataFrame({'TARGET': [0, 1, 2, 3]})

        with pytest.raises(DataValidationError, match="unexpected values"):
            validate_target_column(df, expected_values=[0, 1])

    def test_validate_target_nulls(self):
        """Test that null values raise DataValidationError."""
        df = pd.DataFrame({'TARGET': [0, 1, None, 1]})

        with pytest.raises(DataValidationError, match="null values"):
            validate_target_column(df)

    def test_validate_target_non_numeric(self):
        """Test that non-numeric target raises DataValidationError."""
        df = pd.DataFrame({'TARGET': ['a', 'b', 'c']})

        with pytest.raises(DataValidationError, match="must be numeric"):
            validate_target_column(df)


class TestPredictionValidation:
    """Tests for prediction probability validation."""

    def test_validate_probabilities_valid(self):
        """Test that valid probabilities pass."""
        probs = np.array([0.0, 0.5, 1.0, 0.25, 0.75])

        # Should not raise
        validate_prediction_probabilities(probs)

    def test_validate_probabilities_nan(self):
        """Test that NaN values raise DataValidationError."""
        probs = np.array([0.5, np.nan, 0.7])

        with pytest.raises(DataValidationError, match="NaN values"):
            validate_prediction_probabilities(probs)

    def test_validate_probabilities_inf(self):
        """Test that infinite values raise DataValidationError."""
        probs = np.array([0.5, np.inf, 0.7])

        with pytest.raises(DataValidationError, match="infinite values"):
            validate_prediction_probabilities(probs)

    def test_validate_probabilities_out_of_range(self):
        """Test that out-of-range values raise DataValidationError."""
        probs = np.array([0.5, 1.5, 0.7])

        with pytest.raises(DataValidationError, match="outside \\[0, 1\\] range"):
            validate_prediction_probabilities(probs)

    def test_validate_probabilities_negative(self):
        """Test that negative values raise DataValidationError."""
        probs = np.array([0.5, -0.1, 0.7])

        with pytest.raises(DataValidationError, match="outside \\[0, 1\\] range"):
            validate_prediction_probabilities(probs)

    def test_validate_probabilities_empty(self):
        """Test that empty array raises DataValidationError."""
        probs = np.array([])

        with pytest.raises(DataValidationError, match="empty"):
            validate_prediction_probabilities(probs)


class TestFeatureNameValidation:
    """Tests for feature name validation."""

    def test_validate_features_exact_match(self):
        """Test that exact feature match passes."""
        df = pd.DataFrame({'feat1': [1], 'feat2': [2], 'feat3': [3]})
        expected = ['feat1', 'feat2', 'feat3']

        # Should not raise
        validate_feature_names_match(df, expected)

    def test_validate_features_missing(self):
        """Test that missing features raise SchemaValidationError."""
        df = pd.DataFrame({'feat1': [1], 'feat2': [2]})
        expected = ['feat1', 'feat2', 'feat3', 'feat4']

        with pytest.raises(SchemaValidationError, match="don't match"):
            validate_feature_names_match(df, expected)

    def test_validate_features_extra(self):
        """Test that extra features raise SchemaValidationError."""
        df = pd.DataFrame({'feat1': [1], 'feat2': [2], 'feat3': [3]})
        expected = ['feat1', 'feat2']

        with pytest.raises(SchemaValidationError, match="don't match"):
            validate_feature_names_match(df, expected)

    def test_validate_features_allow_subset(self):
        """Test that subset validation works."""
        df = pd.DataFrame({'feat1': [1], 'feat2': [2]})
        expected = ['feat1', 'feat2', 'feat3', 'feat4']

        # Should not raise when allow_subset=True and df is subset
        validate_feature_names_match(df, expected, allow_subset=True)

    def test_validate_features_subset_with_extra(self):
        """Test that extra features fail even with allow_subset."""
        df = pd.DataFrame({'feat1': [1], 'feat2': [2], 'feat_extra': [3]})
        expected = ['feat1', 'feat2', 'feat3']

        with pytest.raises(SchemaValidationError, match="unexpected features"):
            validate_feature_names_match(df, expected, allow_subset=True)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
