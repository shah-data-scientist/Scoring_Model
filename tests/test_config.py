"""
Tests for configuration module.

Run with: poetry run pytest tests/test_config.py -v
"""
import pytest
from pathlib import Path
import tempfile
import yaml

from src.config import (
    load_config,
    get_data_path,
    get_mlflow_uri,
    get_random_state,
    get_baseline_tags,
    get_optimization_tags,
    get_production_tags,
    get_artifact_path,
    CONFIG
)


class TestConfigLoading:
    """Tests for configuration loading."""

    def test_config_loads_successfully(self):
        """Test that default config loads without errors."""
        config = load_config()
        assert isinstance(config, dict)

    def test_config_has_required_keys(self):
        """Test that config has all required sections."""
        assert 'project' in CONFIG
        assert 'paths' in CONFIG
        assert 'mlflow' in CONFIG
        assert 'business' in CONFIG
        assert 'model' in CONFIG

    def test_config_project_settings(self):
        """Test project settings are valid."""
        assert 'random_state' in CONFIG['project']
        assert isinstance(CONFIG['project']['random_state'], int)
        assert CONFIG['project']['random_state'] >= 0

        assert 'n_folds' in CONFIG['project']
        assert CONFIG['project']['n_folds'] > 1

    def test_config_business_settings(self):
        """Test business settings are valid."""
        assert 'cost_fn' in CONFIG['business']
        assert 'cost_fp' in CONFIG['business']
        assert CONFIG['business']['cost_fn'] > 0
        assert CONFIG['business']['cost_fp'] > 0


class TestConfigAccessors:
    """Tests for config accessor functions."""

    def test_get_data_path(self):
        """Test get_data_path returns Path object."""
        data_path = get_data_path()
        assert isinstance(data_path, Path)

    def test_get_mlflow_uri(self):
        """Test get_mlflow_uri returns string."""
        uri = get_mlflow_uri()
        assert isinstance(uri, str)
        assert 'sqlite:///' in uri or 'http' in uri

    def test_get_random_state(self):
        """Test get_random_state returns integer."""
        random_state = get_random_state()
        assert isinstance(random_state, int)
        assert random_state >= 0


class TestMLflowTagFunctions:
    """Tests for MLflow tag generation functions."""

    def test_get_baseline_tags(self):
        """Test baseline tags generation."""
        tags = get_baseline_tags('lgbm')

        assert isinstance(tags, dict)
        assert tags['stage'] == 'baseline'
        assert tags['model'] == 'lgbm'
        assert 'project' in tags

    def test_get_baseline_tags_with_kwargs(self):
        """Test baseline tags with additional kwargs."""
        tags = get_baseline_tags('lgbm', custom_tag='value', iteration=1)

        assert tags['model'] == 'lgbm'
        assert tags['custom_tag'] == 'value'
        assert tags['iteration'] == 1

    def test_get_optimization_tags(self):
        """Test optimization tags generation."""
        tags = get_optimization_tags('xgboost', optimization_type='grid_search')

        assert tags['stage'] == 'optimization'
        assert tags['model'] == 'xgboost'
        assert tags['optimization'] == 'grid_search'

    def test_get_production_tags(self):
        """Test production tags generation."""
        tags = get_production_tags('random_forest', version='2.0')

        assert tags['stage'] == 'production'
        assert tags['model'] == 'random_forest'
        assert tags['version'] == '2.0'

    def test_get_artifact_path(self):
        """Test artifact path generation."""
        path = get_artifact_path('lgbm', 'plot')

        assert isinstance(path, str)
        assert 'lgbm' in path
        assert 'plot' in path


class TestConfigErrorHandling:
    """Tests for config error handling."""

    def test_load_config_missing_file(self, tmp_path):
        """Test that missing config file is handled gracefully."""
        with pytest.raises(FileNotFoundError):
            load_config(str(tmp_path / "nonexistent.yaml"))

    def test_load_config_malformed_yaml(self, tmp_path):
        """Test that malformed YAML is handled."""
        bad_config = tmp_path / "bad_config.yaml"
        bad_config.write_text("invalid: yaml: syntax: [")

        with pytest.raises(yaml.YAMLError):
            load_config(str(bad_config))

    def test_load_config_empty_file(self, tmp_path):
        """Test that empty config file returns empty dict."""
        empty_config = tmp_path / "empty.yaml"
        empty_config.write_text("")

        config = load_config(str(empty_config))
        assert config is None or config == {}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
