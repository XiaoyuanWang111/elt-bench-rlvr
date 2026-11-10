"""Reading a task's data model file."""

import pytest
import yaml

from elt_rlvr.task import read_data_models

DATA_MODEL = """\
models:
  - name: countries
    description: Each record represents a country.
    columns:
      - name: countrycode
        description: Unique identifier for the country.
      - name: lowest_value
        description: The lowest value of the indicator in 'Health: Population: Structure' from 1960 to 1965.
      - name: footnotes
        description: "Number of footnotes: 0 when there is none."
"""


def test_read_data_models_description_with_an_unquoted_colon_is_read(tmp_path):
    (tmp_path / "data_model.yaml").write_text(DATA_MODEL, encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        yaml.safe_load(DATA_MODEL)

    models = read_data_models(tmp_path, {"Countries": ["countrycode"]})

    assert [model.name for model in models] == ["countries"]
    assert models[0].columns == ("countrycode", "lowest_value", "footnotes")
    assert models[0].key == ("countrycode",)


def test_read_data_models_file_that_is_broken_elsewhere_still_fails(tmp_path):
    (tmp_path / "data_model.yaml").write_text("models:\n  - name: [unclosed\n", encoding="utf-8")

    with pytest.raises(yaml.YAMLError):
        read_data_models(tmp_path, {})
