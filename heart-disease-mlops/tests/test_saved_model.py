"""Check the saved Step 6 model without fitting or changing any artifacts."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from heart_disease import predict as prediction_module
from heart_disease.feature_engineering import EXPECTED_COLUMNS, project_root


def test_saved_model_predicts_without_refitting_preprocessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load the packaged winner and test one real-schema demo feature row."""
    root = project_root()
    artifact_path = root / "artifacts" / "final_heart_disease_pipeline.joblib"
    metadata_path = root / "artifacts" / "final_heart_disease_metadata.json"
    dataset_path = root / "data" / "processed" / "heart_disease_cleaned.csv"

    missing_files = [
        str(path)
        for path in (artifact_path, metadata_path, dataset_path)
        if not path.is_file()
    ]
    if missing_files:
        pytest.skip(
            "The saved-model integration test needs local generated model/data "
            f"files, which are Git-ignored and unavailable here: {missing_files}"
        )

    model = joblib.load(artifact_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert isinstance(model, Pipeline), (
        "The saved artifact must be the complete scikit-learn pipeline."
    )
    assert list(model.feature_names_in_) == EXPECTED_COLUMNS[:-1], (
        "The saved model feature names and order must match the verified "
        "Step 3 schema, excluding the target column."
    )
    assert metadata["required_features_in_order"] == EXPECTED_COLUMNS[:-1], (
        "The metadata feature schema must match the saved pipeline schema."
    )
    assert metadata["target_encoding"]["source_column"] == "num", (
        "Model metadata must identify the project's original UCI target column."
    )
    assert metadata["target_encoding"]["source_values"] == {
        "0": "absence",
        "1-4": "presence",
    }, "Model metadata must preserve the documented original target encoding."
    assert metadata["target_encoding"]["model_classes"] == {
        "0": "absence",
        "1": "presence",
    }, "Model metadata must state the actual binary classes used for prediction."
    assert {"preprocessing", "classifier"}.issubset(model.named_steps), (
        "The saved pipeline must keep preprocessing and the classifier together."
    )

    feature_names = list(model.feature_names_in_)
    demo_features = pd.read_csv(dataset_path, usecols=feature_names).iloc[[0]]
    assert demo_features.columns.tolist() == feature_names, (
        "The demonstration row must contain exactly the saved model's feature "
        "names in the expected order."
    )
    example = demo_features.iloc[0].to_dict()
    expected_class = model.predict(demo_features)

    assert expected_class.shape == (1,), (
        "One demonstration input row must produce exactly one prediction."
    )
    assert expected_class[0] in model.classes_, (
        "The predicted class must be one of the fitted model's valid classes."
    )

    preprocessing = model.named_steps["preprocessing"]
    assert hasattr(preprocessing, "transformers_"), (
        "The saved preprocessing pipeline must already be fitted."
    )

    def fail_if_fit_is_called(*_args: object, **_kwargs: object) -> None:
        pytest.fail(
            "Prediction must use the saved preprocessing state; it must not "
            "fit or refit any preprocessing step."
        )

    fitted_preprocessing_steps = [preprocessing]
    fitted_preprocessing_steps.extend(
        preprocessing.get_params(deep=True).values()
    )
    patched_step_ids: set[int] = set()
    for step in fitted_preprocessing_steps:
        if id(step) in patched_step_ids:
            continue
        patched_step_ids.add(id(step))
        for method_name in ("fit", "fit_transform"):
            if callable(getattr(step, method_name, None)):
                monkeypatch.setattr(step, method_name, fail_if_fit_is_called)

    monkeypatch.setattr(
        prediction_module,
        "_load_pipeline",
        lambda: (model, metadata),
    )
    first_result = prediction_module.predict(
        example,
        "test demonstration row (features only)",
    )
    second_result = prediction_module.predict(
        example,
        "test demonstration row (features only)",
    )

    assert first_result["predicted_class"] == expected_class[0], (
        "The prediction script must return the class produced by the loaded "
        "saved model."
    )
    assert first_result["predicted_class"] in model.classes_, (
        "The prediction script must return only a fitted model class."
    )
    assert first_result["predicted_label"] == {
        0: "absence",
        1: "presence",
    }[first_result["predicted_class"]], (
        "The predicted label must match the project's 0=absence, 1=presence encoding."
    )
    assert second_result["predicted_class"] == first_result["predicted_class"], (
        "The deterministic saved model must return the same class for the same input."
    )

    predict_proba = getattr(model, "predict_proba", None)
    if callable(predict_proba):
        expected_probabilities = predict_proba(demo_features)
        assert expected_probabilities.shape == (1, len(model.classes_)), (
            "A single input must have one probability for each fitted class."
        )
        assert np.isfinite(expected_probabilities).all(), (
            "Supported class probabilities must be finite numbers."
        )
        assert ((expected_probabilities >= 0) & (expected_probabilities <= 1)).all(), (
            "Each supported class probability must be between zero and one."
        )
        assert np.isclose(expected_probabilities.sum(axis=1), 1.0).all(), (
            "Class probabilities for one input must sum to one."
        )
        assert first_result["class_probabilities"], (
            "The prediction script must return probabilities when the model "
            "supports predict_proba."
        )
        reported_probabilities = first_result["class_probabilities"]
        assert len(reported_probabilities) == len(model.classes_), (
            "The prediction script must report exactly one probability per "
            "fitted class."
        )
        assert np.isfinite(list(reported_probabilities.values())).all(), (
            "Probabilities returned by the prediction script must be finite."
        )
        assert all(
            0.0 <= probability <= 1.0
            for probability in reported_probabilities.values()
        ), (
            "Probabilities returned by the prediction script must be between "
            "zero and one."
        )
        assert np.isclose(sum(reported_probabilities.values()), 1.0), (
            "Probabilities returned by the prediction script must sum to one."
        )
        assert second_result["class_probabilities"] == reported_probabilities, (
            "The deterministic saved model must return consistent probabilities "
            "for the same input."
        )
    else:
        assert "class_probabilities" not in first_result, (
            "The prediction script must not invent probabilities when the "
            "saved model does not support predict_proba."
        )
