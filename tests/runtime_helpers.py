"""Tiny real sklearn artifacts for API tests; never model-performance evidence."""
import json
import joblib
import pandas as pd
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import FunctionTransformer
from src.common.schema import FEATURES,SCHEMA_VERSION
from src.common.paths import PROJECT_ROOT


def write_bundle(root):
    root.mkdir(parents=True,exist_ok=True)
    X=pd.DataFrame([[0.]*len(FEATURES),[1.]*len(FEATURES)],columns=FEATURES)
    joblib.dump(DummyClassifier(strategy='prior').fit(X,[0,1]),root/'best_model.joblib')
    joblib.dump(FunctionTransformer(validate=False),root/'preprocessing.joblib')
    card=dict(schema_version=SCHEMA_VERSION,ordered_features=FEATURES,
              model_version='synthetic-fixture',status='development_baseline',
              artifact_layout='split_estimator_preprocessor')
    (root/'model_card.json').write_text(json.dumps(card))
    decoder=yaml.safe_load((PROJECT_ROOT/'configs/decoder.yaml').read_text())
    decoder['model_version']=card['model_version']
    (root/'decoder_config.json').write_text(json.dumps(decoder))
