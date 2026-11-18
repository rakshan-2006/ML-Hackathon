"""Explicit development-only adapter for the verified existing GB Release.

Person A owns the final Predictor. This adapter provides real baseline probabilities
while their selected/tuned split bundle is pending; it is not a new trained model.
"""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.preprocessing import FunctionTransformer

from src.common.paths import PROJECT_ROOT
from src.common.schema import FEATURES, SCHEMA_VERSION
from src.data.download_dataset import download


class DevelopmentScorer:
    def __init__(self, model, preprocessing, card):
        if card.get('status') != 'development_baseline':
            raise ValueError('The development adapter cannot load a final selected model')
        if card.get('ordered_features') != FEATURES or card.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('Development artifact schema/features mismatch')
        if card.get('artifact_layout') != 'split_estimator_preprocessor':
            raise ValueError('Unknown artifact layout')
        self.model, self.preprocessing = model, preprocessing

    def score_candidates(self, candidates):
        if any(name not in candidates for name in FEATURES):
            raise ValueError('Missing candidate features')
        if not len(candidates):
            return np.empty(0,dtype=np.float64)
        classes = list(self.model.classes_)
        if 1 not in classes:
            raise ValueError('Estimator has no positive class')
        transformed = self.preprocessing.transform(candidates[FEATURES])
        scores = np.asarray(self.model.predict_proba(transformed)[:,classes.index(1)],dtype=np.float64)
        if scores.shape != (len(candidates),) or not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any():
            raise ValueError('Invalid scorer probabilities')
        return scores


def prepare(source_root, output_root):
    source, output = Path(source_root), Path(output_root)
    index_path = PROJECT_ROOT/'data/releases/gradient_boosting.json'
    index = json.loads(index_path.read_text())
    source.mkdir(parents=True,exist_ok=True)
    download(index_path,source)
    pipeline = joblib.load(source/'gradient_boosting_pipeline.joblib')
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('Choose an empty development bundle directory')
    card = json.loads((source/'model_card.json').read_text())
    card.update(status='development_baseline',artifact_layout='split_estimator_preprocessor',
                ordered_features=FEATURES,baseline_release=index['release_url'])
    # Native baseline was fitted on a DataFrame; identity preserves column names.
    joblib.dump(pipeline.steps[-1][1],output/'best_model.joblib')
    joblib.dump(FunctionTransformer(validate=False),output/'preprocessing.joblib')
    (output/'model_card.json').write_text(json.dumps(card,indent=2))
    decoder = yaml.safe_load((PROJECT_ROOT/'configs/decoder.yaml').read_text())
    decoder['model_version'] = card['model_version']
    decoder['data_config_sha256'] = hashlib.sha256((PROJECT_ROOT/'configs/data.yaml').read_bytes()).hexdigest()
    (output/'decoder_config.json').write_text(json.dumps(decoder,indent=2))
    manifest = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
    (output/'deployment_manifest.json').write_text(json.dumps({'status':'development_baseline','sha256':manifest},indent=2))
    print(json.dumps({'status':'development_baseline','output':str(output),'model_version':card['model_version']}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',default='artifacts/gradient_boosting')
    p.add_argument('--output-root',default='models/development')
    a=p.parse_args()
    prepare(a.source_root,a.output_root)


if __name__=='__main__':
    main()
