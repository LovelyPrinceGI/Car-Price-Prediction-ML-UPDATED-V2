import pickle

def save(filename:str, obj:object):
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load(filename:str) -> object:
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)
    return b

import mlflow
import os

model_name = os.environ['APP_MODEL_NAME']
# def load_mlflow(stage: str):
#     cache_path = os.path.join("models", stage)
#     if(os.path.exists(cache_path) == False):
#         os.makedirs(cache_path)

#     # Check if we cache the model
#     path = os.path.join(cache_path, model_name)
#     if(os.path.exists( path ) == False):
#         # This will keep load the model again and again
#         model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{stage}")
#         save(filename=path, obj=model)

#     model = load(path)
#     return model

# def load_mlflow(stage: str):
#     # Rely on the environment variable set at runtime
#     tracking_uri = os.environ.get("ML_FLOW_TRACKING_URI")
#     if tracking_uri:
#         mlflow.set_tracking_uri(tracking_uri)
#     else:
#         raise ValueError("ML_FLOW_TRACKING_URI is not set in the environment.")
#     # Construct the model URI, e.g., "models:/my-model/staging"
#     model_uri = f"models:/st124876-a3-model/{stage}"
#     model = mlflow.pyfunc.load_model(model_uri)
#     return model

def register_model_production():
    from mlflow.client import MlflowClient
    client = MlflowClient()
    for model in client.get_registered_model('st124876-a3-model').latest_versions: # type:ignore
        # Find model in staging
        if(model.current_stage == 'staging'):
            version = model.version
            client.transition_model_version_stage(
                name=model_name, version=version, stage='Production', archive_existing_versions=True
            )