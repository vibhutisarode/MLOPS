#end to end data science project
import dagshub
dagshub.init(repo_owner='vibhutisarode', repo_name='MLOPS', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)