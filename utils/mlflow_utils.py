# utils/mlflow_utils.py

import mlflow
import os

def setup_mlflow(
    experiment_name="default",
    tracking_uri=None,
    artifact_location=None,
    create=True,
    print_status=True
):
    """
    Initialiserer MLflow-tracking. Sætter tracking URI, eksperiment og artefakt-sti.
    Kør denne først, før du bruger andre MLflow-funktioner!
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        if print_status:
            print(f"[MLflow] Tracking URI sat til: {tracking_uri}")
    if create:
        # Opret (eller hent) eksperiment
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            exp_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_location
            )
            if print_status:
                print(f"[MLflow] Oprettede nyt experiment: {experiment_name} (ID: {exp_id})")
        else:
            exp_id = exp.experiment_id
            if print_status:
                print(f"[MLflow] Bruger eksisterende experiment: {experiment_name} (ID: {exp_id})")
        mlflow.set_experiment(experiment_name)
    else:
        mlflow.set_experiment(experiment_name)
    return mlflow.get_experiment_by_name(experiment_name)

def start_mlflow_run(run_name=None, tags=None, print_status=True):
    """
    Starter et nyt MLflow-run (brug context-manager, eller luk manuelt).
    """
    run = mlflow.start_run(run_name=run_name, tags=tags)
    if print_status:
        print(f"[MLflow] Startet run: {run.info.run_id} | Navn: {run_name}")
    return run

def log_params(params):
    """
    Logger hyperparametre til MLflow. Params skal være dict.
    """
    mlflow.log_params(params)
    print(f"[MLflow] Loggede params: {params}")

def log_metrics(metrics, step=None):
    """
    Logger et dictionary af metrics (fx accuracy, loss, winrate osv).
    """
    if step is not None:
        for k, v in metrics.items():
            mlflow.log_metric(k, v, step=step)
    else:
        mlflow.log_metrics(metrics)
    print(f"[MLflow] Loggede metrics: {metrics}")

def log_artifact(path, artifact_path=None):
    """
    Logger en fil eller mappe som MLflow artefakt.
    """
    if artifact_path:
        mlflow.log_artifact(path, artifact_path)
    else:
        mlflow.log_artifact(path)
    print(f"[MLflow] Loggede artefakt: {path}")

def log_artefacts_in_dir(directory, artifact_path=None, filetypes=None):
    """
    Logger alle filer i en mappe (valgfrit filter på filtyper, fx ['.png']).
    """
    if not os.path.isdir(directory):
        print(f"[MLflow] Mappe ikke fundet: {directory}")
        return
    for fn in os.listdir(directory):
        if filetypes and not any(fn.endswith(ft) for ft in filetypes):
            continue
        file_path = os.path.join(directory, fn)
        if os.path.isfile(file_path):
            log_artifact(file_path, artifact_path=artifact_path)

def end_mlflow_run(print_status=True):
    """
    Afslutter det aktive MLflow-run.
    """
    mlflow.end_run()
    if print_status:
        print(f"[MLflow] Run afsluttet.")

def get_current_run_id():
    """
    Returnerer det aktive run_id, hvis der er et MLflow-run åbent.
    """
    active_run = mlflow.active_run()
    if active_run is not None:
        return active_run.info.run_id
    return None
