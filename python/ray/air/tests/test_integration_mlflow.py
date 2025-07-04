import os
import shutil
import tempfile
import unittest
from collections import namedtuple
from unittest.mock import MagicMock, patch

from mlflow.tracking import MlflowClient

from ray._private.dict import flatten_dict
from ray.air._internal.mlflow import _MLflowLoggerUtil
from ray.air.integrations.mlflow import MLflowLoggerCallback, _NoopModule, setup_mlflow


class MockTrial(
    namedtuple("MockTrial", ["config", "trial_name", "trial_id", "local_path"])
):
    def __hash__(self):
        return hash(self.trial_id)

    def __str__(self):
        return self.trial_name


class Mock_MLflowLoggerUtil(_MLflowLoggerUtil):
    def save_artifacts(self, dir, run_id):
        self.artifact_saved = True
        self.artifact_info = {"dir": dir, "run_id": run_id}


def clear_env_vars():
    os.environ.pop("MLFLOW_EXPERIMENT_NAME", None)
    os.environ.pop("MLFLOW_EXPERIMENT_ID", None)


class MLflowTest(unittest.TestCase):
    def setUp(self):
        self.tracking_uri = "sqlite:///" + tempfile.mkdtemp() + "/mlflow.sqlite"
        self.registry_uri = "sqlite:///" + tempfile.mkdtemp() + "/mlflow.sqlite"

        client = MlflowClient(
            tracking_uri=self.tracking_uri, registry_uri=self.registry_uri
        )
        client.create_experiment(name="existing_experiment")
        # Mlflow > 2 creates a "Default" experiment which has ID 0, so we start our
        # test with ID 1.
        assert client.get_experiment_by_name("existing_experiment").experiment_id == "1"

    def tearDown(self) -> None:
        pass

    def testMlFlowLoggerCallbackConfig(self):
        # Explicitly pass in all args.
        logger = MLflowLoggerCallback(
            tracking_uri=self.tracking_uri,
            registry_uri=self.registry_uri,
            experiment_name="test_exp",
        )
        logger.setup()
        self.assertEqual(
            logger.mlflow_util._mlflow.get_tracking_uri(), self.tracking_uri
        )
        self.assertEqual(
            logger.mlflow_util._mlflow.get_registry_uri(), self.registry_uri
        )
        self.assertListEqual(
            [e.name for e in logger.mlflow_util._mlflow.search_experiments()],
            ["test_exp", "existing_experiment", "Default"],
        )
        self.assertEqual(logger.mlflow_util.experiment_id, "2")

        # Check if client recognizes already existing experiment.
        logger = MLflowLoggerCallback(
            experiment_name="existing_experiment",
            tracking_uri=self.tracking_uri,
            registry_uri=self.registry_uri,
        )
        logger.setup()
        self.assertEqual(logger.mlflow_util.experiment_id, "1")

        # Pass in experiment name as env var.
        clear_env_vars()
        os.environ["MLFLOW_EXPERIMENT_NAME"] = "test_exp"
        logger = MLflowLoggerCallback(
            tracking_uri=self.tracking_uri, registry_uri=self.registry_uri
        )
        logger.setup()
        self.assertEqual(logger.mlflow_util.experiment_id, "2")

        # Pass in existing experiment name as env var.
        clear_env_vars()
        os.environ["MLFLOW_EXPERIMENT_NAME"] = "existing_experiment"
        logger = MLflowLoggerCallback(
            tracking_uri=self.tracking_uri, registry_uri=self.registry_uri
        )
        logger.setup()
        self.assertEqual(logger.mlflow_util.experiment_id, "1")

        # Pass in existing experiment id as env var.
        clear_env_vars()
        os.environ["MLFLOW_EXPERIMENT_ID"] = "1"
        logger = MLflowLoggerCallback(
            tracking_uri=self.tracking_uri, registry_uri=self.registry_uri
        )
        logger.setup()
        self.assertEqual(logger.mlflow_util.experiment_id, "1")

        # Pass in non existing experiment id as env var.
        # This should create a new experiment.
        clear_env_vars()
        os.environ["MLFLOW_EXPERIMENT_ID"] = "500"
        with self.assertRaises(ValueError):
            logger = MLflowLoggerCallback(
                tracking_uri=self.tracking_uri, registry_uri=self.registry_uri
            )
            logger.setup()

        # Experiment id env var should take precedence over name env var.
        clear_env_vars()
        os.environ["MLFLOW_EXPERIMENT_NAME"] = "test_exp"
        os.environ["MLFLOW_EXPERIMENT_ID"] = "1"
        logger = MLflowLoggerCallback(
            tracking_uri=self.tracking_uri, registry_uri=self.registry_uri
        )
        logger.setup()
        self.assertEqual(logger.mlflow_util.experiment_id, "1")

        # Using tags
        tags = {"user_name": "John", "git_commit_hash": "abc123"}
        clear_env_vars()
        os.environ["MLFLOW_EXPERIMENT_NAME"] = "test_tags"
        os.environ["MLFLOW_EXPERIMENT_ID"] = "1"
        logger = MLflowLoggerCallback(
            tracking_uri=self.tracking_uri, registry_uri=self.registry_uri, tags=tags
        )
        logger.setup()
        self.assertEqual(logger.tags, tags)

    @patch("ray.air.integrations.mlflow._MLflowLoggerUtil", Mock_MLflowLoggerUtil)
    def testMlFlowLoggerLogging(self):
        clear_env_vars()
        trial_config = {"par1": "a", "par2": "b"}
        trial = MockTrial(trial_config, "trial1", 0, "artifact")

        logger = MLflowLoggerCallback(
            tracking_uri=self.tracking_uri,
            registry_uri=self.registry_uri,
            experiment_name="test1",
            save_artifact=True,
            tags={"hello": "world"},
        )
        logger.setup()

        # Check if run is created with proper tags.
        logger.on_trial_start(iteration=0, trials=[], trial=trial)
        all_runs = logger.mlflow_util._mlflow.search_runs(experiment_ids=["2"])
        self.assertEqual(len(all_runs), 1)
        # all_runs is a pandas dataframe.
        all_runs = all_runs.to_dict(orient="records")
        run = logger.mlflow_util._mlflow.get_run(all_runs[0]["run_id"])
        self.assertDictEqual(
            run.data.tags,
            {"hello": "world", "trial_name": "trial1", "mlflow.runName": "trial1"},
        )
        self.assertEqual(logger._trial_runs[trial], run.info.run_id)
        # Params should be logged.
        self.assertDictEqual(run.data.params, trial_config)

        # When same trial is started again, new run should not be created.
        logger.on_trial_start(iteration=0, trials=[], trial=trial)
        all_runs = logger.mlflow_util._mlflow.search_runs(experiment_ids=["2"])
        self.assertEqual(len(all_runs), 1)

        # Check metrics are logged properly.
        result = {
            "metric1": 0.8,
            "metric2": 1,
            "metric3": None,
            "training_iteration": 0,
        }
        logger.on_trial_result(0, [], trial, result)
        run = logger.mlflow_util._mlflow.get_run(run_id=run.info.run_id)
        # metric3 is not logged since it cannot be converted to float.
        self.assertDictEqual(
            run.data.metrics, {"metric1": 0.8, "metric2": 1.0, "training_iteration": 0}
        )

        # Check that artifact is logged on termination.
        logger.on_trial_complete(0, [], trial)
        self.assertTrue(logger.mlflow_util.artifact_saved)
        self.assertDictEqual(
            logger.mlflow_util.artifact_info,
            {"dir": "artifact", "run_id": run.info.run_id},
        )

        # Check if params are logged at the end.
        run = logger.mlflow_util._mlflow.get_run(run_id=run.info.run_id)
        self.assertDictEqual(run.data.params, trial_config)

    @patch("ray.air.integrations.mlflow._MLflowLoggerUtil", Mock_MLflowLoggerUtil)
    def testMlFlowLoggerLogging_logAtEnd(self):
        clear_env_vars()
        trial_config = {"par1": "a", "par2": "b"}
        trial = MockTrial(trial_config, "trial1", 0, "artifact")

        logger = MLflowLoggerCallback(
            tracking_uri=self.tracking_uri,
            registry_uri=self.registry_uri,
            experiment_name="test_log_at_end",
            tags={"hello": "world"},
            log_params_on_trial_end=True,
        )
        logger.setup()
        exp_id = logger.mlflow_util.experiment_id

        logger.on_trial_start(iteration=0, trials=[], trial=trial)
        all_runs = logger.mlflow_util._mlflow.search_runs(experiment_ids=[exp_id])
        self.assertEqual(len(all_runs), 1)
        # all_runs is a pandas dataframe.
        all_runs = all_runs.to_dict(orient="records")
        run = logger.mlflow_util._mlflow.get_run(all_runs[0]["run_id"])

        # Params should NOT be logged at start.
        self.assertDictEqual(run.data.params, {})

        # Check that params are logged at the end.
        logger.on_trial_complete(0, [], trial)
        run = logger.mlflow_util._mlflow.get_run(run_id=run.info.run_id)
        self.assertDictEqual(run.data.params, trial_config)

    def testMlFlowSetupExplicit(self):
        clear_env_vars()
        trial_config = {"par1": 4, "par2": 9.0}

        # No MLflow config passed in.
        with self.assertRaises(ValueError):
            setup_mlflow(trial_config)

        # Invalid experiment-id
        with self.assertRaises(ValueError):
            setup_mlflow(trial_config, experiment_id="500")

        # Set to experiment that does not already exist.
        with self.assertRaises(ValueError):
            setup_mlflow(
                trial_config,
                experiment_id="500",
                experiment_name="new_experiment",
                tracking_uri=self.tracking_uri,
            )

        mlflow = setup_mlflow(
            trial_config,
            experiment_id="500",
            experiment_name="existing_experiment",
            tracking_uri=self.tracking_uri,
        )
        mlflow.end_run()

    @patch("ray.train.get_context")
    def testMlFlowSetupRankNonRankZero(self, mock_get_context):
        """Assert that non-rank-0 workers get a noop module"""
        mock_context = MagicMock()
        mock_context.get_world_rank.return_value = 1

        mock_get_context.return_value = mock_context

        mlflow = setup_mlflow({})
        assert isinstance(mlflow, _NoopModule)

        mlflow.log_metrics()
        mlflow.sklearn.save_model(None, "model_directory")


class MLflowUtilTest(unittest.TestCase):
    def setUp(self):
        self.dirpath = tempfile.mkdtemp()
        import mlflow

        mlflow.set_tracking_uri("sqlite:///" + self.dirpath + "/mlflow.sqlite")
        mlflow.create_experiment(name="existing_experiment")

        self.mlflow_util = _MLflowLoggerUtil()
        self.tracking_uri = mlflow.get_tracking_uri()

    def tearDown(self):
        shutil.rmtree(self.dirpath)

    def test_experiment_id(self):
        self.mlflow_util.setup_mlflow(tracking_uri=self.tracking_uri, experiment_id="0")
        assert self.mlflow_util.experiment_id == "0"

    def test_experiment_id_env_var(self):
        os.environ["MLFLOW_EXPERIMENT_ID"] = "0"
        self.mlflow_util.setup_mlflow(tracking_uri=self.tracking_uri)
        assert self.mlflow_util.experiment_id == "0"
        del os.environ["MLFLOW_EXPERIMENT_ID"]

    def test_experiment_name(self):
        self.mlflow_util.setup_mlflow(
            tracking_uri=self.tracking_uri, experiment_name="existing_experiment"
        )
        assert self.mlflow_util.experiment_id == "1"

    def test_run_started_with_correct_experiment(self):
        experiment_name = "my_experiment_name"
        # Make sure run is started under the correct experiment.
        self.mlflow_util.setup_mlflow(
            tracking_uri=self.tracking_uri, experiment_name=experiment_name
        )
        run = self.mlflow_util.start_run(set_active=True)
        assert (
            run.info.experiment_id
            == self.mlflow_util._mlflow.get_experiment_by_name(
                experiment_name
            ).experiment_id
        )

        self.mlflow_util.end_run()

    def test_experiment_name_env_var(self):
        os.environ["MLFLOW_EXPERIMENT_NAME"] = "existing_experiment"
        self.mlflow_util.setup_mlflow(tracking_uri=self.tracking_uri)
        assert self.mlflow_util.experiment_id == "1"
        del os.environ["MLFLOW_EXPERIMENT_NAME"]

    def test_id_precedence(self):
        os.environ["MLFLOW_EXPERIMENT_ID"] = "0"
        self.mlflow_util.setup_mlflow(
            tracking_uri=self.tracking_uri, experiment_name="new_experiment"
        )
        assert self.mlflow_util.experiment_id == "0"
        del os.environ["MLFLOW_EXPERIMENT_ID"]

    def test_new_experiment(self):
        self.mlflow_util.setup_mlflow(
            tracking_uri=self.tracking_uri, experiment_name="new_experiment"
        )
        assert self.mlflow_util.experiment_id == "2"

    def test_setup_fail(self):
        with self.assertRaises(ValueError):
            self.mlflow_util.setup_mlflow(
                tracking_uri=self.tracking_uri,
                experiment_name="new_experiment2",
                create_experiment_if_not_exists=False,
            )

    def test_log_params(self):
        params = {"a": "a", "x": {"y": "z"}}
        self.mlflow_util.setup_mlflow(
            tracking_uri=self.tracking_uri, experiment_name="new_experiment"
        )
        run = self.mlflow_util.start_run()
        run_id = run.info.run_id
        self.mlflow_util.log_params(params_to_log=params, run_id=run_id)

        run = self.mlflow_util._mlflow.get_run(run_id=run_id)
        assert run.data.params == flatten_dict(params)

        params2 = {"b": "b"}
        self.mlflow_util.start_run(set_active=True)
        self.mlflow_util.log_params(params_to_log=params2, run_id=run_id)
        run = self.mlflow_util._mlflow.get_run(run_id=run_id)
        assert run.data.params == flatten_dict(
            {
                **params,
                **params2,
            }
        )

        self.mlflow_util.end_run()

    def test_log_metrics(self):
        metrics = {"a": 1.0, "x": {"y": 2.0}}
        self.mlflow_util.setup_mlflow(
            tracking_uri=self.tracking_uri, experiment_name="new_experiment"
        )
        run = self.mlflow_util.start_run()
        run_id = run.info.run_id
        self.mlflow_util.log_metrics(metrics_to_log=metrics, run_id=run_id, step=0)

        run = self.mlflow_util._mlflow.get_run(run_id=run_id)
        assert run.data.metrics == flatten_dict(metrics)

        metrics2 = {"b": 1.0}
        self.mlflow_util.start_run(set_active=True)
        self.mlflow_util.log_metrics(metrics_to_log=metrics2, run_id=run_id, step=0)
        assert self.mlflow_util._mlflow.get_run(
            run_id=run_id
        ).data.metrics == flatten_dict(
            {
                **metrics,
                **metrics2,
            }
        )
        self.mlflow_util.end_run()


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main(["-v", __file__]))
