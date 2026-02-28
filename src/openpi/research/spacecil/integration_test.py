"""End-to-end integration tests for SpaceCIL training flow.

Covers CLI parsing, config resolution, scorer construction, eval episode loading,
metrics computation, make_train_fn contract, and import smoke tests across all
functionality added in Tasks 1-11.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

# Add project root to sys.path so `scripts.train_spacecil` is importable.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[4])  # src/openpi/research/spacecil -> project root
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# Trigger _CONFIGS construction before any get_config() calls.
import openpi.training.config  # noqa: F401


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestConfigIntegration:
    def test_make_train_fn_importable(self) -> None:
        """make_train_fn can be imported and has the correct 7-parameter signature."""
        from scripts.train_spacecil import make_train_fn

        sig = inspect.signature(make_train_fn)
        params = list(sig.parameters.keys())
        assert params == [
            "config",
            "train_state",
            "state_sharding",
            "mesh",
            "rng",
            "num_steps_per_task",
            "distillation",
        ]

    def test_all_base_configs_resolve(self) -> None:
        """All 4 task configs + debug resolve via get_config()."""
        from openpi.training.config import get_config

        base_names = [
            "spacecil_rm75_payload",
            "spacecil_rm75_latch",
            "spacecil_rm75_clean",
            "spacecil_rm75_connector",
            "spacecil_debug",
        ]
        for name in base_names:
            cfg = get_config(name)
            assert cfg.name == name

    def test_all_baseline_configs_resolve(self) -> None:
        """All 5 baseline variant types resolve via get_config()."""
        from openpi.training.config import get_config

        variant_names = [
            "spacecil_rm75_payload_fulltune",
            "spacecil_rm75_latch_nodistill",
            "spacecil_rm75_shared_lora",
            "spacecil_rm75_clean_oracle",
            "spacecil_rm75_connector_random",
        ]
        for name in variant_names:
            cfg = get_config(name)
            assert cfg.name == name

    def test_fulltune_config_has_no_freeze(self) -> None:
        """spacecil_rm75_payload_fulltune has freeze_filter=None."""
        from openpi.training.config import get_config

        cfg = get_config("spacecil_rm75_payload_fulltune")
        assert cfg.freeze_filter is None

    def test_total_config_count(self) -> None:
        """get_spacecil_configs() returns exactly 22 configs."""
        from openpi.research.spacecil.config import get_spacecil_configs

        configs = get_spacecil_configs()
        assert len(configs) == 22


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestCLIParsing:
    def test_parse_args_all_flags(self) -> None:
        """parse_args handles all CLI arguments including new ones."""
        from scripts.train_spacecil import parse_args

        test_argv = [
            "train_spacecil.py",
            "--config",
            "spacecil_debug",
            "--task-sequence",
            "payload",
            "latch",
            "clean",
            "connector",
            "--num-steps-per-task",
            "500",
            "--distillation-alpha",
            "0.3",
            "--enable-distillation",
            "--checkpoint-dir",
            "/tmp/ckpts",
            "--seed",
            "123",
            "--exp-name",
            "test_exp",
            "--eval-dir",
            "/tmp/eval",
            "--calibration-dir",
            "/tmp/cal",
            "--oracle-routing",
            "--operational-weights",
            "[0.5, 0.6, 0.7, 0.8]",
        ]
        with mock.patch("sys.argv", test_argv):
            args = parse_args()

        assert args.config_name == "spacecil_debug"
        assert args.task_sequence == ["payload", "latch", "clean", "connector"]
        assert args.num_steps_per_task == 500
        assert args.distillation_alpha == pytest.approx(0.3)
        assert args.enable_distillation is True
        assert args.checkpoint_dir == "/tmp/ckpts"
        assert args.seed == 123
        assert args.exp_name == "test_exp"
        assert args.eval_dir == "/tmp/eval"
        assert args.calibration_dir == "/tmp/cal"
        assert args.oracle_routing is True
        assert args.random_routing is False
        assert args.operational_weights == "[0.5, 0.6, 0.7, 0.8]"

    def test_parse_args_defaults(self) -> None:
        """Default values when no optional args provided."""
        from scripts.train_spacecil import parse_args

        test_argv = [
            "train_spacecil.py",
            "--config",
            "spacecil_debug",
            "--task-sequence",
            "task_a",
            "task_b",
        ]
        with mock.patch("sys.argv", test_argv):
            args = parse_args()

        assert args.config_name == "spacecil_debug"
        assert args.task_sequence == ["task_a", "task_b"]
        assert args.num_steps_per_task == 10_000
        assert args.distillation_alpha == 0.5
        assert args.enable_distillation is False
        assert args.checkpoint_dir == "checkpoints"
        assert args.seed == 42
        assert args.exp_name == "spacecil_run"
        assert args.eval_dir == "data/eval_episodes/"
        assert args.calibration_dir == "data/calibration/"
        assert args.oracle_routing is False
        assert args.random_routing is False
        assert args.operational_weights is None

    def test_oracle_and_random_mutually_exclusive(self) -> None:
        """Both --oracle-routing and --random-routing raises SystemExit."""
        from scripts.train_spacecil import parse_args

        test_argv = [
            "train_spacecil.py",
            "--config",
            "spacecil_debug",
            "--task-sequence",
            "task_a",
            "--oracle-routing",
            "--random-routing",
        ]
        with mock.patch("sys.argv", test_argv), pytest.raises(SystemExit):
            parse_args()


# ---------------------------------------------------------------------------
# Scorer tests
# ---------------------------------------------------------------------------


class TestBuildScorers:
    def test_build_scorers_returns_all_four(self) -> None:
        """_build_scorers() returns dict with 4 task keys."""
        from scripts.train_spacecil import _build_scorers

        scorers = _build_scorers()
        assert set(scorers.keys()) == {"payload", "latch", "clean", "connector"}

    def test_scorers_are_correct_types(self) -> None:
        """Each scorer is the correct class instance."""
        from scripts.train_spacecil import _build_scorers

        from openpi.research.shared.scorer_base import (
            ConnectorMatingScorer,
            LatchActuationScorer,
            PayloadTransferScorer,
            SurfaceCleaningScorer,
        )

        scorers = _build_scorers()
        assert isinstance(scorers["payload"], PayloadTransferScorer)
        assert isinstance(scorers["latch"], LatchActuationScorer)
        assert isinstance(scorers["clean"], SurfaceCleaningScorer)
        assert isinstance(scorers["connector"], ConnectorMatingScorer)


# ---------------------------------------------------------------------------
# Eval episodes tests
# ---------------------------------------------------------------------------


class TestLoadEvalEpisodes:
    def test_load_eval_episodes_empty_dir(self) -> None:
        """Graceful degradation with non-existent directory."""
        from scripts.train_spacecil import _load_eval_episodes

        result = _load_eval_episodes("/nonexistent/path", ["payload", "latch"])
        assert isinstance(result, dict)
        assert result["payload"] == []
        assert result["latch"] == []

    def test_load_eval_episodes_with_data(self) -> None:
        """Create temp episode JSON files and verify loading."""
        from openpi.research.shared.episode_schema import (
            Action,
            Episode,
            EpisodeLabels,
            EpisodeMetadata,
            EpisodeStep,
            Observation,
        )
        from scripts.train_spacecil import _load_eval_episodes

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a task directory with 2 episode files.
            task_dir = os.path.join(tmpdir, "payload")
            os.makedirs(task_dir)

            for i in range(2):
                ep = Episode(
                    metadata=EpisodeMetadata(task_id="payload", env_id="test_env"),
                    labels=EpisodeLabels(success=True),
                    steps=[
                        EpisodeStep(
                            observation=Observation(
                                wrist_rgb=np.zeros((64, 64, 3), dtype=np.uint8),
                                joint_position=np.zeros(7, dtype=np.float32),
                                joint_velocity=np.zeros(7, dtype=np.float32),
                                gripper_position=np.array([0.5], dtype=np.float32),
                            ),
                            action=Action(joint_pos=np.zeros(7, dtype=np.float32), gripper_cmd=0.5),
                            timestamp_s=0.0,
                        )
                    ],
                    prompt="transfer payload",
                )
                with open(os.path.join(task_dir, f"ep_{i:03d}.json"), "w") as f:
                    json.dump(ep.to_dict(), f)

            result = _load_eval_episodes(tmpdir, ["payload", "latch"])
            assert len(result["payload"]) == 2
            assert result["latch"] == []  # no directory for latch
            # Verify loaded episodes are Episode instances.
            assert hasattr(result["payload"][0], "metadata")
            assert result["payload"][0].metadata.task_id == "payload"


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------


class TestMetricsOnMockData:
    def test_metrics_on_mock_result_matrix(self) -> None:
        """Run all 4 metrics on a mock result matrix and verify valid floats."""
        from openpi.research.spacecil.metrics import (
            average_success,
            backward_transfer,
            forgetting,
            operational_forgetting,
        )

        # 3x3 result matrix: tasks get progressively worse after training new ones
        R = np.array(
            [
                [0.9, np.nan, np.nan],
                [0.7, 0.8, np.nan],
                [0.5, 0.6, 0.9],
            ]
        )
        # Replace NaN with 0 for metrics (matching pattern from continual_harness_test.py)
        R_clean = np.nan_to_num(R, nan=0.0)

        avg = average_success(R_clean)
        assert isinstance(avg, float)
        assert 0.0 <= avg <= 1.0

        bt = backward_transfer(R_clean)
        assert isinstance(bt, float)

        fgt = forgetting(R_clean)
        assert isinstance(fgt, float)
        assert fgt >= 0.0

        weights = np.array([0.6, 0.8, 1.0])
        op_fgt = operational_forgetting(R_clean, weights=weights)
        assert isinstance(op_fgt, float)
        assert op_fgt >= 0.0

    def test_operational_forgetting_with_custom_weights(self) -> None:
        """operational_forgetting uses custom weights correctly."""
        from openpi.research.spacecil.metrics import operational_forgetting

        # 3x3 matrix where task 0 degrades badly, task 1 is stable
        R = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.8, 1.0, 0.0],
                [0.2, 0.9, 1.0],
            ]
        )

        # High weight on task 0 → higher operational forgetting
        high_w0 = operational_forgetting(R, weights=np.array([10.0, 0.1, 1.0]))
        # High weight on task 1 → lower operational forgetting (task 1 barely forgot)
        high_w1 = operational_forgetting(R, weights=np.array([0.1, 10.0, 1.0]))

        assert high_w0 > high_w1, (
            f"Weighting the heavily-forgotten task should produce more forgetting: high_w0={high_w0}, high_w1={high_w1}"
        )

    def test_metrics_specific_values(self) -> None:
        """Verify exact metric values on a simple known matrix."""
        from openpi.research.spacecil.metrics import (
            average_success,
            backward_transfer,
            forgetting,
        )

        # 2x2 matrix: task 0 degrades from 1.0 to 0.6 after training task 1
        R = np.array(
            [
                [1.0, 0.0],
                [0.6, 0.8],
            ]
        )

        # average_success = mean(R[-1, :]) = mean([0.6, 0.8]) = 0.7
        assert average_success(R) == pytest.approx(0.7)

        # backward_transfer = mean(R[j][j] - R[-1][j] for j in 0..T-2) = R[0][0] - R[1][0] = 0.4
        assert backward_transfer(R) == pytest.approx(0.4)

        # forgetting = mean(max_k(R[k][j]) - R[-1][j] for j in 0..T-2) = max(1.0,0.6) - 0.6 = 0.4
        assert forgetting(R) == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# make_train_fn contract tests
# ---------------------------------------------------------------------------


class TestMakeTrainFnContract:
    def test_make_train_fn_signature(self) -> None:
        """Verify function signature matches expected parameters."""
        from scripts.train_spacecil import make_train_fn

        sig = inspect.signature(make_train_fn)
        params = sig.parameters

        # Check positional params
        assert list(params.keys()) == [
            "config",
            "train_state",
            "state_sharding",
            "mesh",
            "rng",
            "num_steps_per_task",
            "distillation",
        ]

        # distillation must be keyword-only with default None
        distill_param = params["distillation"]
        assert distill_param.kind == inspect.Parameter.KEYWORD_ONLY
        assert distill_param.default is None

        # num_steps_per_task must be positional
        nsteps_param = params["num_steps_per_task"]
        assert nsteps_param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD

    def test_make_train_fn_no_not_implemented(self) -> None:
        """Verify no NotImplementedError in the source."""
        from scripts import train_spacecil

        source = inspect.getsource(train_spacecil.make_train_fn)
        assert "NotImplementedError" not in source
        assert "raise NotImplemented" not in source


# ---------------------------------------------------------------------------
# Import smoke test
# ---------------------------------------------------------------------------


class TestImportSmoke:
    def test_import_all_main_components(self) -> None:
        """Smoke test that all main imports from train_spacecil.py work."""
        from scripts.train_spacecil import (  # noqa: F401
            SpaceCILArgs,
            _build_scorers,
            _load_eval_episodes,
            make_train_fn,
            parse_args,
        )

        # Verify they are callable
        assert callable(make_train_fn)
        assert callable(parse_args)
        assert callable(_build_scorers)
        assert callable(_load_eval_episodes)

        # Verify SpaceCILArgs is a dataclass
        import dataclasses

        assert dataclasses.is_dataclass(SpaceCILArgs)

    def test_import_spacecil_metrics(self) -> None:
        """All metric functions are importable."""
        from openpi.research.spacecil.metrics import (  # noqa: F401
            average_success,
            backward_transfer,
            forgetting,
            operational_forgetting,
            routing_accuracy,
            routing_entropy,
        )

        assert callable(average_success)
        assert callable(routing_entropy)

    def test_import_spacecil_harness_components(self) -> None:
        """ContinualHarness and ContinualResult are importable."""
        from openpi.research.spacecil.continual_harness import (  # noqa: F401
            ContinualHarness,
            ContinualResult,
        )
        from openpi.research.spacecil.task_adapter_bank import TaskAdapterBank  # noqa: F401
        from openpi.research.spacecil.behavior_distillation import (  # noqa: F401
            BehaviorDistillation,
            CalibrationMemory,
            TeacherSnapshot,
        )

        assert callable(ContinualHarness)
        assert callable(TaskAdapterBank)


# ---------------------------------------------------------------------------
# SpaceCILArgs dataclass tests
# ---------------------------------------------------------------------------


class TestSpaceCILArgs:
    def test_spacecil_args_fields(self) -> None:
        """SpaceCILArgs has all expected fields."""
        from scripts.train_spacecil import SpaceCILArgs

        fields = {f.name for f in SpaceCILArgs.__dataclass_fields__.values()}
        expected = {
            "config_name",
            "task_sequence",
            "num_steps_per_task",
            "distillation_alpha",
            "enable_distillation",
            "checkpoint_dir",
            "seed",
            "eval_dir",
            "calibration_dir",
            "exp_name",
            "operational_weights",
            "oracle_routing",
            "random_routing",
        }
        assert fields == expected

    def test_spacecil_args_defaults(self) -> None:
        """SpaceCILArgs default values are correct."""
        from scripts.train_spacecil import SpaceCILArgs

        args = SpaceCILArgs(config_name="test", task_sequence=["a"])
        assert args.num_steps_per_task == 10_000
        assert args.distillation_alpha == 0.5
        assert args.enable_distillation is False
        assert args.seed == 42
        assert args.oracle_routing is False
        assert args.random_routing is False
        assert args.operational_weights is None
