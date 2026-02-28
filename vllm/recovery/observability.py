import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from vllm.logger import init_logger

logger = init_logger(__name__)


def _parse_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _parse_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class RecoveryConfig:
    obs_enabled: bool
    flags_json_path: Optional[str]
    log_dir: str
    ts_period_ms: int
    recovery_v2: bool
    recovery_budget: bool
    recovery_phase: int
    emit_ts_csv: bool
    emit_events_jsonl: bool
    emit_cycle_events: bool
    emit_request_events: bool
    swap_bytes_per_block: int
    ts_flush_rows: int
    events_flush_rows: int
    events_flush_period_ms: int
    events_low_freq_only: bool

    @classmethod
    def from_sources(cls) -> "RecoveryConfig":
        cfg: Dict[str, Any] = {
            "obs_enabled": True,
            "flags_json_path": os.getenv("VLLM_RECOVERY_FLAGS_JSON"),
            "log_dir": "/tmp/vllm_recovery",
            "ts_period_ms": 50,
            "recovery_v2": False,
            "recovery_budget": False,
            "recovery_phase": 0,
            "emit_ts_csv": True,
            "emit_events_jsonl": True,
            "emit_cycle_events": False,
            "emit_request_events": True,
            "swap_bytes_per_block": 0,
            "ts_flush_rows": 100,
            "events_flush_rows": 20,
            "events_flush_period_ms": 500,
            "events_low_freq_only": True,
        }

        json_path = cfg["flags_json_path"]
        if json_path:
            cfg.update(_load_flags_json(json_path))

        env_bool_fields = {
            "obs_enabled": "VLLM_RECOVERY_OBS",
            "recovery_v2": "VLLM_RECOVERY_V2",
            "recovery_budget": "VLLM_RECOVERY_BUDGET",
            "emit_ts_csv": "VLLM_RECOVERY_EMIT_TS_CSV",
            "emit_events_jsonl": "VLLM_RECOVERY_EMIT_EVENTS_JSONL",
            "emit_cycle_events": "VLLM_RECOVERY_EMIT_CYCLE_EVENTS",
            "emit_request_events": "VLLM_RECOVERY_EMIT_REQUEST_EVENTS",
            "events_low_freq_only": "VLLM_RECOVERY_EVENTS_LOW_FREQ_ONLY",
        }
        for key, env_name in env_bool_fields.items():
            if env_name in os.environ:
                cfg[key] = _parse_bool(os.getenv(env_name), bool(cfg[key]))

        if "VLLM_RECOVERY_LOG_DIR" in os.environ:
            cfg["log_dir"] = os.getenv("VLLM_RECOVERY_LOG_DIR") or cfg[
                "log_dir"]
        if "VLLM_RECOVERY_TS_PERIOD_MS" in os.environ:
            cfg["ts_period_ms"] = _parse_int(
                os.getenv("VLLM_RECOVERY_TS_PERIOD_MS"), cfg["ts_period_ms"])
        if "VLLM_RECOVERY_PHASE" in os.environ:
            cfg["recovery_phase"] = _parse_int(os.getenv("VLLM_RECOVERY_PHASE"),
                                               cfg["recovery_phase"])
        if "VLLM_RECOVERY_SWAP_BYTES_PER_BLOCK" in os.environ:
            cfg["swap_bytes_per_block"] = _parse_int(
                os.getenv("VLLM_RECOVERY_SWAP_BYTES_PER_BLOCK"),
                cfg["swap_bytes_per_block"])
        if "VLLM_RECOVERY_TS_FLUSH_ROWS" in os.environ:
            cfg["ts_flush_rows"] = _parse_int(
                os.getenv("VLLM_RECOVERY_TS_FLUSH_ROWS"), cfg["ts_flush_rows"])
        if "VLLM_RECOVERY_EVENTS_FLUSH_ROWS" in os.environ:
            cfg["events_flush_rows"] = _parse_int(
                os.getenv("VLLM_RECOVERY_EVENTS_FLUSH_ROWS"),
                cfg["events_flush_rows"])
        if "VLLM_RECOVERY_EVENTS_FLUSH_PERIOD_MS" in os.environ:
            cfg["events_flush_period_ms"] = _parse_int(
                os.getenv("VLLM_RECOVERY_EVENTS_FLUSH_PERIOD_MS"),
                cfg["events_flush_period_ms"])

        ts_period_ms = max(1, _parse_int(cfg["ts_period_ms"], 50))
        recovery_phase = max(0, _parse_int(cfg["recovery_phase"], 0))
        swap_bytes_per_block = max(
            0, _parse_int(cfg["swap_bytes_per_block"], 0))
        ts_flush_rows = max(1, _parse_int(cfg["ts_flush_rows"], 100))
        events_flush_rows = max(1, _parse_int(cfg["events_flush_rows"], 20))
        events_flush_period_ms = max(
            1, _parse_int(cfg["events_flush_period_ms"], 500))

        return cls(
            obs_enabled=_parse_bool(cfg["obs_enabled"], True),
            flags_json_path=cfg["flags_json_path"],
            log_dir=str(cfg["log_dir"]),
            ts_period_ms=ts_period_ms,
            recovery_v2=_parse_bool(cfg["recovery_v2"], False),
            recovery_budget=_parse_bool(cfg["recovery_budget"], False),
            recovery_phase=recovery_phase,
            emit_ts_csv=_parse_bool(cfg["emit_ts_csv"], True),
            emit_events_jsonl=_parse_bool(cfg["emit_events_jsonl"], True),
            emit_cycle_events=_parse_bool(cfg["emit_cycle_events"], True),
            emit_request_events=_parse_bool(cfg["emit_request_events"], True),
            swap_bytes_per_block=swap_bytes_per_block,
            ts_flush_rows=ts_flush_rows,
            events_flush_rows=events_flush_rows,
            events_flush_period_ms=events_flush_period_ms,
            events_low_freq_only=_parse_bool(cfg["events_low_freq_only"], True),
        )


def _load_flags_json(path: str) -> Dict[str, Any]:
    key_map = {
        "VLLM_RECOVERY_OBS": "obs_enabled",
        "recovery_obs": "obs_enabled",
        "obs_enabled": "obs_enabled",
        "VLLM_RECOVERY_LOG_DIR": "log_dir",
        "recovery_log_dir": "log_dir",
        "log_dir": "log_dir",
        "VLLM_RECOVERY_TS_PERIOD_MS": "ts_period_ms",
        "recovery_ts_period_ms": "ts_period_ms",
        "ts_period_ms": "ts_period_ms",
        "VLLM_RECOVERY_V2": "recovery_v2",
        "recovery_v2": "recovery_v2",
        "VLLM_RECOVERY_BUDGET": "recovery_budget",
        "recovery_budget": "recovery_budget",
        "VLLM_RECOVERY_PHASE": "recovery_phase",
        "recovery_phase": "recovery_phase",
        "emit_ts_csv": "emit_ts_csv",
        "emit_events_jsonl": "emit_events_jsonl",
        "emit_cycle_events": "emit_cycle_events",
        "emit_request_events": "emit_request_events",
        "VLLM_RECOVERY_EVENTS_LOW_FREQ_ONLY": "events_low_freq_only",
        "events_low_freq_only": "events_low_freq_only",
        "VLLM_RECOVERY_SWAP_BYTES_PER_BLOCK": "swap_bytes_per_block",
        "recovery_swap_bytes_per_block": "swap_bytes_per_block",
        "swap_bytes_per_block": "swap_bytes_per_block",
        "VLLM_RECOVERY_TS_FLUSH_ROWS": "ts_flush_rows",
        "ts_flush_rows": "ts_flush_rows",
        "VLLM_RECOVERY_EVENTS_FLUSH_ROWS": "events_flush_rows",
        "events_flush_rows": "events_flush_rows",
        "VLLM_RECOVERY_EVENTS_FLUSH_PERIOD_MS": "events_flush_period_ms",
        "events_flush_period_ms": "events_flush_period_ms",
    }
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:
        logger.warning("Failed to parse VLLM_RECOVERY_FLAGS_JSON=%s (%r)",
                       path, exc)
        return {}

    if not isinstance(raw, dict):
        logger.warning("Ignoring non-object recovery flags JSON at %s", path)
        return {}

    resolved: Dict[str, Any] = {}
    for key, value in raw.items():
        mapped = key_map.get(str(key))
        if mapped is not None:
            resolved[mapped] = value
    return resolved


_RECOVERY_CONFIG_SINGLETON: Optional[RecoveryConfig] = None


def get_recovery_config() -> RecoveryConfig:
    global _RECOVERY_CONFIG_SINGLETON
    if _RECOVERY_CONFIG_SINGLETON is None:
        _RECOVERY_CONFIG_SINGLETON = RecoveryConfig.from_sources()
    return _RECOVERY_CONFIG_SINGLETON
