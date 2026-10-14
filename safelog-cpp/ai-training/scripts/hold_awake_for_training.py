"""Bounded Windows execution-state request for the authorized overnight run."""
import ctypes
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT.parents[3]/'outputs'
RECEIPT = OUTPUTS/'INDUSTRIAL_NIGHT_AWAKE_2026-10-15.json'
STOP = OUTPUTS/'INDUSTRIAL_NIGHT_AWAKE_STOP_2026-10-15'
HOURS = 8


def main():
    if os.name != 'nt': raise RuntimeError('Windows-only bounded helper')
    assert not RECEIPT.exists() and not STOP.exists(), 'Do not duplicate or restart a stopped helper'
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.SetThreadExecutionState.argtypes = [ctypes.c_uint]
    kernel.SetThreadExecutionState.restype = ctypes.c_uint
    if not kernel.SetThreadExecutionState(0x80000001): raise ctypes.WinError(ctypes.get_last_error())
    started = datetime.now(timezone.utc); expires = started+timedelta(hours=HOURS)
    with RECEIPT.open('x', encoding='utf-8', newline='\n') as out:
        json.dump({'status': 'active', 'pid': os.getpid(), 'started_utc': started.isoformat(), 'expires_utc': expires.isoformat(),
            'maximum_hours': HOURS, 'system_idle_sleep_temporarily_prevented': True, 'display_kept_on': False,
            'global_power_plan_changed': False, 'stop_marker': str(STOP), 'manual_user_sleep_or_shutdown_not_blocked': True}, out, indent=2)
    try:
        end = time.monotonic()+HOURS*3600
        while time.monotonic() < end and not STOP.exists(): time.sleep(min(30, max(.1, end-time.monotonic())))
    finally:
        kernel.SetThreadExecutionState(0x80000000)
        with (OUTPUTS/'INDUSTRIAL_NIGHT_AWAKE_FINISHED_2026-10-15.json').open('x', encoding='utf-8', newline='\n') as out:
            json.dump({'status': 'restored', 'finished_utc': datetime.now(timezone.utc).isoformat(), 'early_stop_marker': STOP.exists()}, out, indent=2)


if __name__ == '__main__': main()
