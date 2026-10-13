"""Current-pid query is non-terminating; unsupported signals are refused."""
import os
import unittest
from safelog_ai.process_probe import process_exists, probe_only


class ProcessProbeTests(unittest.TestCase):
    def test_current_and_nonexistent_pid_queries_do_not_terminate_current_process(self):
        self.assertTrue(process_exists(os.getpid())); probe_only(os.getpid(), 0)
        self.assertTrue(process_exists(os.getpid())); self.assertFalse(process_exists(2147483647))
        with self.assertRaises(ProcessLookupError): probe_only(2147483647, 0)

    def test_termination_signals_and_invalid_pids_are_rejected(self):
        with self.assertRaises(ValueError): probe_only(os.getpid(), 15)
        for pid in [0, -1, True, '123']:
            with self.assertRaises(ValueError): process_exists(pid)
        self.assertTrue(process_exists(os.getpid()))


if __name__ == '__main__': unittest.main()
