"""Read-only process existence checks; never send Windows termination signals."""
import os


def process_exists(pid):
    if isinstance(pid, bool) or not isinstance(pid, int) or not 0 < pid < 2**32:
        raise ValueError('A positive process id is required')
    if os.name != 'nt':
        try: os.kill(pid, 0)
        except ProcessLookupError: return False
        except PermissionError: return True
        return True
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION only.
    if not handle:
        error = ctypes.get_last_error()
        if error == 87: return False
        if error == 5: return True  # Fail closed if the process cannot be inspected.
        raise RuntimeError('Cannot safely query lock process: '+str(error))
    try:
        code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise RuntimeError('Cannot safely read lock process state')
        return code.value == 259  # STILL_ACTIVE.
    finally: kernel.CloseHandle(handle)


def probe_only(pid, signal):
    if signal != 0: raise ValueError('This helper only permits read-only existence probes')
    if not process_exists(pid): raise ProcessLookupError(pid)
