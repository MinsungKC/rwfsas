"""Launcher for the Fire Platform Console -> compiled to FireConsole.exe.

Double-click the exe: it locates the Python that has the platform's packages,
starts fpm/console.py, waits for the server, and opens the browser. Closing the
console window stops the server. Small on purpose (no ML libs are frozen in) --
it drives the real interpreter + models already installed on this machine.
"""
import os, sys, time, socket, subprocess, webbrowser
from shutil import which

PORT = 8095
# repo location (this file lives in <FDV>/fpm)
def _fdv():
    here = os.path.dirname(os.path.abspath(sys.argv[0] if getattr(sys, 'frozen', False) else __file__))
    # frozen exe may sit anywhere; prefer the known checkout, else infer from here
    for cand in (os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../FDV')), os.path.dirname(here), here):
        if os.path.isfile(os.path.join(cand, 'fpm', 'console_app.py')):
            return cand
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../FDV'))

FDV = _fdv()
APP = os.path.join(FDV, 'fpm', 'console_app.py')      # native-window entry


def find_python(windowless=True):
    """Prefer pythonw.exe (no console window) for the native app."""
    base = os.path.dirname(sys.executable)
    names = ('pythonw.exe', 'python.exe') if windowless else ('python.exe', 'pythonw.exe')
    cands = [os.path.join(base, n) for n in names]
    if not getattr(sys, 'frozen', False):
        cands.append(sys.executable)
    for n in names:
        w = which(n)
        if w:
            cands.append(w)
    for c in cands:
        if c and os.path.isfile(c) and os.path.basename(c).lower().startswith('python'):
            return c
    return None


def port_open():
    with socket.socket() as s:
        s.settimeout(0.4)
        try:
            return s.connect_ex(('127.0.0.1', PORT)) == 0
        except OSError:
            return False


def _msg(text, title='Fire Platform Console'):
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, text, title, 0x10)
    except Exception:
        pass


def main():
    if not os.path.isfile(APP):
        _msg('Could not find fpm/console_app.py.\nExpected under the FDV repo.'); return
    py = find_python(windowless=True)
    if not py:
        _msg('No Python interpreter found on this machine.'); return
    # launch the native-window app; DETACHED so no console window appears
    flags = 0x00000008 | 0x08000000  # DETACHED_PROCESS | CREATE_NO_WINDOW
    try:
        subprocess.Popen([py, APP], cwd=FDV, creationflags=flags,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        _msg(f'Failed to start the app:\n{e}')


if __name__ == '__main__':
    main()
