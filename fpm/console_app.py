"""Native desktop-window wrapper for the Fire Platform Console.

Runs the console HTTP server in a background thread and shows the UI inside a
real OS window (Edge WebView2 via pywebview) -- no browser, no URL bar. Closing
the window stops the app.
"""
import os, sys, time, socket
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import console


def _wait_port(port, timeout=15):
    end = time.time() + timeout
    while time.time() < end:
        with socket.socket() as s:
            s.settimeout(0.3)
            if s.connect_ex(('127.0.0.1', port)) == 0:
                return True
        time.sleep(0.2)
    return False


def main():
    # start the server (or attach to an already-running one)
    console.start_server(block=False, open_browser=False)
    _wait_port(console.PORT)
    try:
        import webview
        webview.create_window('Fire Platform Console', f'http://localhost:{console.PORT}',
                              width=1280, height=880, min_size=(900, 640))
        webview.start()          # blocks until the window is closed
    except Exception as e:
        # no webview backend -> fall back to a browser so the app still opens
        print('native window unavailable, using browser:', e)
        import webbrowser; webbrowser.open(f'http://localhost:{console.PORT}')
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
