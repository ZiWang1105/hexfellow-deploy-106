"""Continuous vendor-SDK capture; IPC consumers request only the latest frame."""
import argparse
from multiprocessing.connection import Connection
import threading
import time

from .camera_stream import RealSenseSource, SyntheticSource, ZedSource


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fd', type=int, required=True)
    parser.add_argument('--synthetic', action='store_true', help='Software fixture; opens no camera')
    args = parser.parse_args()
    connection = Connection(args.fd)
    stop, ready = threading.Event(), threading.Event()
    lock = threading.Lock()
    state = dict(frame=None, error=None)
    thread = None
    try:
        config = connection.recv()
        def capture():
            source = None
            try:
                source = SyntheticSource() if args.synthetic else {
                    'zed': ZedSource, 'realsense': RealSenseSource}[config['type']](config)
                source.open()
                period = 1 / float(config.get('fps', 15))
                while not stop.is_set():
                    started = time.monotonic()
                    frame = source.read()
                    if frame is not None:
                        # Each vendor adapter returns owned arrays. Main holds a
                        # reference while sending; capture only replaces the slot.
                        with lock:
                            state['frame'] = frame
                        ready.set()
                    stop.wait(max(0, period-(time.monotonic()-started)))
            except BaseException as error:
                with lock:
                    state['error'] = f'{type(error).__name__}: {error}'
                ready.set()
            finally:
                if source is not None:
                    try:
                        source.close()
                    except Exception as error:
                        with lock:
                            state['error'] = f'Camera close failed: {error}'
        thread = threading.Thread(target=capture, name='vendor-rgbd-capture', daemon=True)
        thread.start()
        if not ready.wait(14):
            raise RuntimeError('Camera did not publish its first frame')
        if state['error']:
            raise RuntimeError(state['error'])
        connection.send({'kind': 'ready'})
        while True:
            operation = connection.recv()
            if operation == 'close':
                stop.set()
                thread.join(timeout=2)
                if thread.is_alive():
                    raise RuntimeError('Camera capture did not close')
                if state['error']:
                    raise RuntimeError(state['error'])
                connection.send({'kind': 'closed'})
                break
            if operation != 'read':
                raise ValueError('Unsupported camera worker operation')
            with lock:
                frame, error = state['frame'], state['error']
            if error:
                raise RuntimeError(error)
            connection.send({'kind': 'frame', 'frame': frame})
    except Exception as error:
        try:
            connection.send({'kind': 'error', 'error': f'{type(error).__name__}: {error}'})
        except (OSError, EOFError):
            pass
    finally:
        stop.set()
        if thread is not None:
            thread.join(timeout=2)
        connection.close()


if __name__ == '__main__':
    main()
