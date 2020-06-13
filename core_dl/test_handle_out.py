import sys
import signal
import atexit

class EventHandle:

    def __init__(self):
        atexit.register(self.release)
        signal.signal(signal.SIGTERM, self.sigHandler)

    def release(self):
        print "Release HS resources..."

    def sigHandler(self, signo, frame):
        sys.exit(0)