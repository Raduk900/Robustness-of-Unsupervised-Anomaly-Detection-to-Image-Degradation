# No method modules are imported here: running methods/<x>.py directly would
# otherwise be a circular import. Only sys.path is set up.
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
