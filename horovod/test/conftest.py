import sys
import os
import pytest
sys.path.append(os.path.abspath('../build/lib.linux-x86_64-3.5/'))

import horovod.tensorflow as hvd

hvd.init()
