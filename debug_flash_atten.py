import subprocess
import os
import sys

command = f"""{sys.executable} -m flux
        --name flux-dev
        --height 1024 --width 1024 
      """
print(command)
subprocess.run(command.split())
