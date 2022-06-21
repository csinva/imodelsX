import subprocess
import sys

if len(sys.argv) > 1:
    s = int(sys.argv[1])
else:
    s = 287318 
e = s + 30900
for j in list(range(s, e+1)) + list(range(e, s+1)):
    print("scancel " + str(j))
    subprocess.call("scancel " + str(j), shell=True)
