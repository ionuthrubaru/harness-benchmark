#!/usr/bin/env python3
"""cpu_sampler.py <pid> <out.csv> [interval_s]: eșantionează CPU-ul unui proces (toate thread-urile) până se termină.

Scrie t_unix, utime_s, stime_s cumulate (din /proc/<pid>/stat) și numărul de thread-uri. Fereastra măsurată a unui
benchmark se selectează după aceea din JSON-ul lui (timestamp de final și wall_time_s). Doar biblioteca standard.
"""
import os
import sys
import time

pid, out = int(sys.argv[1]), sys.argv[2]
dt = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
tck = os.sysconf("SC_CLK_TCK")
with open(out, "w") as f:
    f.write("t_unix,utime_s,stime_s,threads\n")
    while True:
        try:
            with open(f"/proc/{pid}/stat") as s:
                st = s.read()
        except (FileNotFoundError, ProcessLookupError):
            break
        rest = st[st.rfind(")") + 2:].split()   # câmpurile de după comm; comm poate conține spații
        if rest and rest[0] == "Z":              # zombie: procesul s-a terminat
            break
        f.write(f"{time.time():.3f},{int(rest[11]) / tck:.2f},{int(rest[12]) / tck:.2f},{rest[17]}\n")
        f.flush()
        time.sleep(dt)
