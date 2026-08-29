"""Daemonize a command so it survives its launcher's process group being killed.

macOS has no setsid(1), and a harness/background job that gets stopped takes the
whole process group with it -- which killed three dagger4 attempts. Classic
double-fork + os.setsid() reparents the child to init, so nothing upstream can
signal it.  Usage: python detach.py <logfile> <cmd> [args...]
"""
import os
import sys

log, cmd = sys.argv[1], sys.argv[2:]
if os.fork() > 0:
    os._exit(0)
os.setsid()
if os.fork() > 0:
    os._exit(0)
fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
os.dup2(fd, 1)
os.dup2(fd, 2)
os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
os.execvp(cmd[0], cmd)
