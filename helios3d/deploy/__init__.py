"""Deployment files for Helios 3D jobs: systemd units and timers for the
shadow worker and the labeling job. Every unit runs separately from
helios-local.service, so a live restart never drops a shadow job and a shadow
job never shares a process with the live pipeline. Units ship commented out
until they are installed on purpose.
"""
