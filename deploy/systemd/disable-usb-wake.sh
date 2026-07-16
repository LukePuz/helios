#!/bin/sh
# Disables ACPI wake for the USB controllers, convergently (rechecks actual
# state after each write rather than assuming how many toggles a given
# device needs) — see disable-usb-wake.service for the full story of why
# this exists: a USB peripheral spuriously woke the machine 29 minutes
# into what should have been a ~14hr sleep on the night of 2026-07-15.
set -eu

i=0
while [ "$i" -lt 10 ]; do
    still_enabled=$(grep -E '^(XHC0|XHC1|XHC2|XH00)\b.*\*enabled' /proc/acpi/wakeup | awk '{print $1}' | sort -u)
    [ -z "$still_enabled" ] && break
    for name in $still_enabled; do
        echo "$name" > /proc/acpi/wakeup
    done
    i=$((i + 1))
done

logger -t disable-usb-wake "final state: $(grep -E '^(XHC0|XHC1|XHC2|XH00)\b' /proc/acpi/wakeup | tr '\n' ';')"
