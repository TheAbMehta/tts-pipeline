#!/usr/bin/env python3
import csv
import os
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOG_DIR = "/home/abmehta/tts-pipeline/checkpoints/lightning_logs/version_0"
OUT_DIR = "/home/abmehta/tts-pipeline/output"
os.makedirs(OUT_DIR, exist_ok=True)

ea = EventAccumulator(LOG_DIR, size_guidance={"scalars": 0})
ea.Reload()

tags = ea.Tags().get("scalars", [])
start_time = None

for tag in tags:
    events = ea.Scalars(tag)
    if events and (start_time is None or events[0].wall_time < start_time):
        start_time = events[0].wall_time

for tag in tags:
    events = ea.Scalars(tag)
    if not events:
        continue
    safe_name = tag.replace("/", "_")
    fname = os.path.join(OUT_DIR, "metric_" + safe_name + ".csv")
    with open(fname, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "value", "wall_time", "elapsed_min"])
        for e in events:
            elapsed = (e.wall_time - start_time) / 60 if start_time else 0
            w.writerow([e.step, round(e.value, 6), round(e.wall_time), round(elapsed, 2)])
    print("  {}: {} entries -> {}".format(tag, len(events), fname))

fname = os.path.join(OUT_DIR, "training_summary.csv")
all_steps = {}
for tag in tags:
    for e in ea.Scalars(tag):
        if e.step not in all_steps:
            all_steps[e.step] = {
                "step": e.step,
                "wall_time": round(e.wall_time),
                "elapsed_min": round((e.wall_time - start_time) / 60, 2) if start_time else 0,
            }
        all_steps[e.step][tag] = round(e.value, 4)

fields = ["step", "elapsed_min", "wall_time", "epoch", "loss_gen_all", "loss_disc_all", "val_loss"]
with open(fname, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for step in sorted(all_steps.keys()):
        row = all_steps[step]
        w.writerow(row)

print("\n  Combined: {} steps -> {}".format(len(all_steps), fname))

if "loss_gen_all" in tags:
    gen = ea.Scalars("loss_gen_all")
    disc = ea.Scalars("loss_disc_all")
    val = ea.Scalars("val_loss") if "val_loss" in tags else []
    elapsed = (gen[-1].wall_time - start_time) / 60
    rate = gen[-1].step / elapsed if elapsed > 0 else 0
    eta = (10000 - gen[-1].step) / rate if rate > 0 else 0
    print("\n=== CURRENT STATUS ===")
    print("Step: {} / 10000".format(gen[-1].step))
    print("Elapsed: {:.1f} min ({:.1f} hrs)".format(elapsed, elapsed / 60))
    print("Rate: {:.1f} steps/min".format(rate))
    print("ETA: {:.0f} min ({:.1f} hrs)".format(eta, eta / 60))
    print("Gen loss:  {:.2f} -> {:.2f}".format(gen[0].value, gen[-1].value))
    print("Disc loss: {:.2f} -> {:.2f}".format(disc[0].value, disc[-1].value))
    if val:
        print("Val loss:  {:.2f} -> {:.2f}".format(val[0].value, val[-1].value))
