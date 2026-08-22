"""
Minimal GUI for repeating pickup -> grasp-verify -> lift -> throw without
retyping CLI commands each cycle.

Deliberately thin: every step it runs is the exact same subprocess call this
project already uses and has tested (`pickup_and_lift.py`, then
`run_hardware_throw.py throw --arm --confirm`). This file adds no new
control-path logic of its own -- it is UI plus subprocess orchestration.

SAFETY GATE, NOT A CONVENIENCE SHORTCUT AROUND ONE
-----------------------------------------------------
The confirm checkbox unchecks itself after every run. This is deliberate: the
whole point of "repeatedly" is that it's easy to click Run again without
re-reading the workspace -- so the one gate that matters (E-stop in hand,
workspace clear, ball loaded) has to be re-affirmed every single cycle, not
just the first time. If `pickup_and_lift.py` reports it grasped nothing, the
throw step is refused, not silently skipped -- that's surfaced as a popup.

    python3 throw_gui.py
"""

import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import messagebox, ttk

PY = sys.executable  # this script itself must be launched with /usr/bin/python3


class ThrowGUI:
    def __init__(self, root):
        self.root = root
        root.title("Kinova Gen3 -- pickup / throw")
        self.log_q = queue.Queue()
        self.running = False

        frm = ttk.Frame(root, padding=10)
        frm.grid(sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        r = 0
        self.ip = self._field(frm, r, "IP", "192.168.1.101"); r += 1
        self.robot = self._field(frm, r, "Robot profile", "kinova_gen3_dyn"); r += 1
        self.log_path = self._field(frm, r, "Checkpoint log_path", "results_kinetic_chain_gen3/2"); r += 1
        self.target_x = self._field(frm, r, "Target X (m)", "0.75"); r += 1
        self.target_y = self._field(frm, r, "Target Y (m)", "0.05"); r += 1
        self.u_cap = self._field(frm, r, "u_cap (m/s)", "1.60"); r += 1
        self.wrist_offset = self._field(frm, r, "Wrist roll offset (deg)", "90"); r += 1

        ttk.Label(frm, text="speed_scale").grid(row=r, column=0, sticky="w", pady=2)
        self.speed_scale = tk.StringVar(value="0.15")
        speeds = ttk.Frame(frm)
        speeds.grid(row=r, column=1, sticky="w")
        for s in ("0.15", "0.30", "0.60", "1.00"):
            ttk.Radiobutton(speeds, text=s, variable=self.speed_scale, value=s).pack(side="left")
        r += 1

        self.confirm_var = tk.BooleanVar(value=False)
        self.confirm_cb = ttk.Checkbutton(
            frm, text="Ball loaded, workspace clear, E-stop in hand",
            variable=self.confirm_var)
        self.confirm_cb.grid(row=r, column=0, columnspan=2, sticky="w", pady=(8, 2))
        r += 1

        self.run_btn = ttk.Button(frm, text="RUN: pickup -> grasp -> lift -> throw",
                                  command=self.on_run)
        self.run_btn.grid(row=r, column=0, columnspan=2, sticky="ew", pady=6)
        r += 1

        self.status = ttk.Label(frm, text="idle", foreground="gray")
        self.status.grid(row=r, column=0, columnspan=2, sticky="w")
        r += 1

        self.log = tk.Text(frm, width=90, height=24, state="disabled",
                           bg="black", fg="#c0ffc0", font=("Courier", 10))
        self.log.grid(row=r, column=0, columnspan=2, sticky="nsew", pady=(6, 0))
        frm.rowconfigure(r, weight=1)

        self.root.after(100, self._drain_log)

    def _field(self, frm, row, label, default):
        ttk.Label(frm, text=label).grid(row=row, column=0, sticky="w", pady=2)
        var = tk.StringVar(value=default)
        ttk.Entry(frm, textvariable=var, width=30).grid(row=row, column=1, sticky="w")
        return var

    def _append(self, text):
        self.log_q.put(text)

    def _drain_log(self):
        try:
            while True:
                line = self.log_q.get_nowait()
                self.log.configure(state="normal")
                self.log.insert("end", line)
                self.log.see("end")
                self.log.configure(state="disabled")
        except queue.Empty:
            pass
        self.root.after(100, self._drain_log)

    def _set_status(self, text, color="black"):
        self.status.configure(text=text, foreground=color)

    def _run_subprocess(self, args, label):
        self._append(f"\n$ {' '.join(args)}\n")
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            self._append(line)
        proc.wait()
        return proc.returncode

    def on_run(self):
        if self.running:
            return
        if not self.confirm_var.get():
            messagebox.showwarning(
                "Not confirmed",
                "Check \"Ball loaded, workspace clear, E-stop in hand\" first -- "
                "this resets after every run on purpose.")
            return
        self.confirm_var.set(False)   # re-affirm required every cycle, not just once
        self.running = True
        self.run_btn.configure(state="disabled")
        threading.Thread(target=self._do_cycle, daemon=True).start()

    def _do_cycle(self):
        ip, robot = self.ip.get(), self.robot.get()
        try:
            self._set_status("pickup -> grasp -> lift ...", "orange")
            rc = self._run_subprocess(
                [PY, "pickup_and_lift.py", "--ip", ip, "--robot", robot], "pickup")
            if rc != 0:
                self._set_status("REFUSED: grasp failed (closed on nothing?)", "red")
                self.root.after(0, lambda: messagebox.showerror(
                    "Grasp failed",
                    "pickup_and_lift.py did not detect a real grasp -- "
                    "closed on nothing. Throw refused. Check the log."))
                return

            self._set_status(f"throwing at speed_scale={self.speed_scale.get()} ...", "orange")
            args = [PY, "run_hardware_throw.py", "throw", "--arm", "--confirm",
                   "--ip", ip, "--robot", robot,
                   "--log_path", self.log_path.get(),
                   "--target", self.target_x.get(), self.target_y.get(),
                   "--speed_scale", self.speed_scale.get(),
                   "--u_cap", self.u_cap.get(),
                   "--wrist_roll_offset_deg", self.wrist_offset.get()]
            rc = self._run_subprocess(args, "throw")
            if rc != 0:
                self._set_status(f"THROW FAILED (exit {rc}) -- see log", "red")
                self.root.after(0, lambda: messagebox.showerror(
                    "Throw failed", f"run_hardware_throw.py exited {rc}. Check the log "
                                    f"and run hw_readonly_check.py before trying again."))
                return

            self._set_status("done -- clean run", "green")
        finally:
            self.running = False
            self.root.after(0, lambda: self.run_btn.configure(state="normal"))


if __name__ == "__main__":
    root = tk.Tk()
    ThrowGUI(root)
    root.mainloop()
