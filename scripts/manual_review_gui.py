#!/usr/bin/env python3
"""Simple desktop GUI for manual EEG artifact review."""
from __future__ import annotations

from pathlib import Path
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Allow direct execution from the repository root without package install.
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eeg_pipeline.manual_review import default_sidecar_path, review_file


def _default_cleaned_path(input_path: Path, mode: str) -> Path:
    suffix = "_manual_cleaned_raw.fif" if mode == "raw" else "_manual_cleaned_epo.fif"
    return input_path.with_name(f"{input_path.stem}{suffix}")


class ManualReviewApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("EEG Manual Artifact Review")
        self.geometry("980x360")
        self.minsize(760, 300)
        self.resizable(True, True)

        self.input_var = tk.StringVar()
        self.sidecar_var = tk.StringVar()
        self.mode_var = tk.StringVar(value="auto")
        self.save_cleaned_var = tk.BooleanVar(value=False)
        self.cleaned_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Select a file and click Start Review.")

        self._build_ui()

    def _build_ui(self) -> None:
        padx = 8
        pady = 6

        frm = ttk.Frame(self)
        frm.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(5, weight=1)

        ttk.Label(frm, text="Input EEG File").grid(row=0, column=0, sticky=tk.W, padx=padx, pady=pady)
        ttk.Entry(frm, textvariable=self.input_var, width=72).grid(
            row=0, column=1, sticky=tk.EW, padx=padx, pady=pady
        )
        ttk.Button(frm, text="Choose File", command=self._browse_input).grid(
            row=0, column=2, sticky=tk.W, padx=padx, pady=pady
        )

        ttk.Label(frm, text="Review Mode").grid(row=1, column=0, sticky=tk.W, padx=padx, pady=pady)
        ttk.Combobox(
            frm,
            textvariable=self.mode_var,
            values=["auto", "raw", "epochs"],
            width=10,
            state="readonly",
        ).grid(row=1, column=1, sticky=tk.W, padx=padx, pady=pady)

        ttk.Label(frm, text="Sidecar JSON").grid(row=2, column=0, sticky=tk.W, padx=padx, pady=pady)
        ttk.Entry(frm, textvariable=self.sidecar_var, width=72).grid(
            row=2, column=1, sticky=tk.EW, padx=padx, pady=pady
        )
        ttk.Button(frm, text="Browse", command=self._browse_sidecar).grid(row=2, column=2, sticky=tk.W, padx=padx, pady=pady)

        ttk.Checkbutton(
            frm,
            text="Save cleaned FIF after review",
            variable=self.save_cleaned_var,
            command=self._toggle_cleaned,
        ).grid(row=3, column=0, sticky=tk.W, padx=padx, pady=pady)
        self.cleaned_entry = ttk.Entry(frm, textvariable=self.cleaned_var, width=72, state=tk.DISABLED)
        self.cleaned_entry.grid(row=3, column=1, sticky=tk.EW, padx=padx, pady=pady)
        self.cleaned_btn = ttk.Button(frm, text="Browse", command=self._browse_cleaned, state=tk.DISABLED)
        self.cleaned_btn.grid(row=3, column=2, sticky=tk.W, padx=padx, pady=pady)

        ttk.Button(frm, text="Start Review", command=self._start_review).grid(
            row=4, column=1, sticky=tk.W, padx=padx, pady=10
        )
        ttk.Label(frm, textvariable=self.status_var, foreground="#1F2937").grid(
            row=5, column=0, columnspan=3, sticky=tk.W, padx=padx, pady=pady
        )

    def _browse_input(self) -> None:
        path = filedialog.askopenfilename(
            title="Select EEG file",
            filetypes=[
                ("EEG Files", "*.vhdr *.set *.fif"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        p = Path(path)
        self.input_var.set(str(p))
        self.sidecar_var.set(str(default_sidecar_path(p)))
        mode = self.mode_var.get()
        if mode in {"raw", "epochs"}:
            self.cleaned_var.set(str(_default_cleaned_path(p, mode)))
        self.status_var.set("Ready to start review.")

    def _browse_sidecar(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save sidecar JSON",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.sidecar_var.set(path)

    def _browse_cleaned(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save cleaned FIF",
            defaultextension=".fif",
            filetypes=[("FIF", "*.fif"), ("All files", "*.*")],
        )
        if path:
            self.cleaned_var.set(path)

    def _toggle_cleaned(self) -> None:
        enabled = self.save_cleaned_var.get()
        state = tk.NORMAL if enabled else tk.DISABLED
        self.cleaned_entry.configure(state=state)
        self.cleaned_btn.configure(state=state)
        if not enabled:
            self.cleaned_var.set("")
            return

        input_path = self.input_var.get().strip()
        mode = self.mode_var.get()
        if input_path and mode in {"raw", "epochs"} and not self.cleaned_var.get().strip():
            self.cleaned_var.set(str(_default_cleaned_path(Path(input_path), mode)))

    def _start_review(self) -> None:
        input_text = self.input_var.get().strip()
        if not input_text:
            messagebox.showerror("Missing input", "Select an input EEG file first.")
            return

        input_path = Path(input_text)
        if not input_path.exists():
            messagebox.showerror("Missing file", f"Input file not found:\n{input_path}")
            return

        sidecar_text = self.sidecar_var.get().strip()
        sidecar = Path(sidecar_text) if sidecar_text else default_sidecar_path(input_path)

        save_cleaned = None
        if self.save_cleaned_var.get():
            cleaned_text = self.cleaned_var.get().strip()
            if not cleaned_text:
                messagebox.showerror("Missing output", "Provide a cleaned FIF output path or uncheck save option.")
                return
            save_cleaned = Path(cleaned_text)

        self.status_var.set("Review in progress. Close the MNE browser window to continue.")
        self.update_idletasks()

        try:
            result = review_file(
                input_path=input_path,
                mode=self.mode_var.get(),
                sidecar_path=sidecar,
                save_cleaned_path=save_cleaned,
                block=True,
            )
        except Exception as e:  # pragma: no cover - UI pathway
            messagebox.showerror("Review failed", str(e))
            self.status_var.set("Review failed. Fix inputs and retry.")
            return

        msg = (
            f"Mode: {result.mode}\n"
            f"Sidecar: {result.sidecar_path}\n"
            f"Bad channels: {len(result.bad_channels)}\n"
            f"Annotations: {result.n_annotations}\n"
            f"Dropped epochs: {result.n_dropped_epochs}"
        )
        if result.cleaned_output_path is not None:
            msg += f"\nCleaned file: {result.cleaned_output_path}"
        messagebox.showinfo("Manual review complete", msg)
        self.status_var.set("Review complete.")


def main() -> None:
    app = ManualReviewApp()
    app.mainloop()


if __name__ == "__main__":
    main()
