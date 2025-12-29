import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import os


class NegativityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Transformation into Negativity")
        self.root.geometry("1200x650")
        self.root.minsize(1000, 600)

        # Images
        self.original_image = None     # PIL Image (RGB)
        self.output_image = None       # PIL Image (RGB)
        self.tk_input = None           # PhotoImage reference
        self.tk_output = None          # PhotoImage reference
        self.loaded_path = None

        self._build_ui()

        # Keep images scaled nicely when resizing window
        self.input_canvas.bind("<Configure>", lambda e: self._refresh_previews())
        self.output_canvas.bind("<Configure>", lambda e: self._refresh_previews())

    # ---------------- UI ----------------
    def _build_ui(self):
        # Root grid: 1 row header controls, 1 row images
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # 1) CONTROL FRAME (Top)
        self.control_frame = ttk.LabelFrame(self.root, text="Control Frame")
        self.control_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
        self.control_frame.columnconfigure(4, weight=1)

        ttk.Button(self.control_frame, text="Load Image", command=self.load_image).grid(
            row=0, column=0, padx=8, pady=10, sticky="w"
        )
        ttk.Button(self.control_frame, text="Apply Negativity Effect", command=self.apply_negative).grid(
            row=0, column=1, padx=8, pady=10, sticky="w"
        )
        ttk.Button(self.control_frame, text="Reset", command=self.reset).grid(
            row=0, column=2, padx=8, pady=10, sticky="w"
        )
        ttk.Button(self.control_frame, text="Save Output", command=self.save_output).grid(
            row=0, column=3, padx=8, pady=10, sticky="w"
        )

        self.status_var = tk.StringVar(value="Ready - Load an image to begin.")
        ttk.Label(self.control_frame, textvariable=self.status_var).grid(
            row=0, column=4, padx=8, pady=10, sticky="e"
        )

        # 2) MAIN CONTAINER for INPUT + OUTPUT frames (Bottom)
        container = ttk.Frame(self.root)
        container.grid(row=1, column=0, sticky="nsew", padx=10, pady=(6, 10))
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(0, weight=1)

        # 2) INPUT FRAME (Left)
        self.input_frame = ttk.LabelFrame(container, text="Input Frame (Original Image)")
        self.input_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.input_frame.rowconfigure(0, weight=1)
        self.input_frame.columnconfigure(0, weight=1)

        self.input_canvas = tk.Canvas(self.input_frame, bg="white", highlightthickness=1)
        self.input_canvas.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # 3) OUTPUT FRAME (Right)
        self.output_frame = ttk.LabelFrame(container, text="Output Frame (Negative Result)")
        self.output_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self.output_frame.rowconfigure(0, weight=1)
        self.output_frame.columnconfigure(0, weight=1)

        self.output_canvas = tk.Canvas(self.output_frame, bg="white", highlightthickness=1)
        self.output_canvas.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Optional: placeholder text
        self._draw_placeholder(self.input_canvas, "No input image loaded")
        self._draw_placeholder(self.output_canvas, "Output will appear here")

    def _draw_placeholder(self, canvas, text):
        canvas.delete("all")
        w = canvas.winfo_width() or 400
        h = canvas.winfo_height() or 300
        canvas.create_text(w // 2, h // 2, text=text, fill="gray", font=("Segoe UI", 14))

    # ---------------- Image handling ----------------
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return

        try:
            img = Image.open(file_path)
            if img.mode != "RGB":
                img = img.convert("RGB")

            self.loaded_path = file_path
            self.original_image = img
            self.output_image = None

            self.status_var.set(f"Loaded: {os.path.basename(file_path)}  |  Size: {img.size}")
            self._refresh_previews()
        except Exception as e:
            messagebox.showerror("Error", f"Could not open image:\n{e}")

    def apply_negative(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return

        # Transformation into Negativity: negative = 255 - pixel
        arr = np.array(self.original_image, dtype=np.uint8)
        neg = 255 - arr
        self.output_image = Image.fromarray(neg, mode="RGB")

        self.status_var.set("Negativity effect applied.")
        self._refresh_previews()

    def reset(self):
        if self.original_image is None and self.output_image is None:
            self.status_var.set("Nothing to reset.")
            return
        self.output_image = None
        self.status_var.set("Reset done. Output cleared.")
        self._refresh_previews()

    def save_output(self):
        if self.output_image is None:
            messagebox.showwarning("Warning", "No output image to save! Apply the negativity effect first.")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Output Image",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
        )
        if not file_path:
            return

        try:
            self.output_image.save(file_path)
            self.status_var.set(f"Saved output: {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save image:\n{e}")

    # ---------------- Preview rendering ----------------
    def _refresh_previews(self):
        # Input
        if self.original_image is None:
            self._draw_placeholder(self.input_canvas, "No input image loaded")
        else:
            self.tk_input = self._render_on_canvas(self.input_canvas, self.original_image)

        # Output
        if self.output_image is None:
            self._draw_placeholder(self.output_canvas, "Output will appear here")
        else:
            self.tk_output = self._render_on_canvas(self.output_canvas, self.output_image)

    def _render_on_canvas(self, canvas, pil_img):
        canvas.delete("all")

        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw <= 2 or ch <= 2:
            cw, ch = 500, 400

        iw, ih = pil_img.size
        scale = min(cw / iw, ch / ih, 1.0)  # no upscale
        nw, nh = int(iw * scale), int(ih * scale)

        resized = pil_img.resize((nw, nh), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(resized)

        x = (cw - nw) // 2
        y = (ch - nh) // 2
        canvas.create_image(x, y, anchor="nw", image=tk_img)

        # Keep reference alive
        return tk_img


def main():
    root = tk.Tk()
    app = NegativityApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
