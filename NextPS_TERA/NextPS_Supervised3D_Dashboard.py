import sys
import threading
import time
import collections
import re
import serial
import serial.tools.list_ports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import customtkinter as ctk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Global Theme setup
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class NextPSSupervisedDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("TERA NextPM User-Defined Supervised ML Dashboard")
        self.geometry("1400x980")

        # Serial & Data Management States
        self.ser = None
        self.is_running = False
        self.data_history = collections.deque(maxlen=60)
        self.full_log = []
        
        # Supervised Training Datasets
        self.training_X = []
        self.training_y = []
        self.model = None
        self.is_training_active = False

        self.current_reading = {}
        self.setup_ui_layout()
        self.refresh_com_ports()
        self.log_message("Dashboard initialized. Ready to connect.")

    def setup_ui_layout(self):
        # Left Control Panel Sidebar
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)

        ctk.CTkLabel(self.sidebar, text="CONNECTION CONTROLS", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=10)
        
        self.com_dropdown = ctk.CTkOptionMenu(self.sidebar, values=["No Ports Found"])
        self.com_dropdown.pack(pady=5, fill="x", padx=15)

        self.btn_refresh = ctk.CTkButton(self.sidebar, text="🔄 Refresh Ports", command=self.refresh_com_ports, fg_color="gray")
        self.btn_refresh.pack(pady=5, fill="x", padx=15)

        self.btn_connect = ctk.CTkButton(self.sidebar, text="🔌 Connect Stream", command=self.toggle_connection, fg_color="green")
        self.btn_connect.pack(pady=10, fill="x", padx=15)

        # --- USER-DEFINED TRAINING SECTION ---
        ctk.CTkLabel(self.sidebar, text="USER-DEFINED TRAINING LABELS", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(15, 5))
        
        self.lbl_class_prompt = ctk.CTkLabel(self.sidebar, text="Activity / Regime Name:", font=ctk.CTkFont(size=12))
        self.lbl_class_prompt.pack(anchor="w", padx=15)
        
        self.entry_class_name = ctk.CTkEntry(self.sidebar, placeholder_text="e.g., Cooking, Idle, Cleaning")
        self.entry_class_name.pack(pady=5, fill="x", padx=15)
        self.entry_class_name.insert(0, "Idle")

        self.btn_toggle_record = ctk.CTkButton(self.sidebar, text="🔴 Record Sample", command=self.toggle_recording, fg_color="#b22222")
        self.btn_toggle_record.pack(pady=5, fill="x", padx=15)

        self.lbl_sample_count = ctk.CTkLabel(self.sidebar, text="Recorded Samples: 0", font=ctk.CTkFont(size=12, weight="bold"), text_color="#aaaaaa")
        self.lbl_sample_count.pack(pady=2, padx=15)

        self.btn_train = ctk.CTkButton(self.sidebar, text="🧠 Train Random Forest Model", command=self.train_model, fg_color="#1f538d", state="disabled")
        self.btn_train.pack(pady=8, fill="x", padx=15)

        # Live Prediction Status Box
        ctk.CTkLabel(self.sidebar, text="LIVE PREDICTION STATUS", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10, 2))
        self.lbl_prediction_status = ctk.CTkLabel(
            self.sidebar, 
            text="Model Not Trained Yet", 
            font=ctk.CTkFont(size=13, weight="bold"), 
            fg_color="#1a1a1a", 
            corner_radius=6,
            text_color="#aaaaaa"
        )
        self.lbl_prediction_status.pack(pady=5, fill="x", padx=15, ipady=8)

        # Learning & Tools
        ctk.CTkLabel(self.sidebar, text="TOOLS & GUIDES", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10, 2))
        self.btn_guide = ctk.CTkButton(self.sidebar, text="📖 Educational Guide", command=self.open_educational_guide, fg_color="gray")
        self.btn_guide.pack(pady=3, fill="x", padx=15)

        self.btn_export = ctk.CTkButton(self.sidebar, text="💾 Export Dataset w/ Labels", command=self.export_data_file, state="disabled", fg_color="purple")
        self.btn_export.pack(pady=3, fill="x", padx=15)

        # Activity Log Box
        ctk.CTkLabel(self.sidebar, text="ACTIVITY LOG (RAW STREAM)", font=ctk.CTkFont(size=10, weight="bold")).pack(pady=(10, 2))
        self.log_box = ctk.CTkTextbox(self.sidebar, font=ctk.CTkFont(family="Courier", size=8), height=90)
        self.log_box.pack(fill="x", padx=15, pady=2)
        self.log_box.configure(state="disabled")

        # Main Panel Readouts
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.metrics_row = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.metrics_row.pack(side="top", fill="x", pady=(0, 10))

        # Metrics display rows
        top_metrics = ctk.CTkFrame(self.metrics_row, fg_color="transparent")
        top_metrics.pack(fill="x", expand=True)
        self.lbl_pm03 = ctk.CTkLabel(top_metrics, text="> 0.3 um: -- pcs/m3", font=ctk.CTkFont(size=13, weight="bold"), text_color="#2ca02c")
        self.lbl_pm03.pack(side="left", expand=True, fill="both", padx=5, pady=2)
        self.lbl_pm05 = ctk.CTkLabel(top_metrics, text="> 0.5 um: -- pcs/m3", font=ctk.CTkFont(size=13, weight="bold"), text_color="#1f77b4")
        self.lbl_pm05.pack(side="left", expand=True, fill="both", padx=5, pady=2)
        self.lbl_pm10 = ctk.CTkLabel(top_metrics, text="> 1.0 um: -- pcs/m3", font=ctk.CTkFont(size=13, weight="bold"), text_color="#9467bd")
        self.lbl_pm10.pack(side="left", expand=True, fill="both", padx=5, pady=2)

        bottom_metrics = ctk.CTkFrame(self.metrics_row, fg_color="transparent")
        bottom_metrics.pack(fill="x", expand=True)
        self.lbl_pm25 = ctk.CTkLabel(bottom_metrics, text="> 2.5 um: -- pcs/m3", font=ctk.CTkFont(size=13, weight="bold"), text_color="#ff7f0e")
        self.lbl_pm25.pack(side="left", expand=True, fill="both", padx=5, pady=2)
        self.lbl_pm50 = ctk.CTkLabel(bottom_metrics, text="> 5.0 um: -- pcs/m3", font=ctk.CTkFont(size=13, weight="bold"), text_color="#d62728")
        self.lbl_pm50.pack(side="left", expand=True, fill="both", padx=5, pady=2)

        # Dual-Plot Canvas (Top: Time Series, Bottom: 3D PCA Class Space)
        self.fig = plt.figure(figsize=(7, 9), facecolor='#242424')
        self.fig.subplots_adjust(hspace=0.25, top=0.92, bottom=0.08)
        
        self.ax1 = self.fig.add_subplot(2, 1, 1)
        self.ax2 = self.fig.add_subplot(2, 1, 2, projection='3d')

        # Style Plots
        self.ax1.set_facecolor('#1a1a1a')
        self.ax1.tick_params(colors='white')
        self.ax1.spines['bottom'].set_color('white')
        self.ax1.spines['left'].set_color('white')
        self.ax1.spines['top'].set_visible(False)
        self.ax1.spines['right'].set_visible(False)
        self.ax1.grid(True, color='#444444', linestyle='--')
        self.ax1.set_title("Real-Time Particle Counts", color='white', weight='bold')
        self.ax1.set_ylabel("Particles / m3", color='white')

        self.ax2.set_facecolor('#242424')
        self.ax2.xaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
        self.ax2.yaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
        self.ax2.zaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
        self.ax2.tick_params(colors='white')
        self.ax2.set_title("Interactive 3D PCA Feature Space (User Classes)", color='white', weight='bold')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def log_message(self, msg):
        timestamp = time.strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {msg}\n"
        print(formatted.strip())
        try:
            self.log_box.configure(state="normal")
            self.log_box.insert("end", formatted)
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
        except Exception:
            pass

    def open_educational_guide(self):
        guide_win = ctk.CTkToplevel(self)
        guide_win.title("Educational Guide: Supervised Learning & Random Forest")
        guide_win.geometry("700x600")
        guide_win.attributes("-topmost", True)
        txt_box = ctk.CTkTextbox(guide_win, font=ctk.CTkFont(family="Arial", size=13), wrap="word")
        txt_box.pack(fill="both", expand=True, padx=15, pady=15)
        txt_box.insert("0.0", """=== SUPERVISED ACTIVITY RECOGNITION GUIDE ===

1. WHAT IS SUPERVISED MACHINE LEARNING?
   - Unlike unsupervised clustering (where the computer groups data blindly), supervised learning requires you to provide examples with labels (e.g., typing 'Cooking' and recording data).
   - The model learns the statistical boundary separating your defined classes.

2. HOW THE RANDOM FOREST CLASSIFIER WORKS
   - A Random Forest is an ensemble of multiple decision trees. Each tree votes on what activity is currently occurring based on your 5 particle size channels and ratio features.
   - The majority vote determines the final live prediction and confidence score.

3. INTERACTIVE 3D PCA SPACE
   - Principal Component Analysis reduces your multi-channel sensor data down to 3 axes (PC1, PC2, PC3).
   - In the bottom 3D plot, your recorded classes appear in distinct colors, letting you visually verify whether your custom activities occupy separate regions in space!
""")
        txt_box.configure(state="disabled")

    def toggle_recording(self):
        if not self.is_training_active:
            self.is_training_active = True
            self.btn_toggle_record.configure(text="⏹ Stop Recording", fg_color="green")
            self.log_message(f"Started recording samples for class: '{self.entry_class_name.get()}'")
        else:
            self.is_training_active = False
            self.btn_toggle_record.configure(text="🔴 Record Sample", fg_color="#b22222")
            self.log_message(f"Stopped recording. Total training samples: {len(self.training_y)}")

    def train_model(self):
        if len(self.training_X) < 10:
            self.log_message("Training failed: Need at least 10 recorded samples.")
            return
        try:
            X = np.array(self.training_X)
            y = np.array(self.training_y)
            
            self.model = RandomForestClassifier(n_estimators=50, random_state=42)
            self.model.fit(X, y)
            self.btn_train.configure(fg_color="green")
            self.log_message(f"Random Forest successfully trained on {len(X)} samples across {len(np.unique(y))} classes!")
            self.lbl_prediction_status.configure(text="Model Trained & Active!", text_color="#2ca02c")
        except Exception as e:
            self.log_message(f"Training Error: {e}")

    def refresh_com_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports()]
        if ports:
            self.com_dropdown.configure(values=ports)
            self.com_dropdown.set(ports[0])
            self.log_message(f"Found ports: {', '.join(ports)}")
        else:
            self.com_dropdown.configure(values=["No Ports Found"])
            self.com_dropdown.set("No Ports Found")

    def toggle_connection(self):
        if not self.is_running:
            selected_port = self.com_dropdown.get()
            if selected_port == "No Ports Found":
                return
            try:
                self.ser = serial.Serial(selected_port, 115200, timeout=1)
                self.is_running = True
                self.btn_connect.configure(text="⏹ Stop Stream", fg_color="red")
                self.btn_export.configure(state="normal")
                threading.Thread(target=self.serial_listener_worker, daemon=True).start()
                self.log_message("Connected to stream.")
            except Exception as e:
                self.log_message(f"Connection Error: {e}")
        else:
            self.is_running = False
            if self.ser:
                self.ser.close()
            self.btn_connect.configure(text="🔌 Connect Stream", fg_color="green")
            self.log_message("Stream stopped.")

    def serial_listener_worker(self):
        while self.is_running:
            try:
                if self.ser and self.ser.in_waiting > 0:
                    raw_line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if "Sensor state" in raw_line:
                        self.current_reading = {"timestamp": time.time()}
                    
                    match = re.search(r'>\s*([\d\.]+)\s*um:\s*(\d+)', raw_line)
                    if match and self.current_reading:
                        self.current_reading[f"pm_{match.group(1)}"] = int(match.group(2))
                    
                    if "---" in raw_line and "pm_0.3" in self.current_reading:
                        features = [
                            self.current_reading.get("pm_0.3", 0),
                            self.current_reading.get("pm_0.5", 0),
                            self.current_reading.get("pm_1.0", 0),
                            self.current_reading.get("pm_2.5", 0),
                            self.current_reading.get("pm_5.0", 0),
                            self.current_reading.get("pm_0.3", 0) / (self.current_reading.get("pm_2.5", 1) + 1)
                        ]
                        
                        current_label = "Unlabeled"
                        if self.is_training_active:
                            current_label = self.entry_class_name.get().strip()
                            self.training_X.append(features)
                            self.training_y.append(current_label)
                            self.after(0, lambda: self.lbl_sample_count.configure(text=f"Recorded Samples: {len(self.training_y)}"))
                            if len(self.training_y) >= 10:
                                self.btn_train.configure(state="normal")

                        predicted_label = "Model Not Trained"
                        confidence = 0.0
                        if self.model is not None:
                            pred = self.model.predict([features])[0]
                            prob = np.max(self.model.predict_proba([features]))
                            predicted_label = str(pred)
                            confidence = float(prob) * 100

                        data_point = {
                            "time": self.current_reading["timestamp"],
                            "pm03": features[0], "pm05": features[1], "pm10": features[2],
                            "pm25": features[3], "pm50": features[4],
                            "label": current_label,
                            "prediction": predicted_label,
                            "confidence": confidence
                        }
                        self.data_history.append(data_point)
                        self.full_log.append(data_point)
                        self.after(0, self.update_dashboard_view, data_point)
            except Exception as e:
                pass
            time.sleep(0.01)

    def update_dashboard_view(self, last_packet):
        self.lbl_pm03.configure(text=f"> 0.3 um: {last_packet['pm03']:,} pcs/m3")
        self.lbl_pm05.configure(text=f"> 0.5 um: {last_packet['pm05']:,} pcs/m3")
        self.lbl_pm10.configure(text=f"> 1.0 um: {last_packet['pm10']:,} pcs/m3")
        self.lbl_pm25.configure(text=f"> 2.5 um: {last_packet['pm25']:,} pcs/m3")
        self.lbl_pm50.configure(text=f"> 5.0 um: {last_packet['pm50']:,} pcs/m3")

        if self.model is not None:
            self.lbl_prediction_status.configure(
                text=f"Activity: {last_packet['prediction']} ({last_packet['confidence']:.1f}%)", 
                text_color="#2ca02c"
            )

        self.ax1.clear()
        self.ax2.clear()

        self.ax1.set_facecolor('#1a1a1a')
        self.ax1.tick_params(colors='white')
        self.ax1.spines['bottom'].set_color('white')
        self.ax1.spines['left'].set_color('white')
        self.ax1.spines['top'].set_visible(False)
        self.ax1.spines['right'].set_visible(False)
        self.ax1.grid(True, color='#444444', linestyle='--')
        self.ax1.set_title("Real-Time Particle Counts", color='white', weight='bold')
        self.ax1.set_ylabel("Particles / m3", color='white')

        self.ax2.set_facecolor('#242424')
        self.ax2.xaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
        self.ax2.yaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
        self.ax2.zaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
        self.ax2.tick_params(colors='white')
        self.ax2.set_title("Interactive 3D PCA Feature Space (User Classes)", color='white', weight='bold')

        if len(self.data_history) > 0:
            df = pd.DataFrame(list(self.data_history))
            x_time = df['time'] - df['time'].iloc[0]

            for col, label, color in [('pm03', '> 0.3 um', '#2ca02c'), ('pm05', '> 0.5 um', '#1f77b4'), 
                                      ('pm10', '> 1.0 um', '#9467bd'), ('pm25', '> 2.5 um', '#ff7f0e'), ('pm50', '> 5.0 um', '#d62728')]:
                self.ax1.plot(x_time, df[col], label=label, color=color, linewidth=2)
            self.ax1.legend(facecolor='#242424', labelcolor='white', loc="upper left", fontsize='small')

            # 3D PCA plot colored by user labels
            if len(self.training_X) >= 4:
                try:
                    X_all = np.array(self.training_X)
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_all)
                    pca = PCA(n_components=3)
                    X_pca = pca.fit_transform(X_scaled)
                    
                    unique_labels = list(set(self.training_y))
                    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
                    
                    for idx, lbl in enumerate(unique_labels):
                        indices = [i for i, y in enumerate(self.training_y) if y == lbl]
                        subset = X_pca[indices]
                        self.ax2.scatter(subset[:, 0], subset[:, 1], subset[:, 2], label=lbl, color=colors[idx], s=50, alpha=0.9)
                    self.ax2.legend(facecolor='#242424', labelcolor='white', loc="upper left", fontsize='small')
                except Exception:
                    pass

        self.canvas.draw()

    def export_data_file(self):
        if not self.full_log:
            return
        df = pd.DataFrame(self.full_log)
        filename = f"nextpm_supervised_log_{int(time.time())}.csv"
        df.to_csv(filename, index=False)
        self.log_message(f"Exported supervised dataset to {filename}")

if __name__ == "__main__":
    app = NextPSSupervisedDashboard()
    app.mainloop()