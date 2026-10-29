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
from sklearn.mixture import GaussianMixture
import customtkinter as ctk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D projections
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Global Theme setup
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class NextPMDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("TERA NextPM Educational Air Quality & ML Dashboard")
        self.geometry("1350x980")

        # Serial & Data Management States
        self.ser = None
        self.is_running = False
        self.data_history = collections.deque(maxlen=60) # Store last 60 seconds rolling window
        self.full_log = [] # Holds all data for statistics & CSV export
        
        self.current_reading = {}

        self.setup_ui_layout()
        self.refresh_com_ports()
        self.log_message("Dashboard initialized. Ready to connect.")

    def setup_ui_layout(self):
        # Left Control Panel Sidebar
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)

        ctk.CTkLabel(self.sidebar, text="CONNECTION CONTROLS", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=15)
        
        self.com_dropdown = ctk.CTkOptionMenu(self.sidebar, values=["No Ports Found"])
        self.com_dropdown.pack(pady=10, fill="x", padx=15)

        self.btn_refresh = ctk.CTkButton(self.sidebar, text="🔄 Refresh Ports", command=self.refresh_com_ports, fg_color="gray")
        self.btn_refresh.pack(pady=5, fill="x", padx=15)

        self.btn_connect = ctk.CTkButton(self.sidebar, text="🔌 Connect Stream", command=self.toggle_connection, fg_color="green")
        self.btn_connect.pack(pady=15, fill="x", padx=15)

        ctk.CTkLabel(self.sidebar, text="LIVE ML & REGIME STATUS", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(15, 5))
        
        self.lbl_regime_status = ctk.CTkLabel(
            self.sidebar, 
            text="Regime: Standby", 
            font=ctk.CTkFont(size=13, weight="bold"), 
            fg_color="#1a1a1a", 
            corner_radius=6,
            text_color="#aaaaaa"
        )
        self.lbl_regime_status.pack(pady=5, fill="x", padx=15, ipady=8)

        ctk.CTkLabel(self.sidebar, text="LEARNING & ANALYSIS TOOLS", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(20, 5))
        
        # Educational Guide Pop-up Button
        self.btn_guide = ctk.CTkButton(self.sidebar, text="📖 Educational Guide", command=self.open_educational_guide, fg_color="#1f538d")
        self.btn_guide.pack(pady=5, fill="x", padx=15)

        self.btn_stats = ctk.CTkButton(self.sidebar, text="📊 Run Descriptive Stats", command=self.calculate_statistics, state="disabled")
        self.btn_stats.pack(pady=5, fill="x", padx=15)

        self.btn_export = ctk.CTkButton(self.sidebar, text="💾 Export Dataset w/ ML", command=self.export_data_file, state="disabled", fg_color="purple")
        self.btn_export.pack(pady=5, fill="x", padx=15)

        # Real-time Activity Log Box at bottom of sidebar
        ctk.CTkLabel(self.sidebar, text="ACTIVITY LOG (RAW STREAM)", font=ctk.CTkFont(size=11, weight="bold")).pack(pady=(20, 2))
        self.log_box = ctk.CTkTextbox(self.sidebar, font=ctk.CTkFont(family="Courier", size=9), height=120)
        self.log_box.pack(fill="x", padx=15, pady=5)
        self.log_box.configure(state="disabled")

        # Main Panel Readouts
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.metrics_row = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.metrics_row.pack(side="top", fill="x", pady=(0, 10))

        # Top row metrics (3 items)
        top_metrics = ctk.CTkFrame(self.metrics_row, fg_color="transparent")
        top_metrics.pack(fill="x", expand=True)

        self.lbl_pm03 = ctk.CTkLabel(top_metrics, text="> 0.3 um: -- pcs/m3", font=ctk.CTkFont(size=13, weight="bold"), text_color="#2ca02c")
        self.lbl_pm03.pack(side="left", expand=True, fill="both", padx=5, pady=2)

        self.lbl_pm05 = ctk.CTkLabel(top_metrics, text="> 0.5 um: -- pcs/m3", font=ctk.CTkFont(size=13, weight="bold"), text_color="#1f77b4")
        self.lbl_pm05.pack(side="left", expand=True, fill="both", padx=5, pady=2)

        self.lbl_pm10 = ctk.CTkLabel(top_metrics, text="> 1.0 um: -- pcs/m3", font=ctk.CTkFont(size=13, weight="bold"), text_color="#9467bd")
        self.lbl_pm10.pack(side="left", expand=True, fill="both", padx=5, pady=2)

        # Bottom row metrics (2 items)
        bottom_metrics = ctk.CTkFrame(self.metrics_row, fg_color="transparent")
        bottom_metrics.pack(fill="x", expand=True)

        self.lbl_pm25 = ctk.CTkLabel(bottom_metrics, text="> 2.5 um: -- pcs/m3", font=ctk.CTkFont(size=13, weight="bold"), text_color="#ff7f0e")
        self.lbl_pm25.pack(side="left", expand=True, fill="both", padx=5, pady=2)

        self.lbl_pm50 = ctk.CTkLabel(bottom_metrics, text="> 5.0 um: -- pcs/m3", font=ctk.CTkFont(size=13, weight="bold"), text_color="#d62728")
        self.lbl_pm50.pack(side="left", expand=True, fill="both", padx=5, pady=2)

        # Dynamic Dual-Plot Canvas (Top: Time Series, Bottom: Interactive 3D PCA)
        self.fig = plt.figure(figsize=(7, 9), facecolor='#242424')
        self.fig.subplots_adjust(hspace=0.25, top=0.92, bottom=0.08)
        
        self.ax1 = self.fig.add_subplot(2, 1, 1)
        self.ax2 = self.fig.add_subplot(2, 1, 2, projection='3d')

        # Style Top Plot
        self.ax1.set_facecolor('#1a1a1a')
        self.ax1.tick_params(colors='white')
        self.ax1.spines['bottom'].set_color('white')
        self.ax1.spines['left'].set_color('white')
        self.ax1.spines['top'].set_visible(False)
        self.ax1.spines['right'].set_visible(False)
        self.ax1.grid(True, color='#444444', linestyle='--')
        self.ax1.set_title("Real-Time Particle Counts (Time Series)", color='white', weight='bold')
        self.ax1.set_xlabel("Elapsed Time (s)", color='white')
        self.ax1.set_ylabel("Particles / m3", color='white')

        # Style Bottom 3D Plot
        self.ax2.set_facecolor('#242424')
        self.ax2.xaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
        self.ax2.yaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
        self.ax2.zaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
        self.ax2.tick_params(colors='white')

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
        guide_win.title("Educational Guide: Air Quality & Machine Learning")
        guide_win.geometry("700x650")
        guide_win.attributes("-topmost", True)

        txt_box = ctk.CTkTextbox(guide_win, font=ctk.CTkFont(family="Arial", size=13), wrap="word")
        txt_box.pack(fill="both", expand=True, padx=15, pady=15)

        guide_content = """=== HOW THIS DASHBOARD WORKS ===

1. THE HARDWARE SENSOR (TERA NextPM via ESP32)
   - Your sensor samples particulate matter simultaneously across 5 distinct size thresholds: >0.3 µm, >0.5 µm, >1.0 µm, >2.5 µm, and >5.0 µm.
   - Different activities change the *shape* of these size distributions. For example, cooking creates massive spikes in ultrafine particles (>0.3 µm), while dusting or walking around introduces coarser particles (>2.5 µm).

2. UNSUPERVISED MACHINE LEARNING (Gaussian Mixture Models - GMM)
   - Instead of you having to manually label what is "cooking" or "sleeping", GMM is an unsupervised algorithm that groups data points based on their statistical similarity.
   - Bayesian Information Criterion (BIC): The dashboard automatically tests different cluster counts on the fly and picks the optimal number of operational regimes your room is experiencing without crashing or guessing.

3. DIMENSIONALITY REDUCTION (Principal Component Analysis - PCA)
   - Because your data has 6 dimensions (5 particle channels + 1 size ratio), human eyes cannot look at a 6D graph.
   - PCA compresses those 6 channels down into 3 orthogonal axes (PC1, PC2, PC3) that capture the maximum variance. 
   - You can click and drag inside the bottom 3D graph with your mouse to rotate and zoom into these distinct behavioral clusters!

4. EXPORTING DATA & LABELS
   - When you click 'Export Dataset w/ ML', every single timestamp row saves both your raw counts and its corresponding ML 'regime' cluster number so you can review your home's behavioral history later.
"""
        txt_box.insert("0.0", guide_content)
        txt_box.configure(state="disabled")

    def refresh_com_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports()]
        if ports:
            self.com_dropdown.configure(values=ports)
            self.com_dropdown.set(ports[0])
            self.log_message(f"Found ports: {', '.join(ports)}")
        else:
            self.com_dropdown.configure(values=["No Ports Found"])
            self.com_dropdown.set("No Ports Found")
            self.log_message("No serial ports detected.")

    def toggle_connection(self):
        if not self.is_running:
            selected_port = self.com_dropdown.get()
            if selected_port == "No Ports Found":
                self.log_message("Connection failed: No port selected.")
                return
            try:
                self.log_message(f"Opening {selected_port} at 115,200 baud...")
                self.ser = serial.Serial(selected_port, 115200, timeout=1)
                self.is_running = True
                self.btn_connect.configure(text="⏹ Stop Stream", fg_color="red")
                self.btn_stats.configure(state="normal")
                self.btn_export.configure(state="normal")
                
                threading.Thread(target=self.serial_listener_worker, daemon=True).start()
                self.log_message("Connected! Listening for raw data...")
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
                    if raw_line:
                        self.log_message(f"RX: {raw_line}")
                    
                    if "Sensor state" in raw_line:
                        self.current_reading = {"timestamp": time.time()}
                    
                    match = re.search(r'>\s*([\d\.]+)\s*um:\s*(\d+)', raw_line)
                    if match and self.current_reading:
                        label = match.group(1)
                        val = int(match.group(2))
                        self.current_reading[f"pm_{label}"] = val
                    
                    if "---" in raw_line and "pm_0.3" in self.current_reading:
                        data_point = {
                            "time": self.current_reading["timestamp"],
                            "pm03": self.current_reading.get("pm_0.3", 0),
                            "pm05": self.current_reading.get("pm_0.5", 0),
                            "pm10": self.current_reading.get("pm_1.0", 0),
                            "pm25": self.current_reading.get("pm_2.5", 0),
                            "pm50": self.current_reading.get("pm_5.0", 0),
                            "regime": -1  # Placeholder before ML evaluation
                        }
                        self.data_history.append(data_point)
                        self.full_log.append(data_point)
                        self.after(0, self.update_dashboard_view, data_point)
            except Exception as e:
                self.log_message(f"Receiver Error: {e}")
            time.sleep(0.01)

    def update_dashboard_view(self, last_packet):
        self.lbl_pm03.configure(text=f"> 0.3 um: {last_packet['pm03']:,} pcs/m3")
        self.lbl_pm05.configure(text=f"> 0.5 um: {last_packet['pm05']:,} pcs/m3")
        self.lbl_pm10.configure(text=f"> 1.0 um: {last_packet['pm10']:,} pcs/m3")
        self.lbl_pm25.configure(text=f"> 2.5 um: {last_packet['pm25']:,} pcs/m3")
        self.lbl_pm50.configure(text=f"> 5.0 um: {last_packet['pm50']:,} pcs/m3")

        # Clear axes
        self.ax1.clear()
        self.ax2.clear()

        # Re-style Top Plot
        self.ax1.set_facecolor('#1a1a1a')
        self.ax1.tick_params(colors='white')
        self.ax1.spines['bottom'].set_color('white')
        self.ax1.spines['left'].set_color('white')
        self.ax1.spines['top'].set_visible(False)
        self.ax1.spines['right'].set_visible(False)
        self.ax1.grid(True, color='#444444', linestyle='--')
        self.ax1.set_title("Real-Time Particle Counts (Time Series)", color='white', weight='bold')
        self.ax1.set_xlabel("Elapsed Time (s)", color='white')
        self.ax1.set_ylabel("Particles / m3", color='white')

        # Re-style Bottom 3D Plot
        self.ax2.set_facecolor('#242424')
        self.ax2.xaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
        self.ax2.yaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
        self.ax2.zaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
        self.ax2.tick_params(colors='white')
        self.ax2.set_title("Interactive 3D PCA Cluster Space (PC1, PC2, PC3)", color='white', weight='bold')
        self.ax2.set_xlabel("PC 1", color='white', labelpad=8)
        self.ax2.set_ylabel("PC 2", color='white', labelpad=8)
        self.ax2.set_zlabel("PC 3", color='white', labelpad=8)

        if len(self.data_history) > 0:
            df = pd.DataFrame(list(self.data_history))
            x_time = df['time'] - df['time'].iloc[0]

            # Top Plot: Time Series
            self.ax1.plot(x_time, df['pm03'], label="> 0.3 um", color="#2ca02c")
            self.ax1.plot(x_time, df['pm05'], label="> 0.5 um", color="#1f77b4")
            self.ax1.plot(x_time, df['pm10'], label="> 1.0 um", color="#9467bd")
            self.ax1.plot(x_time, df['pm25'], label="> 2.5 um", color="#ff7f0e")
            self.ax1.plot(x_time, df['pm50'], label="> 5.0 um", color="#d62728")
            self.ax1.legend(facecolor='#242424', labelcolor='white', loc="upper left", fontsize='small')

            # Bottom Plot: 3D PCA Clustering
            if len(df) >= 4:
                try:
                    df['ratio_03_25'] = df['pm03'] / (df['pm25'] + 1)
                    features = ['pm03', 'pm05', 'pm10', 'pm25', 'pm50', 'ratio_03_25']
                    X = df[features].fillna(0).values
                    
                    if np.all(X == X[0,:]):
                        self.lbl_regime_status.configure(text="Regime: Stabilizing...", text_color="#aaaaaa")
                    else:
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        
                        pca = PCA(n_components=3)
                        X_pca = pca.fit_transform(X_scaled)

                        best_gmm = None
                        lowest_bic = np.inf
                        best_n = 2
                        for n in range(1, min(4, len(df))):
                            try:
                                gmm_test = GaussianMixture(n_components=n, random_state=42, covariance_type='spherical', reg_covar=1e-3)
                                gmm_test.fit(X_scaled)
                                bic = gmm_test.bic(X_scaled)
                                if bic < lowest_bic:
                                    lowest_bic = bic
                                    best_gmm = gmm_test
                                    best_n = n
                            except:
                                continue

                        df['regime'] = best_gmm.predict(X_scaled)
                        
                        # Assign the calculated cluster back into the active historical record
                        if len(self.full_log) > 0:
                            self.full_log[-1]['regime'] = int(df['regime'].iloc[-1])

                        regime_colors = ["#2ca02c", "#ff7f0e", "#1f77b4", "#9467bd"]
                        for r_idx in range(best_n):
                            r_subset = df[df['regime'] == r_idx]
                            indices = r_subset.index
                            r_pca = X_pca[indices]
                            self.ax2.scatter(
                                r_pca[:, 0], r_pca[:, 1], r_pca[:, 2],
                                label=f"Cluster {r_idx}",
                                color=regime_colors[r_idx % len(regime_colors)],
                                s=50, depthshade=True, alpha=0.9
                            )

                        self.ax2.legend(facecolor='#242424', labelcolor='white', loc="upper left", fontsize='small')

                        current_regime = int(df['regime'].iloc[-1])
                        active_color = regime_colors[current_regime % len(regime_colors)]
                        self.lbl_regime_status.configure(
                            text=f"Active Regime State: #{current_regime} ({best_n} active clusters)", 
                            text_color=active_color
                        )
                except Exception as ml_err:
                    self.lbl_regime_status.configure(text="Regime: Learning...", text_color="#aaaaaa")
            else:
                self.lbl_regime_status.configure(text=f"Regime: Collecting ({len(df)}/4)...", text_color="#aaaaaa")

        self.canvas.draw()

    def calculate_statistics(self):
        if not self.full_log:
            return
        df = pd.DataFrame(self.full_log)
        
        stats_window = ctk.CTkToplevel(self)
        stats_window.title("Descriptive Analysis Report")
        stats_window.geometry("600x450")
        stats_window.attributes("-topmost", True)

        txt_box = ctk.CTkTextbox(stats_window, font=ctk.CTkFont(family="Courier", size=12))
        txt_box.pack(fill="both", expand=True, padx=15, pady=15)

        summary_buffer = "==== DATASETS ANALYTICAL SUMMARY ====\n\n"
        summary_buffer += f"Total Samples Evaluated: {len(df)} counts\n\n"
        summary_buffer += "PARTICLE COUNT STATISTICS (particles/m3):\n"
        summary_buffer += df[['pm03', 'pm05', 'pm10', 'pm25', 'pm50']].describe().to_string()

        txt_box.insert("0.0", summary_buffer)
        self.log_message("Generated descriptive statistics report window.")

    def export_data_file(self):
        if not self.full_log:
            return
        df = pd.DataFrame(self.full_log)
        filename = f"nextpm_particle_log_with_ml_{int(time.time())}.csv"
        df.to_csv(filename, index=False)
        self.log_message(f"Exported dataset with ML clusters to {filename}")

if __name__ == "__main__":
    app = NextPMDashboard()
    app.mainloop()