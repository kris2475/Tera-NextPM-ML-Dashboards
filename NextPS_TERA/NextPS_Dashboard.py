import sys
import threading
import time
import collections
import re
import serial
import serial.tools.list_ports
import pandas as pd
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Global Theme setup
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class NextPMDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("TERA NextPM Real-Time Air Quality Dashboard")
        self.geometry("1200x750")

        # Serial & Data Management States
        self.ser = None
        self.is_running = False
        self.data_history = collections.deque(maxlen=60) # Store last 60 seconds for plot
        self.full_log = [] # Holds all data for statistics operations
        
        # Temp storage while reading a full text block from ESP32
        self.current_reading = {}

        self.setup_ui_layout()
        self.refresh_com_ports()

    def setup_ui_layout(self):
        # Left Control Panel Sidebar
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)

        ctk.CTkLabel(self.sidebar, text="CONNECTION CONTROLS", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=15)
        
        self.com_dropdown = ctk.CTkOptionMenu(self.sidebar, values=["No Ports Found"])
        self.com_dropdown.pack(pady=10, fill="x", padx=15)

        self.btn_refresh = ctk.CTkButton(self.sidebar, text="🔄 Refresh Ports", command=self.refresh_com_ports, fg_color="gray")
        self.btn_refresh.pack(pady=5, fill="x", padx=15)

        self.btn_connect = ctk.CTkButton(self.sidebar, text="🔌 Connect Stream", command=self.toggle_connection, fg_color="green")
        self.btn_connect.pack(pady=15, fill="x", padx=15)

        ctk.CTkLabel(self.sidebar, text="DATA ANALYSIS TOOLS", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=20)
        
        self.btn_stats = ctk.CTkButton(self.sidebar, text="📊 Run Descriptive Stats", command=self.calculate_statistics, state="disabled")
        self.btn_stats.pack(pady=5, fill="x", padx=15)

        self.btn_export = ctk.CTkButton(self.sidebar, text="💾 Export CSV Dataset", command=self.export_data_file, state="disabled", fg_color="purple")
        self.btn_export.pack(pady=5, fill="x", padx=15)

        # Main Panel Readouts (Organized in 2 rows to fit all 5 particle metrics cleanly)
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.metrics_row = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.metrics_row.pack(side="top", fill="x", pady=(0, 10))

        # Top row metrics (3 items)
        top_metrics = ctk.CTkFrame(self.metrics_row, fg_color="transparent")
        top_metrics.pack(fill="x", expand=True)

        self.lbl_pm03 = ctk.CTkLabel(top_metrics, text="> 0.3 um: -- pcs/m3", font=ctk.CTkFont(size=14, weight="bold"), text_color="#2ca02c")
        self.lbl_pm03.pack(side="left", expand=True, fill="both", padx=5, pady=2)

        self.lbl_pm05 = ctk.CTkLabel(top_metrics, text="> 0.5 um: -- pcs/m3", font=ctk.CTkFont(size=14, weight="bold"), text_color="#1f77b4")
        self.lbl_pm05.pack(side="left", expand=True, fill="both", padx=5, pady=2)

        self.lbl_pm10 = ctk.CTkLabel(top_metrics, text="> 1.0 um: -- pcs/m3", font=ctk.CTkFont(size=14, weight="bold"), text_color="#9467bd")
        self.lbl_pm10.pack(side="left", expand=True, fill="both", padx=5, pady=2)

        # Bottom row metrics (2 items)
        bottom_metrics = ctk.CTkFrame(self.metrics_row, fg_color="transparent")
        bottom_metrics.pack(fill="x", expand=True)

        self.lbl_pm25 = ctk.CTkLabel(bottom_metrics, text="> 2.5 um: -- pcs/m3", font=ctk.CTkFont(size=14, weight="bold"), text_color="#ff7f0e")
        self.lbl_pm25.pack(side="left", expand=True, fill="both", padx=5, pady=2)

        self.lbl_pm50 = ctk.CTkLabel(bottom_metrics, text="> 5.0 um: -- pcs/m3", font=ctk.CTkFont(size=14, weight="bold"), text_color="#d62728")
        self.lbl_pm50.pack(side="left", expand=True, fill="both", padx=5, pady=2)

        # Dynamic Graph Canvas
        self.fig, self.ax = plt.subplots(figsize=(6, 4), facecolor='#242424')
        self.ax.set_facecolor('#1a1a1a')
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.grid(True, color='#444444', linestyle='--')
        self.ax.set_title("Real-Time Particle Counts (Particles / m3)", color='white', weight='bold')
        self.ax.set_xlabel("Elapsed Time (s)", color='white')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def refresh_com_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports()]
        if ports:
            self.com_dropdown.configure(values=ports)
            self.com_dropdown.set(ports[0])
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
                self.btn_stats.configure(state="normal")
                self.btn_export.configure(state="normal")
                
                threading.Thread(target=self.serial_listener_worker, daemon=True).start()
            except Exception as e:
                print(f"Error connecting: {e}")
        else:
            self.is_running = False
            if self.ser:
                self.ser.close()
            self.btn_connect.configure(text="🔌 Connect Stream", fg_color="green")

    def serial_listener_worker(self):
        while self.is_running:
            try:
                if self.ser.in_waiting > 0:
                    raw_line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    
                    if "Sensor state" in raw_line:
                        self.current_reading = {"timestamp": time.time()}
                    
                    # Regex match to extract label and values from your specific logs
                    match = re.search(r'>\s*([\d\.]+)\s*um:\s*(\d+)', raw_line)
                    if match and self.current_reading:
                        label = match.group(1)
                        val = int(match.group(2))
                        self.current_reading[f"pm_{label}"] = val
                    
                    if "---" in raw_line and "pm_0.3" in self.current_reading:
                        # Full packet assembly complete
                        data_point = {
                            "time": self.current_reading["timestamp"],
                            "pm03": self.current_reading.get("pm_0.3", 0),
                            "pm05": self.current_reading.get("pm_0.5", 0),
                            "pm10": self.current_reading.get("pm_1.0", 0),
                            "pm25": self.current_reading.get("pm_2.5", 0),
                            "pm50": self.current_reading.get("pm_5.0", 0),
                        }
                        self.data_history.append(data_point)
                        self.full_log.append(data_point)
                        self.after(0, self.update_dashboard_view, data_point)
            except Exception as e:
                print(f"Receiver error: {e}")
            time.sleep(0.01)

    def update_dashboard_view(self, last_packet):
        self.lbl_pm03.configure(text=f"> 0.3 um: {last_packet['pm03']:,} pcs/m3")
        self.lbl_pm05.configure(text=f"> 0.5 um: {last_packet['pm05']:,} pcs/m3")
        self.lbl_pm10.configure(text=f"> 1.0 um: {last_packet['pm10']:,} pcs/m3")
        self.lbl_pm25.configure(text=f"> 2.5 um: {last_packet['pm25']:,} pcs/m3")
        self.lbl_pm50.configure(text=f"> 5.0 um: {last_packet['pm50']:,} pcs/m3")

        self.ax.clear()
        self.ax.grid(True, color='#444444', linestyle='--')
        self.ax.set_title("Real-Time Particle Counts (Particles / m3)", color='white', weight='bold')
        self.ax.set_xlabel("Elapsed Time (s)", color='white')
        self.ax.tick_params(colors='white')

        if len(self.data_history) > 1:
            df = pd.DataFrame(list(self.data_history))
            # Correct X-axis: Calculate elapsed seconds from the start of the window
            x = df['time'] - df['time'].iloc[0]

            # Plot all 5 size traces with distinct colors
            self.ax.plot(x, df['pm03'], label="> 0.3 um", color="#2ca02c")
            self.ax.plot(x, df['pm05'], label="> 0.5 um", color="#1f77b4")
            self.ax.plot(x, df['pm10'], label="> 1.0 um", color="#9467bd")
            self.ax.plot(x, df['pm25'], label="> 2.5 um", color="#ff7f0e")
            self.ax.plot(x, df['pm50'], label="> 5.0 um", color="#d62728")
            
            self.ax.legend(facecolor='#242424', labelcolor='white', loc="upper left", fontsize='small')

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

    def export_data_file(self):
        if not self.full_log:
            return
        df = pd.DataFrame(self.full_log)
        filename = f"nextpm_particle_log_{int(time.time())}.csv"
        df.to_csv(filename, index=False)
        print(f"Dataset saved inside project directory as: {filename}")

if __name__ == "__main__":
    app = NextPMDashboard()
    app.mainloop()

