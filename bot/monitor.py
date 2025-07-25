import psutil
import GPUtil
import time
import threading


from utils.project_path import PROJECT_ROOT  # AUTO PATH CONVERTED
class ResourceMonitor:
    def __init__(self, ram_max=90, cpu_max=90, gpu_max=90, gpu_temp_max=85, check_interval=10, 
                 action="pause", log_file=None, verbose=True):
        """
        :param ram_max: Maksimal RAM-forbrug i %
        :param cpu_max: Maksimal CPU-forbrug i %
        :param gpu_max: Maksimal GPU-forbrug i %
        :param gpu_temp_max: Maksimal GPU-temperatur i Celsius
        :param check_interval: Hvor ofte overvågningen kører (sekunder)
        :param action: 'pause', 'kill', 'warn'
        :param log_file: Filnavn til log (valgfrit)
        :param verbose: Print status til konsol
        """
        self.ram_max = ram_max
        self.cpu_max = cpu_max
        self.gpu_max = gpu_max
        self.gpu_temp_max = gpu_temp_max
        self.check_interval = check_interval
        self.action = action
        self.log_file = log_file
        self.verbose = verbose
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        
    def start(self):
        self._thread.start()
        if self.verbose:
            print("[Monitor] Overvågning startet.")

    def stop(self):
        self._stop_event.set()
        if self.verbose:
            print("[Monitor] Overvågning stoppet.")

    def _monitor(self):
        while not self._stop_event.is_set():
            ram = psutil.virtual_memory().percent
            cpu = psutil.cpu_percent(interval=1)
            
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Hvis flere GPUer, brug den første
                gpu_load = gpu.load * 100  # 0–100%
                gpu_temp = gpu.temperature  # grader celsius
            else:
                gpu_load, gpu_temp = 0, 0

            msg = (f"[Monitor] RAM: {ram:.1f}% | CPU: {cpu:.1f}% | "
                   f"GPU: {gpu_load:.1f}% | GPU-temp: {gpu_temp:.1f}C")
            
            if self.verbose:
                print(msg)

            # === Rettelse: Opret mappe hvis nødvendigt ===
            if self.log_file:
                log_dir = os.path.dirname(self.log_file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
                with open(self.log_file, "a") as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{ram},{cpu},{gpu_load},{gpu_temp}\n")

            # ACTIONS VED OVERLOAD
            if ram > self.ram_max or cpu > self.cpu_max or gpu_load > self.gpu_max or gpu_temp > self.gpu_temp_max:
                overload_reason = []
                if ram > self.ram_max:
                    overload_reason.append("RAM")
                if cpu > self.cpu_max:
                    overload_reason.append("CPU")
                if gpu_load > self.gpu_max:
                    overload_reason.append("GPU")
                if gpu_temp > self.gpu_temp_max:
                    overload_reason.append("GPU-temp")
                overload_msg = f"🚨 RESSOURCE-ALARM ({', '.join(overload_reason)}) 🚨"
                print(overload_msg)
                if self.action == "warn":
                    pass  # Bare advar
                elif self.action == "pause":
                    print("[Monitor] Pauser træning/datajob i 60 sek...")
                    time.sleep(60)
                elif self.action == "kill":
                    print("[Monitor] Lukker process pga. overload...")
                    os._exit(1)
                else:
                    print("[Monitor] Ukendt action – fortsætter.")
            time.sleep(self.check_interval)

# Eksempel på brug
if __name__ == "__main__":
    monitor = ResourceMonitor(
        ram_max=80,
        cpu_max=90,
        gpu_max=95,
        gpu_temp_max=80,
        check_interval=10,
        action="pause",  # Kan også være "kill" eller "warn"
        log_file=PROJECT_ROOT / "outputs" / "debug/resource_log.csv"  # AUTO PATH CONVERTED
    )
    monitor.start()
    # Din træningskode her – monitoren kører i baggrunden
    for i in range(1000):
        print(f"Dummy main loop {i}")
        time.sleep(3)
    monitor.stop()