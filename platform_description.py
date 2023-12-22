import platform
import psutil
from psutil import virtual_memory
import torch


def get_cpu_info():
    ram_gb = virtual_memory().total / 1e9
    cpu_info = {
        "Processor": platform.processor(),
        "Physical cores": psutil.cpu_count(logical=False),
        "Total cores": psutil.cpu_count(logical=True),
        "Memory": ram_gb,
        "Max Frequency": f"{psutil.cpu_freq().current:.2f}Mhz"
    }
    return cpu_info


def get_gpu_info():
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info["CUDA Version"] = torch.version.cuda
        gpu_info["Number of GPUs"] = torch.cuda.device_count()
        gpu_info["GPU Name"] = torch.cuda.get_device_name(0)
    return gpu_info


def create_summary(cpu_info, gpu_info):
    summary_parts = [
        f"{cpu_info['Total cores']} CPU cores {cpu_info['Processor']}",
        f"{cpu_info['Memory']:.2f} GB",
    ]

    if 'Number of GPUs' in gpu_info and gpu_info['Number of GPUs'] > 0:
        summary_parts.append(f"{gpu_info['Number of GPUs']} {gpu_info['GPU Name']}")

    return " - ".join(summary_parts)


def get_hardware_descriptor() -> dict:
    # Gather CPU and GPU information
    cpu_info = get_cpu_info()
    gpu_info = get_gpu_info()

    # Create a summary string
    summary = create_summary(cpu_info, gpu_info)
    return {"description": summary, "cpu": cpu_info, "gpu": gpu_info}


if __name__ == "__main__":
    print(get_hardware_descriptor()["description"])
