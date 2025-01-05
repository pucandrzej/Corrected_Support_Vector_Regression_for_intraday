import psutil

def wait_for_low_cpu(threshold=10, interval=300):
    """
    Waits until the CPU usage is below the threshold for a specified duration.

    :param threshold: CPU usage percentage to consider as low (default: 10%)
    :param interval: Interval in seconds to check the CPU usage (default: 5 seconds)
    """
    low_cpu_start = None  # Start time when CPU is below the threshold

    while True:
        # Get current CPU usage
        cpu_usage = psutil.cpu_percent(interval=interval)

        print(f"CPU usage: {cpu_usage}")

        if cpu_usage < threshold:
            break

if __name__ == "__main__":
    wait_for_low_cpu()
