import sys
import statistics

def calc_stats(values):
    """Calculate the average, the median, and the std deviation of a list of float values."""
    values = [float(v) for v in values]
    avg = statistics.mean(values)
    median = statistics.median(values)
    std_dev = statistics.stdev(values) if len(values) > 1 else 0.0  # Handling single-value case
    return f"{avg:.6f},{median:.6f},{std_dev:.6f}"

def main():
    """Main function to parse arguments and calculate statistics."""
    if len(sys.argv) != 4:
        print("Error: Incorrect number of arguments.")
        print("Usage: python calc_stats.py <wall_times> <cpu_times> <gflops>")
        sys.exit(1)

    # Read arguments and split into lists
    wall_times = sys.argv[1].split()
    cpu_times = sys.argv[2].split()
    gflops = sys.argv[3].split()

    # Calculate stats
    result = (
        calc_stats(wall_times) + "," +
        calc_stats(cpu_times) + "," +
        calc_stats(gflops)
    )

    print(result)

if __name__ == "__main__":
    main()
