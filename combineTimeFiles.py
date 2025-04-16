import os
import csv


def aggregate_second_column(folder_path="time2", output_csv="time2/aggregated_results_time.csv"):
    """
    For every CSV file in `folder_path`, compute the average and sum of values
    in the second column (index=1). Write these to `output_csv`.
    """
    # Open the output CSV and write a header row
    with open(output_csv, "w", newline="") as outf:
        writer = csv.writer(outf)
        writer.writerow(["file_name", "average_time_per_graph", "total_time"])

        # Iterate over all files in the target folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(folder_path, filename)

                # Collect values from the second column of this file
                second_col_values = []

                with open(file_path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)

                    # If your CSV files have a header and you want to skip it:
                    next(reader, None)  # Uncomment if needed

                    for row in reader:
                        # Ensure there's at least a second column
                        if len(row) >= 2:
                            try:
                                val = float(row[1])  # read second column
                                second_col_values.append(val)
                            except ValueError:
                                # If the cell is not convertible to float, skip it
                                continue

                # Compute average and sum
                if second_col_values:
                    sum_val = sum(second_col_values)
                    avg_val = sum_val / len(second_col_values)
                else:
                    sum_val = 0
                    avg_val = 0

                # Write one row per file
                writer.writerow([filename, avg_val, sum_val])

    print(f"Aggregation complete. Results saved to {output_csv}")

# Example usage:
aggregate_second_column("time2", "time2/aggregated_results_time.csv")