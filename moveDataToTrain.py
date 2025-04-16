# Example filenames (change these as needed)

#groudn truth data
# train_file = "data/train/synthetic.train"
# other_values_file = "data/train/importance_synthetic.csv"
# output_file = "data/train/synthetic_u.train"

# Leiden Algorithm
# train_file = "data/train_leiden_test/synthetic.train"
# other_values_file = "data/train_leiden_test/importance_synthetic_testing.csv"
# output_file = "data/train_leiden_test/synthetic_u.train"

# # SAF Algorithm
train_file = "data/train_SAF_adj_long/synthetic.train"
other_values_file = "data/train_SAF_adj_long/importance_synthetic.csv"
output_file = "data/train_SAF_adj_long/synthetic_u.train"

# SAF Algorithm
# train_file = "data/train_gnn_LRP_long/synthetic.train"
# other_values_file = "data/train_gnn_LRP_long/importance_synthetic.csv"
# output_file = "data/train_gnn_LRP_long/synthetic_u.train"

# Step 1: Load the "other values" data into a dictionary
other_values_dict = {}
with open(other_values_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # Split by comma
        parts = line.split(',')

        # The first part is the graph filename
        graph_filename = parts[0].strip()
        # The rest are numeric values
        numeric_values = parts[1:]

        # Store in dictionary
        other_values_dict[graph_filename] = numeric_values

# Step 2: Process the train file and substitute values
with open(train_file, 'r') as f_in, open(output_file, 'w') as f_out:
    for line in f_in:
        line = line.strip()
        if not line:
            continue
        # Split by comma
        parts = line.split(',')

        # The first part is the graph filename from the train file
        graph_filename = parts[0].strip()

        # If this graph filename exists in the other_values_dict, substitute
        if graph_filename in other_values_dict:
            new_values = other_values_dict[graph_filename]
            # Write the new line: the filename + the replaced numeric values
            new_line = graph_filename + ',' + ','.join(new_values) + '\n'
            f_out.write(new_line)
        else:
            # If no substitution is found, write the line as is
            print("No substitution found for: ", graph_filename)
            f_out.write(line + '\n')
