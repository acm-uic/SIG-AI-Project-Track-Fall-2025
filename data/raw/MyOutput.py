# Open the file you want to read from
with open("USA Housing Dataset.csv", "r") as infile:
    # Open the file you want to write to
    with open("output.txt", "w") as outfile:
        # Loop through each line in the input file
        for line in infile:
            # Remove the newline at the end (optional)
            line = line.strip()
            # You can process or modify the line here
            new_line = f"Processed: {line}\n"

            # Write the modified line to the output file
            outfile.write(new_line)
