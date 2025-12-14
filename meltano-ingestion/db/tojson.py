import csv

input_file = "olist_orders_dataset.csv"
output_file = "olist_orders_dataset_clean.csv"

with open(input_file, newline='', encoding='utf-8') as infile, \
     open(output_file, 'w', newline='', encoding='utf-8') as outfile:

    # Read the original CSV (handles inconsistent quotes)
    reader = csv.DictReader(infile)
    
    # Write out a normalized CSV with consistent quoting
    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    
    for row in reader:
        writer.writerow(row)

print(f"Normalized CSV saved to {output_file}")