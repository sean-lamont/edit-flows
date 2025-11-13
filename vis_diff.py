import csv
import difflib
import argparse
import sys
import os
import webbrowser


def create_diff_report(input_file, col1, col2, col3, output_file):
    """
    Reads a CSV, compares col1-vs-col2 and col2-vs-col3,
    and writes a single HTML diff report file.
    """

    # 1. Read the CSV data
    try:
        with open(input_file, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # Check if all columns exist
            if not all(c in reader.fieldnames for c in [col1, col2, col3]):
                print(f"❌ Error: One or more columns not found in CSV.")
                print(f"Your columns: {reader.fieldnames}")
                print(f"You asked for: '{col1}', '{col2}', and '{col3}'")
                sys.exit(1)

            # Read all rows into memory
            rows = list(reader)

    except FileNotFoundError:
        print(f"❌ Error: Input file not found at '{input_file}'")
        sys.exit(1)
    except Exception as e:
        print(f"❌ An error occurred reading the file: {e}")
        sys.exit(1)

    print(f"Found {len(rows)} rows. Generating 2 diffs per row...")

    # 2. Setup the HTML diff generator
    d = difflib.HtmlDiff(wrapcolumn=70)  # 70 chars wide before wrapping

    # 3. Build the HTML content
    html_parts = [
        f"<html><head><meta charset='utf-8'><title>Diff Report</title>",
        # Embed the CSS styles directly into the file
        f"<style>{d._styles}</style>",
        "</head><body>",
        f"<h1>Diff Report</h1>",
        f"<h3>Comparing '{col1}' vs '{col2}' and '{col2}' vs '{col3}'</h3>",
        f"<p>Source file: {input_file}</p><hr style='border-top: 3px solid #000;'>"
    ]

    # 4. Loop over each row and create two diff tables
    for i, row in enumerate(rows):
        html_parts.append(f"<h2>Row {i + 1}</h2>")

        # Get the text, default to empty string if missing
        text1 = str(row.get(col1, '')).splitlines()
        text2 = str(row.get(col2, '')).splitlines()
        text3 = str(row.get(col3, '')).splitlines()

        # --- Generate Diff 1 (col1 vs col2) ---
        html_parts.append(f"<h3>Comparison: {col1} vs {col2}</h3>")
        diff_table_1v2 = d.make_table(
            fromlines=text1,
            tolines=text2,
            fromdesc=f"{col1} (Row {i + 1})",
            todesc=f"{col2} (Row {i + 1})",
            context=True
        )
        html_parts.append(diff_table_1v2)

        # --- Generate Diff 2 (col2 vs col3) ---
        html_parts.append(f"<h3>Comparison: {col2} vs {col3}</h3>")
        diff_table_2v3 = d.make_table(
            fromlines=text2,  # Use text2 as the "from"
            tolines=text3,  # Use text3 as the "to"
            fromdesc=f"{col2} (Row {i + 1})",
            todesc=f"{col3} (Row {i + 1})",
            context = True
        )
        html_parts.append(diff_table_2v3)
        html_parts.append("<hr style='border-top: 2px solid #bbb;'>")  # Separator

    html_parts.append("</body></html>")
    final_html = "\n".join(html_parts)

    # 5. Write the final HTML file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final_html)
    except Exception as e:
        print(f"❌ Error writing output file: {e}")
        sys.exit(1)

    return os.path.abspath(output_file)


def main():
    parser = argparse.ArgumentParser(description="Generate an HTML diff report from three CSV columns.")

    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input CSV file."
    )
    parser.add_argument(
        "col1",
        type=str,
        help="Name of the first column."
    )
    parser.add_argument(
        "col2",
        type=str,
        help="Name of the second column (to compare with col1 and col3)."
    )
    parser.add_argument(
        "col3",
        type=str,
        help="Name of the third column (to compare with col2)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="diff_report.html",
        help="Name of the output HTML file (default: diff_report.html)."
    )
    parser.add_argument(
        "--no_open",
        action="store_true",
        help="Do not automatically open the report in a web browser."
    )

    args = parser.parse_args()

    # Run the function
    report_path = create_diff_report(
        args.input_file,
        args.col1,
        args.col2,
        args.col3,
        args.output
    )

    print(f"\n✅ Success! Diff report saved to:")
    print(f"{report_path}")

    # 6. Open in browser
    if not args.no_open:
        try:
            webbrowser.open_new_tab(f"file://{report_path}")
            print("Opening report in your default browser...")
        except Exception as e:
            print(f"Could not open browser: {e}")


if __name__ == "__main__":
    main()