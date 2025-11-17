"""

A utility script for visualizing the differences between two sequences of code.

This is useful for analyzing the model's performance by comparing the original code,
the ground truth corrected code, and the model's generated output.
It generates a HTML diff for easy viewing.

"""

import csv
import difflib
import argparse
import sys
import os
import webbrowser

# Add NLTK for BLEU score calculation
try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    NLTK_AVAILABLE = True
    # Use a standard smoothing function to avoid 0.0 scores on short/no-match text
    SMOOTHER = SmoothingFunction().method7
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: 'nltk' library not found. BLEU score metrics will be skipped.")


def _calculate_metrics(tp, fp, fn):
    """
    Helper function to safely calculate Precision, Recall, and F1-Score
    based on hunk content counts.
    """
    # Precision: Of all edit hunks made, how many were correct?
    if (tp + fp) == 0:
        precision = 1.0  # No edits made, no incorrect edits.
    else:
        precision = tp / (tp + fp)

    # Recall: Of all required edit hunks, how many were made?
    if (tp + fn) == 0:
        recall = 1.0  # No edits required, no missed edits.
    else:
        recall = tp / (tp + fn)

    # F1-Score: The harmonic mean of Precision and Recall
    if (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1


def _get_diff_content(s, text_from, text_to):
    """
    Extracts the sets of *content* that were added or deleted.
    """
    deleted_content = set()
    added_content = set()

    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'delete':
            deleted_content.add(text_from[i1:i2])
        elif tag == 'insert':
            added_content.add(text_to[j1:j2])
        elif tag == 'replace':
            deleted_content.add(text_from[i1:i2])
            added_content.add(text_to[j1:j2])

    return deleted_content, added_content


def create_diff_report(input_file, col1, col2, col3, output_file):
    """
    Reads a CSV, performs a character-level "diff-of-diffs" analysis
    based on diff *content*, and writes an HTML diff report.

    - col1: Buggy Code
    - col2: Correct Code (Ground Truth)
    - col3: Attempted Fix (Hypothesis)
    """

    # 1. Read the CSV data (Unchanged)
    try:
        with open(input_file, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if not all(c in reader.fieldnames for c in [col1, col2, col3]):
                print(f"❌ Error: One or more columns not found in CSV.")
                print(f"Your columns: {reader.fieldnames}")
                print(f"You asked for: '{col1}', '{col2}', and '{col3}'")
                sys.exit(1)
            rows = list(reader)
    except FileNotFoundError:
        print(f"❌ Error: Input file not found at '{input_file}'")
        sys.exit(1)
    except Exception as e:
        print(f"❌ An error occurred reading the file: {e}")
        sys.exit(1)

    print(f"Found {len(rows)} rows. Generating diffs and metrics...")

    # 2. Setup the HTML diff generator (Unchanged)
    d_html = difflib.HtmlDiff(wrapcolumn=70)  # For visual HTML tables

    # 3. Initialize collectors for metrics and HTML parts (Unchanged)
    total_tp_hunks = 0
    total_fp_hunks = 0
    total_fn_hunks = 0
    total_correct_adds = 0
    total_correct_dels = 0
    total_perfect_fixes = 0

    # [UPDATED] Collectors for BLEU metrics
    total_bleu_abs_gain = 0

    html_row_parts = []  # To store HTML for each row

    # 4. Loop over each row to create diffs AND calculate metrics
    for i, row in enumerate(rows):
        # --- Get text for diffs (Unchanged) ---
        text1_lines = str(row.get(col1, '')).splitlines()
        text2_lines = str(row.get(col2, '')).splitlines()
        text3_lines = str(row.get(col3, '')).splitlines()

        col1_text = str(row.get(col1, ''))
        col2_text = str(row.get(col2, ''))
        col3_text = str(row.get(col3, ''))

        # --- "Diff-of-Diffs" Metrics (Unchanged) ---
        s_gt = difflib.SequenceMatcher(None, col1_text, col2_text, autojunk=False)
        gt_deleted_content, gt_added_content = _get_diff_content(s_gt, col1_text, col2_text)

        s_fix = difflib.SequenceMatcher(None, col1_text, col3_text, autojunk=False)
        fix_deleted_content, fix_added_content = _get_diff_content(s_fix, col1_text, col3_text)

        row_correct_dels = len(gt_deleted_content.intersection(fix_deleted_content))
        row_correct_adds = len(gt_added_content.intersection(fix_added_content))
        row_fp_dels = len(fix_deleted_content.difference(gt_deleted_content))
        row_fp_adds = len(fix_added_content.difference(gt_added_content))
        row_fn_dels = len(gt_deleted_content.difference(fix_deleted_content))
        row_fn_adds = len(gt_added_content.difference(fix_added_content))

        row_tp = row_correct_dels + row_correct_adds
        row_fp = row_fp_dels + row_fp_adds
        row_fn = row_fn_dels + row_fn_adds

        row_precision, row_recall, row_f1 = _calculate_metrics(row_tp, row_fp, row_fn)
        row_perfect_fix = 1 if col2_text == col3_text else 0

        # Update grand totals for diff-of-diffs (Unchanged)
        total_tp_hunks += row_tp
        total_fp_hunks += row_fp
        total_fn_hunks += row_fn
        total_correct_adds += row_correct_adds
        total_correct_dels += row_correct_dels
        total_perfect_fixes += row_perfect_fix

        # --- [UPDATED] Calculate BLEU Score Metrics ---
        row_bleu_html = ""
        if NLTK_AVAILABLE:
            # Tokenize text for BLEU
            ref_tokens = col2_text.split()  # Reference = Correct code
            hyp_initial_tokens = col1_text.split()  # Hypothesis 1 = Buggy code
            hyp_attempt_tokens = col3_text.split()  # Hypothesis 2 = Attempted fix

            # Calculate BLEU scores against the reference
            bleu_initial = sentence_bleu([ref_tokens], hyp_initial_tokens, smoothing_function=SMOOTHER)
            bleu_attempt = sentence_bleu([ref_tokens], hyp_attempt_tokens, smoothing_function=SMOOTHER)

            # Cap scores at 1.0 to avoid smoothing/float artifacts > 100%
            bleu_initial = min(1.0, bleu_initial)
            bleu_attempt = min(1.0, bleu_attempt)

            # Calculate Absolute BLEU Gain
            bleu_abs_gain = bleu_attempt - bleu_initial

            # Update total for absolute gain
            total_bleu_abs_gain += bleu_abs_gain

            # Build HTML for this row's BLEU metrics
            row_bleu_html = (
                f"<hr style='border-top: 1px dashed #999; margin: 10px 0;'>"
                f"<b>BLEU Score Metrics (vs Correct Code)</b>"
                "<ul>"
                f"<li><b>Baseline BLEU (Buggy):</b> {bleu_initial:.2%}</li>"
                f"<li><b>Attempt BLEU:</b> {bleu_attempt:.2%}</li>"
                f"<li><b>Absolute BLEU Gain:</b> <b style='color: {'green' if bleu_abs_gain > 0 else 'red' if bleu_abs_gain < 0 else 'black'}'>{bleu_abs_gain:+.2%}</b></li>"
                "</ul>"
            )

        # --- Build this row's HTML string (Unchanged) ---
        current_row_html = []
        current_row_html.append(f"<h2>Row {i + 1}</h2>")

        # --- Add the metrics summary box for this row (Unchanged) ---
        current_row_html.append(
            f"<div style='background-color: #f0f0f0; border: 1px solid #ccc; padding: 10px; margin-bottom: 15px;'>")
        current_row_html.append(f"<h4>'Diff-of-Diffs' Metrics (Content-Hunk Level)</h4>")
        current_row_html.append("<p>Comparing edit content from (Buggy -> Correct) vs (Buggy -> Attempt)</p>")
        current_row_html.append("<ul>")
        current_row_html.append(f"<li><b>Correct Addition Hunks (TP-Add):</b> {row_correct_adds}</li>")
        current_row_html.append(f"<li><b>Correct Deletion Hunks (TN-Del):</b> {row_correct_dels}</li>")
        current_row_html.append(
            f"<li><b>Incorrect Edit Hunks (FP):</b> {row_fp} (Add: {row_fp_adds}, Del: {row_fp_dels})</li>")
        current_row_html.append(
            f"<li><b>Missed Edit Hunks (FN):</b> {row_fn} (Add: {row_fn_adds}, Del: {row_fn_dels})</li>")
        current_row_html.append(f"<li style='margin-top: 8px;'><b>Hunk Precision:</b> {row_precision:.2%}</li>")
        current_row_html.append(f"<li><b>Hunk Recall:</b> {row_recall:.2%}</li>")
        current_row_html.append(f"<li><b>Hunk F1-Score:</b> {row_f1:.2%}</li>")
        current_row_html.append(
            f"<li style='margin-top: 8px;'><b>Perfect Fix (Exact Match):</b> {'✅ Yes' if row_perfect_fix else '❌ No'}</li>")
        current_row_html.append("</ul>")

        # Add BLEU HTML to the box if available
        current_row_html.append(row_bleu_html)

        current_row_html.append("</div>")  # Close metrics box

        # --- Generate Diff 1 (Buggy vs Correct) (Unchanged) ---
        diff_table_1v2 = d_html.make_table(
            fromlines=text1_lines,
            tolines=text2_lines,
            fromdesc=f"{col1} (Buggy - Row {i + 1})",
            todesc=f"{col2} (Correct - Row {i + 1})",
            context=True
        )
        current_row_html.append("<h3>Ground Truth Diff (Buggy vs Correct)</h3>")
        current_row_html.append(diff_table_1v2)

        # --- Generate Diff 2 (Correct vs Attempt) (Unchanged) ---
        diff_table_2v3 = d_html.make_table(
            fromlines=text2_lines,
            tolines=text3_lines,
            fromdesc=f"{col2} (Correct - Row {i + 1})",
            todesc=f"{col3} (Attempt - Row {i + 1})",
            context=True
        )
        current_row_html.append("<h3>Attempt vs Correct Diff (How close was the fix?)</h3>")
        current_row_html.append(diff_table_2v3)
        current_row_html.append("<hr style='border-top: 2px solid #bbb;'>" + "\n")

        # Add this row's complete HTML to the main list
        html_row_parts.append("".join(current_row_html))

    # 5. Build the *final* HTML content (Unchanged)
    html_parts = [
        f"<html><head><meta charset='utf-8'><title>Diff Report</title>",
        f"<style>{d_html._styles}</style>",
        "</head><body>",
        f"<h1>Code Repair Diff Report</h1>",
        f"<h3>Comparing '{col1}' (Buggy) vs '{col2}' (Correct) vs '{col3}' (Attempt)</h3>",
        f"<p>Source file: {input_file} ({len(rows)} rows processed)</p>",
    ]

    # --- [UPDATED] Add the OVERALL SUMMARY section ---

    # Calculate overall diff-of-diffs metrics
    overall_precision, overall_recall, overall_f1 = _calculate_metrics(total_tp_hunks, total_fp_hunks, total_fn_hunks)
    perfect_fix_rate = total_perfect_fixes / len(rows) if rows else 0.0

    html_parts.append(
        "<div style='background-color: #e6f7ff; border: 2px solid #0056b3; padding: 15px; margin-bottom: 20px;'>")
    html_parts.append(f"<h2>Overall 'Diff-of-Diffs' Metrics (Content-Hunk Level)</h2>")
    html_parts.append(
        "<p>These are the 'micro-averaged' metrics, calculated from the total content hunk counts (TP, FP, FN) across all rows.</p>")
    html_parts.append("<ul style='font-size: 1.1em;'>")
    html_parts.append(f"<li><b>Total Correct Addition Hunks:</b> {total_correct_adds}</li>")
    html_parts.append(f"<li><b>Total Correct Deletion Hunks:</b> {total_correct_dels}</li>")
    html_parts.append(f"<li style='margin-top: 8px;'><b>Total Incorrect/Spurious Hunks (FP):</b> {total_fp_hunks}</li>")
    html_parts.append(f"<li><b>Total Missed Hunks (FN):</b> {total_fn_hunks}</li>")
    html_parts.append(f"<li style='margin-top: 10px;'><b>Overall Hunk Precision:</b> {overall_precision:.2%}</li>")
    html_parts.append(f"<li><b>Overall Hunk Recall:</b> {overall_recall:.2%}</li>")
    html_parts.append(f"<li><b>Overall Hunk F1-Score:</b> {overall_f1:.2%}</li>")
    html_parts.append(
        f"<li style='margin-top: 10px;'><b>Total Perfect Fixes:</b> {total_perfect_fixes} out of {len(rows)} ({perfect_fix_rate:.2%})</li>")
    html_parts.append("</ul>")

    # [UPDATED] Add overall BLEU summary if available
    if NLTK_AVAILABLE:
        # Calculate overall average absolute gain
        avg_bleu_gain = total_bleu_abs_gain / len(rows) if rows else 0.0

        html_parts.append(
            f"<hr style='border-top: 1px solid #0056b3; margin: 15px 0;'>"
            f"<h2 style='margin-bottom: 5px;'>Overall BLEU Score Metrics</h2>"
            f"<p>This metric shows the average absolute change in BLEU score from 'Buggy' to 'Attempt' (relative to 'Correct').</p>"
            f"<ul style='font-size: 1.1em;'>"
            f"<li><b>Overall Average BLEU Gain: <span style='color: {'green' if avg_bleu_gain > 0 else 'red' if avg_bleu_gain < 0 else 'black'}'>{avg_bleu_gain:+.2%}</span></b></li>"
            f"</ul>"
        )

    html_parts.append("</div>")  # Close overall summary box
    html_parts.append("<hr style='border-top: 3px solid #000;'>")

    # 6. Add all the individual row-by-row diffs (Unchanged)
    html_parts.append("".join(html_row_parts))

    # 7. Close HTML (Unchanged)
    html_parts.append("</body></html>")
    final_html = "".join(html_parts)

    # 8. Write the final HTML file (Unchanged)
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
        help="Name of the 'Buggy Code' column."
    )
    parser.add_argument(
        "col2",
        type=str,
        help="Name of the 'Correct Code' (Ground Truth) column."
    )
    parser.add_argument(
        "col3",
        type=str,
        help="Name of the 'Attempted Fix' column."
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

    # 6. Open in browser (Unchanged)
    if not args.no_open:
        try:
            webbrowser.open_new_tab(f"file://{report_path}")
            print("Opening report in your default browser...")
        except Exception as e:
            print(f"Could not open browser: {e}")


if __name__ == "__main__":
    main()