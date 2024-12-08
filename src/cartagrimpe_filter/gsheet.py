"""Google Spreadsheet handling."""

from collections.abc import Sequence

import gspread
import numpy as np
import pandas as pd
from gspread.utils import rowcol_to_a1

g_spread_client = gspread.service_account()  # pyright:ignore[reportPrivateImportUsage]


def update_spreadsheet(
    df: pd.DataFrame,
    sheet_key: str,
    raw_data_worksheet_title: str = "Raw",
    other_worksheet_titles: Sequence[str] = ("Full", "Events"),
) -> None:
    """Update the raw data sheet."""
    sheet = g_spread_client.open_by_key(sheet_key)
    raw_worksheet = sheet.worksheet(raw_data_worksheet_title)

    initial_row_count = raw_worksheet.row_count

    # Replace nans with None
    values = [
        [None if isinstance(x, float) and np.isnan(x) else x for x in row]
        for row in df.values.tolist()
    ]

    nb_row, nb_col = df.values.shape
    new_row_count = nb_row + 1
    # Range starts at A2 to avoid column titles
    range_name = f"A2:{rowcol_to_a1(new_row_count, nb_col)}"
    raw_worksheet.update(values, range_name=range_name, raw=False)

    # Remove other rows if there are now less rows than before
    if (first_empty_row_idx := new_row_count + 1) <= initial_row_count:
        raw_worksheet.delete_rows(first_empty_row_idx, initial_row_count)

    # Update other worksheets by adding or removing rows
    for title in other_worksheet_titles:
        worksheet = sheet.worksheet(title)

        if first_empty_row_idx <= initial_row_count:
            worksheet.delete_rows(first_empty_row_idx, initial_row_count)
        elif new_row_count > initial_row_count:
            # Copy the last row
            source_range = (
                f"A{initial_row_count}:{rowcol_to_a1(initial_row_count, worksheet.col_count)}"
            )
            dest_range = (
                f"A{initial_row_count + 1}:{rowcol_to_a1(new_row_count, worksheet.col_count)}"
            )
            worksheet.copy_range(source_range, dest_range)
