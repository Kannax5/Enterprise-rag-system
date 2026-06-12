"""
csv_parser.py — pandas-based CSV / tabular data parser.
"""

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class CSVParser:
    """
    Converts CSV rows into text snippets suitable for embedding.

    Each row is serialised as: "col1: val1 | col2: val2 | …"
    """

    def __init__(
        self,
        text_columns: Optional[List[str]] = None,
        separator: str = " | ",
        encoding: str = "utf-8",
    ):
        """
        Args:
            text_columns: Subset of columns to include. If None, all columns are used.
            separator:    String used to join column values within a row.
            encoding:     File encoding (default UTF-8).
        """
        self.text_columns = text_columns
        self.separator = separator
        self.encoding = encoding

    def parse(self, file_path: str | Path) -> List[dict]:
        """
        Parse a CSV and return one dict per row:
            {
                "row_index": int,
                "text": str,
                "source": str,
            }
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"CSV not found: {file_path}")

        try:
            df = pd.read_csv(file_path, encoding=self.encoding)
        except Exception as exc:
            logger.error("Failed to read CSV %s: %s", file_path, exc)
            raise

        cols = self.text_columns if self.text_columns else list(df.columns)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in {file_path.name}: {missing}")

        records: List[dict] = []
        for idx, row in df[cols].iterrows():
            parts = [f"{col}: {row[col]}" for col in cols if pd.notna(row[col])]
            text = self.separator.join(parts).strip()
            if not text:
                continue
            records.append(
                {
                    "row_index": int(idx),
                    "text": text,
                    "source": str(file_path.resolve()),
                }
            )

        logger.info("Parsed %d row(s) from %s", len(records), file_path.name)
        return records
