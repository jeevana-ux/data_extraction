"""Table content cleaning."""

import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class TableCleaner:
    """Clean tables by removing disclaimer rows and empty content."""
    
    @property
    def name(self) -> str:
        return "Table Cleaner"
    
    def _is_empty_value(self, v) -> bool:
        """Check if a cell value is effectively empty."""
        # Empty containers
        if isinstance(v, (list, tuple, dict)):
            return not v

        # Handle pandas objects (Series/DataFrame) explicitly
        if isinstance(v, (pd.Series, pd.DataFrame)):
            na = pd.isna(v)
            # For Series: na is a Series → use .all()
            # For DataFrame: na is a DataFrame → use .all().all()
            try:
                if isinstance(v, pd.Series):
                    return bool(na.all())
                else:  # DataFrame
                    return bool(na.all().all())
            except Exception:
                # If anything weird happens, treat it as not empty
                return False

        # Scalar NA values
        try:
            if pd.isna(v):
                return True
        except Exception:
            # Some objects might not work with pd.isna; ignore and continue
            pass

        # Empty string
        if isinstance(v, str):
            return v.strip() == ""

        return False

    
    def _looks_like_disclaimer(self, text: str) -> bool:
        """Check if text looks like disclaimer content."""
        if not isinstance(text, str):
            return False
        
        text_lower = text.lower()
        disclaimer_keywords = [
            "confidential",
            "disclaimer",
            "unauthorized",
            "intended recipient",
            "caution",
        ]
        
        return any(keyword in text_lower for keyword in disclaimer_keywords)
    
    def _clean_cell(self, val):
        """Clean a single table cell."""
        if self._is_empty_value(val):
            return ""
        
        if isinstance(val, str):
            # Check if cell contains disclaimer text
            if self._looks_like_disclaimer(val):
                return ""
            
            # Remove Gmail noise
            if val.startswith("[image:") or val.startswith("[cid:"):
                return ""
            
            # Clean whitespace
            return val.strip()
        
        return val
    
    def clean(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Clean a table DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame, or None if table becomes empty
        """
        if df is None:
            return None
        if df.empty:
            return None
        
        # Create a copy to avoid modifying original
        cleaned = df.copy()
        
        # Clean all cells
        for col in cleaned.columns:
            cleaned[col] = cleaned[col].apply(self._clean_cell)
        
        # Remove rows that are entirely empty
        cleaned = cleaned[~cleaned.apply(
            lambda row: all(self._is_empty_value(v) for v in row),
            axis=1
        )]
        
        # Remove columns that are entirely empty
        cleaned = cleaned.loc[:, ~cleaned.apply(
            lambda col: all(self._is_empty_value(v) for v in col),
            axis=0
        )]
        
        # If table is now empty, return None
        if cleaned.empty:
            logger.debug("Table became empty after cleaning")
            return None
        
        # Preserve original metadata
        if hasattr(df, 'attrs'):
            cleaned.attrs = df.attrs.copy()
        
        return cleaned
