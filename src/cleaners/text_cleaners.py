"""Text content cleaning filters."""

import re
from typing import List
import logging

logger = logging.getLogger(__name__)


# Disclaimer detection patterns
DISCLAIMER_KEY_PHRASES = [
    "this email and any files transmitted with it are confidential",
    "if you are not the intended recipient",
    "please notify the sender immediately",
    "delete this email",
    "unauthorized use",
    "confidentiality notice",
    "privileged communication",
    "disclaimer",
    "caution: external email",
    "think before you click",
]

# Email headers to remove
EMAIL_HEADER_PREFIXES = (
    "From:",
    "To:",
    "Cc:",
    "Bcc:",
    "Sent:",
    "Date:",
)

# Gmail-specific noise patterns
GMAIL_NOISE_PATTERNS = [
    r"^\s*\[image:.*?\]\s*$",
    r"^\s*\[cid:.*?\]\s*$",
    r"^\s*<image\d+\..*?>\s*$",
]

SEPARATOR_LINE_RE = re.compile(r"^\s*-{5,}\s*$")


class DisclaimerFilter:
    """Filter to remove email disclaimers and caution notices."""
    
    @property
    def name(self) -> str:
        return "Disclaimer Filter"
    
    def looks_like_disclaimer(self, text: str) -> bool:
        """
        Check if text block looks like a disclaimer.
        
        Args:
            text: Text to check
            
        Returns:
            True if text appears to be a disclaimer
        """
        text_lower = text.lower()
        
        # Check for disclaimer key phrases
        phrase_count = sum(
            1 for phrase in DISCLAIMER_KEY_PHRASES
            if phrase in text_lower
        )
        
        # If multiple phrases found, likely a disclaimer
        if phrase_count >= 2:
            return True
        
        # Check for very long single-paragraph blocks (common in disclaimers)
        if len(text) > 500 and "\n\n" not in text:
            if any(phrase in text_lower for phrase in DISCLAIMER_KEY_PHRASES):
                return True
        
        return False
    
    def clean(self, text: str) -> str:
        """
        Remove disclaimer blocks from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with disclaimers removed
        """
        lines = text.split("\n")
        cleaned_lines = []
        in_disclaimer = False
        
        for line in lines:
            # Check if this line starts a disclaimer
            if self.looks_like_disclaimer(line):
                in_disclaimer = True
                continue
            
            # Check if we're still in a disclaimer block
            if in_disclaimer:
                # Exit disclaimer if we hit a separator or empty line
                if SEPARATOR_LINE_RE.match(line) or not line.strip():
                    in_disclaimer = False
                continue
            
            cleaned_lines.append(line)
        
        return "\n".join(cleaned_lines)


class EmailHeaderFilter:
    """Filter to remove email headers while preserving Subject lines."""
    
    @property
    def name(self) -> str:
        return "Email Header Filter"
    
    def clean(self, text: str) -> str:
        """
        Remove email headers except Subject.
        
        Args:
            text: Input text
            
        Returns:
            Text with headers removed
        """
        lines = text.split("\n")
        cleaned_lines = []
        
        for line in lines:
            # Skip email headers (but keep Subject)
            if line.startswith(EMAIL_HEADER_PREFIXES):
                continue
            
            cleaned_lines.append(line)
        
        return "\n".join(cleaned_lines)


class GmailNoiseFilter:
    """Filter to remove Gmail-specific noise (image placeholders, etc.)."""
    
    def __init__(self):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in GMAIL_NOISE_PATTERNS]
    
    @property
    def name(self) -> str:
        return "Gmail Noise Filter"
    
    def clean(self, text: str) -> str:
        """
        Remove Gmail noise patterns.
        
        Args:
            text: Input text
            
        Returns:
            Text with Gmail noise removed
        """
        lines = text.split("\n")
        cleaned_lines = []
        
        for line in lines:
            # Check if line matches any noise pattern
            if any(pattern.match(line) for pattern in self.patterns):
                continue
            
            cleaned_lines.append(line)
        
        return "\n".join(cleaned_lines)


class ContentCleaner:
    """Main content cleaner that applies all filters."""
    
    def __init__(self):
        """Initialize with all available filters."""
        self.filters = [
            EmailHeaderFilter(),
            DisclaimerFilter(),
            GmailNoiseFilter(),
        ]
    
    def clean_text(self, text: str) -> str:
        """
        Apply all text cleaning filters.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        cleaned = text
        
        for filter_obj in self.filters:
            cleaned = filter_obj.clean(cleaned)
            logger.debug(f"Applied {filter_obj.name}")
        
        # Final cleanup: remove excessive blank lines
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        return cleaned.strip()
