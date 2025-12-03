class DoclingProcessor:
    @staticmethod
    def clean_text(text: str) -> str:
        """Basic cleaning before chunking."""
        text = text.replace("\n\n", "\n")
        text = text.replace("  ", " ")
        return text.strip()
