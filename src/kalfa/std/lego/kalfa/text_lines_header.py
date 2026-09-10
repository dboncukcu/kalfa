from kalfa.registration import lego
from kalfa.std.source.base import TextLines


@lego("/lego/kalfa/text_lines_header", description="The text field and the line count of a text file")
def text_lines_header(path):
    return {"columns": ["text"], "dtypes": {"text": "string"}, "rows": len(TextLines(path))}
