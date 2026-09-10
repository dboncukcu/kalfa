from kalfa.registration import lego
from kalfa.std.common.figure import Figure


@lego("/plot/kalfa/architecture_text", partial=True, alias="architecture_text",
      description="The report models printed as text under plots/<name>.txt, the module repr of each")
def architecture_text(predictions, history, models, record, name=None, figures=None):
    figures = figures or Figure()
    lines = []
    for label, model in (models or {}).items():
        lines.append(f"== {label}")
        lines.append(repr(model))
        lines.append("")
    figures.target(record, f"{name or 'architecture_text'}.txt").write_text("\n".join(lines))
    return None
