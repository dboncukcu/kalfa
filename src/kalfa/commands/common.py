from kalfa.core.run import report_choice
from kalfa.core.state import checkpoint_path


def loaded_which(ctx, source, which=None):
    if which is not None:
        return which
    return checkpoint_path(source, report_choice(ctx, source)).stem
