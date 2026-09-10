import re

from kalfa.registration import lego
from kalfa.std.transform.base import table_only


@lego("/transform/kalfa/rename", alias="rename", partial=True, needs_table=True,
      description="Rename the columns a regular expression matches, pattern to the replacement, backreferences "
                  "allowed (cms_(.*)_Z_score to z_\\1)")
def rename(df, pattern, to):
    return table_only(df, "rename").rename(columns=lambda name: re.sub(pattern, to, name))
