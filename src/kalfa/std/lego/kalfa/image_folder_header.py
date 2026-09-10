from kalfa.registration import lego
from kalfa.std.source.base import ImageFolder


@lego("/lego/kalfa/image_folder_header",
      description="The fields, the dtypes, the image count and the classes of an image folder")
def image_folder_header(path):
    folder = ImageFolder(path)
    return {"columns": list(folder.fields), "dtypes": dict(folder.dtypes), "rows": len(folder),
            "classes": list(folder.classes)}
