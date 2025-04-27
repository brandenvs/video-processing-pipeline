from .image_to_text import ImageToTextService


SERVICES = {
    'ImageToText': ImageToTextService().infer,

}