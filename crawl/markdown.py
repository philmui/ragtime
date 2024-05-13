from markdownify import markdownify

##############################################################################
# markdown functions
###############################################################################

from markdownify import MarkdownConverter

class CustomMarkdownConverter(MarkdownConverter):
    def convert_a(self, el, text, convert_as_inline):
        # Remove href and URLs from links in the actual output
        return text
    
    def convert_img(self, el, text, convert_as_inline):
        # Remove src and href attributes from images
        return ""

class MarkdownConverterFactory:

    _converter: CustomMarkdownConverter = None

    @classmethod
    def get_converter(cls) -> CustomMarkdownConverter:
        if cls._converter == None:
            cls._converter = CustomMarkdownConverter()

        return cls._converter