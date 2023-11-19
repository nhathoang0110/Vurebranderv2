from abc import ABCMeta, abstractmethod


class DetectorBase(object, metaclass=ABCMeta):
    def __init__(self):

        """
        @The constructor
        """
        pass

    @abstractmethod
    def _preprocess(self, img_src):
        pass

    @abstractmethod
    def detect(self, img_src):
        """
        @Get output of Detection models
        @return: bbox coordinates
        """
        pass

    @abstractmethod
    def _postprocess(self, bboxes):
        pass