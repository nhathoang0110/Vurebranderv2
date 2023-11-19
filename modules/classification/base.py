from abc import ABCMeta, abstractmethod


class ClassifyBase(object, metaclass=ABCMeta):
    def __init__(self):

        """
        @The constructor
        """
        pass

    @abstractmethod
    def _preprocess(self, img_src):
        pass

    @abstractmethod
    def classify(self, img_src):
        """
        @Get output of Detection models
        @return: bbox coordinates
        """
        pass