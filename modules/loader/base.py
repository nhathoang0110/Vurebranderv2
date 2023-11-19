from abc import ABCMeta, abstractmethod


class LoaderBase(metaclass=ABCMeta):
    def __init__(self):
        """
        @The constructor
        """
        pass

    @abstractmethod
    def connect(self):
        """
        @Connect to source video
        """
        pass

    @abstractmethod
    def disconnect(self):
        """
        @Disconnect to source video
        :return:
        """

    @abstractmethod
    def extract_frame(self):
        """
        @Extract frame from video
        :return: frame id, queue_size, frame_image
        """
        pass