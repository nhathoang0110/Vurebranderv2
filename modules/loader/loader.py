from ..stream.video import FileVideoStream
from imutils.video import FPS
import imutils
from .base import LoaderBase
from datetime import datetime


class Loader(LoaderBase):
    def __init__(self, cfg):
        """
        :param resize: Int
        """
        super(Loader, self).__init__()
        self.__stream = None
        self._fps = None
        self.width = None
        self.height = None
        self._isOpen = False
        self._resize = cfg.get("resize", None)
        self.config = cfg
        self.current_time = self.config.get("current_time", None)

    @property
    def delta_time(self):
        return self.__stream.delta_time

    def video(self):
        """
        Connect to video source
        :return void
        """
        print("[Loader] Streaming from video")
        self.__stream = FileVideoStream(
            self.config["video"]["path"],
            cam_fps=self.config["video"].get("fps", 30),
            fps=self.config.get("fps", 30),
            current_time=self.current_time,
        ).start()

    def camera(self):
        """
        Connect to camera source
        :return void
        """
        print("[Loader] Streaming from camera")
        params_time = ""
        if self.current_time is not None:
            str_c_time = datetime.strptime(
                self.current_time, "%Y-%m-%d %H:%M:%S"
            ).strftime("%Y-%m-%dT%H:%M:%S")
            params_time = "?pos={}".format(str_c_time)
        cam_url = "rtsp://{}:{}@{}:{}/{}{}".format(
            self.config["cam"]["username"],
            self.config["cam"]["password"],
            self.config["cam"]["ip"],
            self.config["cam"]["port"],
            self.config["cam"]["id"],
            params_time,
        )
        print(cam_url)
        self.__stream = FileVideoStream(
            cam_url,
            cam_fps=self.config["cam"]["fps"],
            fps=self.config["fps"],
            current_time=self.current_time,
        ).start()

    def connect(self):
        """
        Connect to stream source
        :return void
        """
        if self.config.get("type", "video") == "video":
            self.video()
        else:
            self.camera()
        self._fps = FPS().start()
        self.width, self.height = self.__stream.width, self.__stream.height
        if self._resize:
            ratio = self._resize / self.width
            self.width, self.height = self._resize, int(self.height * ratio)
        self._isOpen = True

    def disconnect(self):
        """
        Disconnect from stream source
        :return void
        """
        self._fps.stop()
        self.__stream.stop()
        self._isOpen = False
        print("[Loader] Elasped time: {}".format(self._fps.elapsed()))
        print("[Loader] FPS: {}".format(self._fps.fps()))

    def extract_frame(self):
        """
        Extract frame from stream
        :return (frame_id, queue_size, frame)
        """
        if self._isOpen is False:
            raise ConnectionError("Can not connect to source video")

        while self.__stream.more():
            try:
                (timestamp, frame_id, frame) = self.__stream.read()
                if frame is None:
                    continue
                if self._resize:
                    frame = imutils.resize(frame, width=self._resize)
                self._fps.update()
                return timestamp, frame_id, frame
            except Exception as ex:
                self._isOpen = False
                print("[Loader] Extract frame from streaming source error {}", ex)

        return None, None, None