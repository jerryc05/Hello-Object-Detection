import cv2 as _cv2
from abc import ABC as _ABC
from datetime import timedelta as _timedelta

# camera0 = __cv.VideoCapture(0)


class VideoFrameSlicerStrategy(_ABC):
    pass


class VideoFrameSliceByInterval(VideoFrameSlicerStrategy):
    def __init__(self, interval: _timedelta):
        assert isinstance(interval, _timedelta)
        self._interval = interval


class VideoFrameSliceByFrame(VideoFrameSlicerStrategy):
    def __init__(self, frame_num: int):
        self._frame_num = frame_num


class VideoFrameSlicer:
    def __init__(self,
                 v_cap: _cv2.VideoCapture,
                 slice_strat: VideoFrameSlicerStrategy,
                 auto_release=True,
                 flip_img=False,
                 show_img=False):
        assert isinstance(v_cap, _cv2.VideoCapture)
        assert isinstance(slice_strat, VideoFrameSlicerStrategy)
        assert isinstance(auto_release, bool)
        assert isinstance(flip_img, bool)
        assert isinstance(show_img, bool)

        self.__v_cap = v_cap
        self.__slice_strat = slice_strat
        self.__auto_release = auto_release
        self.__flip_img = flip_img
        self.__show_img = show_img
        self.__fps = v_cap.get(_cv2.CAP_PROP_FPS)
        self.__size = (int(v_cap.get(_cv2.CAP_PROP_FRAME_WIDTH)),
                       int(v_cap.get(_cv2.CAP_PROP_FRAME_HEIGHT)))
        self.__frames = int(v_cap.get(_cv2.CAP_PROP_FRAME_COUNT))

        func_info = 'VideoFrameSlicer.__init__'
        print(f'[{func_info}]: FPS \t= {self.__fps}')
        print(f'[{func_info}]: SIZE \t= {self.__size[0]} x {self.__size[1]}')
        print(
            f'[{func_info}]: FRAMES \t= {self.__frames if self.__frames > 0 else "--"}'
        )

    def __enter__(self):
        if isinstance(self.__slice_strat, VideoFrameSliceByInterval):
            return self.__slice_by_interval(
                self.__slice_strat._interval.total_seconds() * 1000)

        elif isinstance(self.__slice_strat, VideoFrameSliceByFrame):
            return self.__slice_by_interval(
              self.__slice_strat._frame_num / self.__fps * 1000)

    def __slice_by_interval(self, interval_ms: float):
        interval_ms = int(interval_ms)
        while self.__v_cap.isOpened():
            _, frame = self.__v_cap.read()
            if self.__flip_img:
                frame = _cv2.flip(frame, 1)
            if self.__show_img:
                _cv2.imshow(str(self.__v_cap), frame)

            key = _cv2.waitKey(interval_ms) & 0xFF
            # if key == ord('s'):
            #     print('s')
            from datetime import datetime
            __file_name__ = f"{datetime.now().isoformat('T').replace(':','_')}.jpg"
            print(__file_name__)
            _cv2.imwrite(__file_name__, frame)

            if key == ord('q'):
                print('q')
                break

    def __exit__(self):
        if self.__auto_release:
            self.__v_cap.release()


async def destroy_cv_windows():
    _cv2.destroyAllWindows()


if __name__ == '__main__':
    slicer = VideoFrameSlicer(_cv2.VideoCapture(0),
                              VideoFrameSliceByInterval(
                                  _timedelta(seconds=2/3)),
                              flip_img=True,
                              show_img=True)
    with slicer:
        pass
