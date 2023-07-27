import os
import onnxruntime
import cv2
import contextlib
import datetime
import time

import numpy as np


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def add_diff(self, diff, average=True):
        self.total_time += diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

    @contextlib.contextmanager
    def tic_and_toc(self):
        try:
            yield self.tic()
        finally:
            self.toc()

    def tic(self):
        # Using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

    @classmethod
    def new(cls, *args):
        """Return a dict that contains specified timers.

        Parameters
        ----------
        args : str...
            The key(s) to create timers.

        Returns
        -------
        Dict[Timer]
            The timer dict.

        """
        return dict([(k, Timer()) for k in args])
   

def get_progress_info(timer, curr_step, max_steps):
    """Return a info of current progress.

    Parameters
    ----------
    timer : Timer
        The timer to get progress.
    curr_step : int
        The current step.
    max_steps : int
        The total number of steps.

    Returns
    -------
    str
        The progress info.

    """
    average_time = timer.average_time
    eta_seconds = average_time * (max_steps - curr_step)
    eta = str(datetime.timedelta(seconds=int(eta_seconds)))
    progress = (curr_step + 1.) / max_steps
    return '< PROGRESS: {:.2%} | SPEED: {:.3f}s / iter | ETA: {} >' \
        .format(progress, timer.average_time, eta)


class OrtEngine:
    '''模型'''

    def __init__(self, onnxfile):
        self._onnx = onnxfile
        self._sess = onnxruntime.InferenceSession(onnxfile)
        self._timer = Timer.new('PreProcess', 'Forward', 'PostProcess')

    def computation_metrics(self):
        device = onnxruntime.get_device()
        print("======================")
        print(f"Evaluating Model: {os.path.basename(self._onnx)}")
        print(
            f"Input & Output:   {self._sess.get_inputs()[0].shape} {self._sess.get_outputs()[0].shape}")
        print(f"Total Calls:      {self._timer['Forward'].calls}")
        print(f"[{device}] BS: 1 Elapsed Time PreProcess:  " +
              f"{self._timer['PreProcess'].average_time * 1000:.6f} ms")
        print(f"[{device}] BS: 1 Elapsed Time Forward:     " +
              f"{self._timer['Forward'].average_time * 1000:.6f} ms")
        print(f"[{device}] BS: 1 Elapsed Time PostProcess: " +
              f"{self._timer['PostProcess'].average_time * 1000:.6f} ms")

    @staticmethod
    def resize_and_pad(img, dsize=(320, 320), channel=1, border_color=255):
        '''
        :param img: (H, W, C)
        :param dsize: (W, H) 目标大小
        :return:
        '''
        ih, iw = img.shape[:2]
        dw, dh = dsize

        # 计算最优目标 (nw, nh)
        max_wh_ratio = max(float(iw) / ih, float(dw) / dh) # 获取最大宽高比
        nh = dh
        nw = max_wh_ratio * dh
        nw = int(int(nw / 32.0 + 0.5) * 32) # 32倍数四舍五入

        if float(iw) / ih > float(nw) / nh:
            # 图宽了
            ratio = 1.0 * nw / iw
        else:
            # 图高了
            ratio = 1.0 * nh / ih
        
        # 保持宽高比缩放
        resized = cv2.resize(img, None, fx=ratio, fy=ratio)
        new_image = np.zeros((nh, nw), dtype=np.uint8) + border_color
        new_image[:resized.shape[0], :resized.shape[1]] = resized
        return new_image


class QRDefective(OrtEngine):
    def __init__(self, onnxfile):
        super(QRDefective, self).__init__(onnxfile)
        self._x = self._sess.get_inputs()[0].name
        self._y = self._sess.get_outputs()[0].name

    def _preprocess(self, image):
        image = self.resize_and_pad(image)
        image = np.expand_dims(image, 2)
        tensor = np.transpose(image, (2, 0, 1))
        tensor = np.float32(tensor)# / 127.5 - 1.0
        tensor = tensor[np.newaxis, ...]
        return tensor
    
    def _postprocess(self, preds):
        return np.where(preds > 0.5, 1, 0)

    def __call__(self, image):
        with self._timer['PreProcess'].tic_and_toc():
               tensor = self._preprocess(image)
   
        with self._timer['Forward'].tic_and_toc():
            preds = self._sess.run([self._y], input_feed={
                self._x: tensor
            })
    
        with self._timer['PostProcess'].tic_and_toc():
            result = self._postprocess(preds[0][0])
   
        return result


if __name__ == '__main__':
    model = QRDefective("../data/qrcode-unet-mbv3-100.3.onnx")
    gray = cv2.imread("QRCodeDatasets/defective/000002.png", 0)
    result = model(gray)
    print(model.computation_metrics())
    print(result.shape, result.min(), result.max())
    cv2.imwrite("result.png", np.uint8(result[0] * 255))
