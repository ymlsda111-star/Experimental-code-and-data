import cv2
import numpy as np


class MotionDeblurrer:
    """运动模糊处理核心类"""

    def __init__(self, kernel_size=30, angle=0):
        self.kernel_size = kernel_size
        self.angle = angle

    def wiener_filter(self, image):
        # 转换为浮点型并归一化
        image = image.astype(np.float32) / 255.0
        psf = self._create_motion_psf()
        restored = np.zeros_like(image)

        for c in range(3):
            ch_img = image[:, :, c]
            ch_psf = psf[:, :, c]

            # 频域中心化处理
            psf_padded = np.zeros_like(ch_img)
            psf_padded[:ch_psf.shape[0], :ch_psf.shape[1]] = ch_psf
            psf_fft = np.fft.fftshift(np.fft.fft2(psf_padded))  # 中心化

            img_fft = np.fft.fftshift(np.fft.fft2(ch_img))  # 中心化

            # 维纳滤波计算
            H = np.conj(psf_fft) / (np.abs(psf_fft) ** 2 + 0.01)
            restored_c = np.abs(np.fft.ifft2(np.fft.ifftshift(img_fft * H)))  # 反中心化

            # 动态范围调整
            restored_c = np.clip(restored_c * 255, 0, 255).astype(np.uint8)
            restored[:, :, c] = restored_c

        restored = np.clip(restored, 0, 255).astype(np.uint8)
        return restored

    def _create_motion_psf(self):
        # 生成单通道运动模糊PSF（模拟水平运动）
        psf = np.zeros((self.kernel_size, self.kernel_size))
        center = self.kernel_size // 2
        psf[center:center + 1, :] = 1  # 水平线状模糊核
        psf = cv2.warpAffine(psf,
                             cv2.getRotationMatrix2D((center, center),
                                                     self.angle, 1),
                             (self.kernel_size, self.kernel_size))
        # 扩展为三通道并归一化
        psf = np.repeat(psf[:, :, np.newaxis], 3, axis=2)
        return psf / psf.sum()