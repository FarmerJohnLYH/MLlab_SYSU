import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from PIL import Image

def compress_image(image_path, k):
    """使用SVD压缩图像并返回压缩后的图像数据"""
    # 加载图像并转换为RGB图像
    img = Image.open(image_path)
    img_data = np.array(img)

    # 分别对RGB三个通道执行SVD分解
    compressed_img_data = np.zeros(img_data.shape)
    for i in range(3):
        U, S, Vt = svd(img_data[:, :, i])
        
        # 截断奇异值，只保留前k个奇异值
        S_k = np.zeros((k, k))
        np.fill_diagonal(S_k, S[:k])
        U_k = U[:, :k]
        Vt_k = Vt[:k, :]
        
        # 重构压缩后的图像
        compressed_img_data[:, :, i] = U_k.dot(S_k).dot(Vt_k)
    
    return compressed_img_data
name = "50"
def plot_images(original_img, compressed_img):
    """绘制原始图像和压缩后的图像"""
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(compressed_img.astype(int))
    plt.title('Compressed Image')
    plt.savefig(name+"result.png",dpi=600)
    plt.show()

# 主程序
if __name__ == "__main__":
    image_path = 'ladybug.png'  # 替换为你的图像路径
    original_img = Image.open(image_path)
    original_img_data = np.array(original_img)

    k = 50  # 设置k值，调整压缩级别
    # k = 10
    compressed_img_data = compress_image(image_path, k)

    # 由于压缩后的图像数据可能超出了0-255的范围，所以我们需要将其裁剪到这个范围
    compressed_img_data = np.clip(compressed_img_data, 0, 255)

    # 将 compressed_img_data 保存为图片 
    int_compressed_img_data = compressed_img_data.astype(np.uint8)
    int_compressed_img = Image.fromarray(int_compressed_img_data)
    int_compressed_img.save(name+'compressed_ladybug.png')

    plot_images(original_img_data, compressed_img_data)