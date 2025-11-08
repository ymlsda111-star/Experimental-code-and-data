import math
import re
import os


class CameraIntrinsics:
    """相机内参类"""

    def __init__(self):
        self.fx = 1047.706  # x轴焦距 (单位: 像素)
        self.fy = 1047.706  # y轴焦距 (单位: 像素)
        self.cx = 1107.448  # 主点x坐标 (单位: 像素)
        self.cy = 618.709  # 主点y坐标 (单位: 像素)


def pixel_to_meter(pixel_value, focal_length, principal_point, object_distance=1.0):
    """
    将像素坐标转换为米
    假设物体距离相机的距离为object_distance米
    """
    # 像素坐标相对于主点的偏移
    pixel_offset = pixel_value - principal_point

    # 转换为米: (像素偏移 × 物体距离) / 焦距
    meter_value = (pixel_offset * object_distance) / focal_length

    return meter_value


def read_data_from_file(filename):
    """
    从文本文件读取轨迹数据
    处理格式: (001, 978.5, N/A) 或 (011, 889.5, 914.392)
    """
    data = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # 跳过空行
                    continue

                # 使用正则表达式匹配括号内的内容
                match = re.match(r'\((\d+),\s*([\d.]+),\s*([\w./]+)\)', line)
                if not match:
                    print(f"警告: 第{line_num}行格式不正确，已跳过: {line}")
                    continue

                try:
                    frame = int(match.group(1))  # 提取帧号
                    actual_pixel = float(match.group(2))  # 提取真实值(像素)

                    # 处理预测值（可能是"N/A"或数字）
                    predicted_str = match.group(3)
                    if predicted_str.upper() == 'N/A':
                        predicted_pixel = None
                    else:
                        predicted_pixel = float(predicted_str)

                    data.append((frame, actual_pixel, predicted_pixel))
                except ValueError as e:
                    print(f"警告: 第{line_num}行数据转换错误: {e}，已跳过该行")
                    continue

    except FileNotFoundError:
        print(f"错误: 文件未找到 - {filename}")
        return []
    except Exception as e:
        print(f"错误: 读取文件时发生异常 - {e}")
        return []

    return data


def convert_pixel_to_meter(data, camera_intrinsics, object_distance=1.0):
    """
    将像素数据转换为米单位
    object_distance: 假设物体距离相机的距离(米)
    """
    converted_data = []

    for frame, actual_pixel, predicted_pixel in data:
        # 转换为米单位
        actual_meter = pixel_to_meter(actual_pixel, camera_intrinsics.fx, camera_intrinsics.cx, object_distance)

        if predicted_pixel is not None:
            predicted_meter = pixel_to_meter(predicted_pixel, camera_intrinsics.fx, camera_intrinsics.cx,
                                             object_distance)
        else:
            predicted_meter = None

        converted_data.append((frame, actual_meter, predicted_meter))

    return converted_data


def process_single_file(file_path, camera_intrinsics, object_distance=1.0):
    """
    处理单个文件并返回误差统计[3](@ref)
    """
    print(f"\n处理文件: {os.path.basename(file_path)}")

    # 从文件读取数据（像素单位）
    pixel_data = read_data_from_file(file_path)

    if not pixel_data:
        print(f"  警告: 无法读取文件或文件为空")
        return 0, 0, 0, 0, 0

    # 转换为米单位
    meter_data = convert_pixel_to_meter(pixel_data, camera_intrinsics, object_distance)

    # 过滤掉预测值为None的数据点
    valid_data = [(frame, actual, predicted) for frame, actual, predicted in meter_data if predicted is not None]

    if len(valid_data) == 0:
        print(f"  警告: 文件没有有效的预测数据")
        return 0, 0, 0, 0, 0

    n = len(valid_data)

    # 计算平方误差
    squared_errors = []
    total_squared_error = 0
    errors = []

    for frame, actual, predicted in valid_data:
        error = actual - predicted
        squared_error = error ** 2
        squared_errors.append(squared_error)
        total_squared_error += squared_error
        errors.append(error)

    # 计算单个文件的RMSE
    mse = total_squared_error / n if n > 0 else 0
    rmse = math.sqrt(mse) if mse > 0 else 0

    # 计算其他统计量
    mae = sum(abs(e) for e in errors) / n if n > 0 else 0
    max_error = max(errors) if errors else 0
    min_error = min(errors) if errors else 0
    mean_error = sum(errors) / n if n > 0 else 0

    print(f"  有效样本数: {n}")
    print(f"  文件RMSE: {rmse:.6f} 米")

    return total_squared_error, n, mae, max_error, min_error


def calculate_global_rmse(file_paths, camera_intrinsics, object_distance=1.0):
    """
    批量处理多个文件并计算全局RMSE[1,3](@ref)
    """
    # 初始化全局统计量
    global_total_squared_error = 0.0
    global_total_samples = 0
    all_errors = []

    # 存储每个文件的统计信息
    file_stats = []

    print("开始批量处理文件...")
    print("=" * 70)

    # 处理每个文件
    for i, file_path in enumerate(file_paths, 1):
        print(f"进度: {i}/{len(file_paths)}")

        # 处理单个文件
        file_se, file_n, file_mae, file_max_err, file_min_err = process_single_file(
            file_path, camera_intrinsics, object_distance
        )

        if file_n > 0:
            global_total_squared_error += file_se
            global_total_samples += file_n

            # 记录文件统计信息
            file_stats.append({
                'file_path': os.path.basename(file_path),
                'samples': file_n,
                'rmse': math.sqrt(file_se / file_n) if file_se > 0 else 0,
                'mae': file_mae,
                'max_error': file_max_err,
                'min_error': file_min_err
            })

    print("=" * 70)

    # 计算全局RMSE
    if global_total_samples > 0:
        global_mse = global_total_squared_error / global_total_samples
        global_rmse = math.sqrt(global_mse)

        # 输出详细结果
        print(f"\n批量处理完成!")
        print(f"总共处理文件数: {len(file_paths)}")
        print(f"其中包含有效数据的文件数: {len(file_stats)}")
        print(f"全局总样本数: {global_total_samples}")
        print(f"全局总平方误差: {global_total_squared_error:.10f} 米²")
        print(f"全局均方误差 (MSE): {global_mse:.10f} 米²")
        print(f"全局均方根误差 (RMSE): {global_rmse:.6f} 米")

        # 显示每个文件的统计信息
        print(f"\n各文件详细统计:")
        print("文件名\t\t样本数\tRMSE(米)\tMAE(米)")
        print("-" * 60)
        for stat in file_stats:
            print(f"{stat['file_path']}\t{stat['samples']}\t{stat['rmse']:.6f}\t{stat['mae']:.6f}")

        return global_rmse, global_total_samples, global_total_squared_error, file_stats
    else:
        print("错误: 没有有效的预测数据可供计算")
        return None, 0, 0, []


# 主程序
if __name__ == "__main__":
    # 定义要处理的文件路径列表[1](@ref)
    FILE_PATHS = [
        r"D:\YOLO5\yolov5-master-new\depeth\txt\Backup\PF\1\1-9\useful\P1-9time-Y.txt",
        r"D:\YOLO5\yolov5-master-new\depeth\txt\Backup\PF\2\2-9\useful\P2-9time-1-Y.txt",
        r"D:\YOLO5\yolov5-master-new\depeth\txt\Backup\PF\2\2-9\useful\P2-9time-2-Y.txt",
        r"D:\YOLO5\yolov5-master-new\depeth\txt\Backup\PF\3\3-9\useful\P3-9time-4-Y.txt",
        r"D:\YOLO5\yolov5-master-new\depeth\txt\Backup\PF\3\3-9\useful\P3-9time-5-Y.txt",
        r"D:\YOLO5\yolov5-master-new\depeth\txt\Backup\PF\3\3-9\useful\P3-9time-6-Y.txt",
        r"D:\YOLO5\yolov5-master-new\depeth\txt\Backup\PF\4\4-9\useful\P4-9time-7-Y.txt",
        r"D:\YOLO5\yolov5-master-new\depeth\txt\Backup\PF\4\4-9\useful\P4-9time-8-Y.txt",
        r"D:\YOLO5\yolov5-master-new\depeth\txt\Backup\PF\4\4-9\useful\P4-9time-9-Y.txt",
        r"D:\YOLO5\yolov5-master-new\depeth\txt\Backup\PF\4\4-9\useful\P4-9time-10-Y.txt",
        r"D:\YOLO5\yolov5-master-new\depeth\txt\Backup\PF\5\5-9\useful\P5-9time-11-Y.txt",
        r"D:\YOLO5\yolov5-master-new\depeth\txt\Backup\PF\5\5-9\useful\P5-9time-12-Y.txt",
        r"D:\YOLO5\yolov5-master-new\depeth\txt\Backup\PF\5\5-9\useful\P5-9time-13-Y.txt",
        r"D:\YOLO5\yolov5-master-new\depeth\txt\Backup\PF\5\5-9\useful\P5-9time-14-Y.txt",
        r"D:\YOLO5\yolov5-master-new\depeth\txt\Backup\PF\5\5-9\useful\P5-9time-15-Y.txt",
    ]

    # 初始化相机内参
    camera = CameraIntrinsics()

    # 假设物体距离相机的距离（单位：米）- 您可以根据实际情况调整
    object_distance = 1.0  # 默认1米，请根据实际场景修改

    print("相机内参信息:")
    print(f"焦距 fx: {camera.fx} 像素, fy: {camera.fy} 像素")
    print(f"主点 cx: {camera.cx} 像素, cy: {camera.cy} 像素")
    print(f"假设物体距离: {object_distance} 米")
    print(f"待处理文件数量: {len(FILE_PATHS)}")
    print("-" * 70)

    # 批量处理所有文件并计算全局RMSE
    global_rmse, total_samples, total_se, file_stats = calculate_global_rmse(
        FILE_PATHS, camera, object_distance
    )

    if global_rmse is not None:
        print(f"\n最终结果:")
        print(f"全局RMSE: {global_rmse:.6f} 米")
        print(f"基于 {len(file_stats)} 个文件中的 {total_samples} 个有效样本")

        # 计算其他全局统计量
        if file_stats:
            all_mae = [stat['mae'] for stat in file_stats]
            all_max_errors = [stat['max_error'] for stat in file_stats]
            all_min_errors = [stat['min_error'] for stat in file_stats]

            # 加权平均MAE（按样本数加权）
            weighted_mae = sum(stat['mae'] * stat['samples'] for stat in file_stats) / total_samples
            global_max_error = max(all_max_errors) if all_max_errors else 0
            global_min_error = min(all_min_errors) if all_min_errors else 0

            print(f"加权平均绝对误差 (MAE): {weighted_mae:.6f} 米")
            print(f"全局最大正误差: {global_max_error:.6f} 米")
            print(f"全局最大负误差: {global_min_error:.6f} 米")

            # 计算R²决定系数（需要真实值的平均值）
            # 注意：这里需要所有真实值来计算，但为了效率我们跳过详细计算
            print("\n注: 如需计算全局R²决定系数，需要合并所有真实值数据")

    # 可选：保存结果到文件
    save_results = input("\n是否将结果保存到文件? (y/n): ").lower().strip()
    if save_results == 'y':
        output_file = "global_rmse_results.txt"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("全局RMSE计算结果\n")
                f.write("=" * 50 + "\n")
                f.write(f"处理文件数量: {len(FILE_PATHS)}\n")
                f.write(f"有效文件数量: {len(file_stats)}\n")
                f.write(f"总样本数: {total_samples}\n")
                f.write(f"全局RMSE: {global_rmse:.6f} 米\n")
                f.write(f"全局MSE: {total_se / total_samples:.10f} 米²\n")
                f.write("\n各文件详细统计:\n")
                for stat in file_stats:
                    f.write(
                        f"{stat['file_path']}: 样本数={stat['samples']}, RMSE={stat['rmse']:.6f}米, MAE={stat['mae']:.6f}米\n")
            print(f"结果已保存到: {output_file}")
        except Exception as e:
            print(f"保存文件时出错: {e}")