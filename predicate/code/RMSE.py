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


def get_time_phase(frame, fps=15):
    """
    根据帧号确定时间阶段
    假设帧率为15fps，总时长10秒，共150帧
    """
    time_sec = frame / fps

    if time_sec <= 2:
        return "phase1"  # 0-2秒：初始加速阶段
    elif time_sec <= 6:
        return "phase2"  # 2-6秒：稳定过渡阶段
    else:
        return "phase3"  # 6-10秒：减速阶段


def process_single_file(file_path, camera_intrinsics, object_distance=1.0):
    """
    处理单个文件并返回各阶段的误差统计
    """
    print(f"\n处理文件: {os.path.basename(file_path)}")

    # 从文件读取数据（像素单位）
    pixel_data = read_data_from_file(file_path)

    if not pixel_data:
        print(f"  警告: 无法读取文件或文件为空")
        return None

    # 转换为米单位
    meter_data = convert_pixel_to_meter(pixel_data, camera_intrinsics, object_distance)

    # 按阶段分类数据
    phase_data = {
        "phase1": [],  # 0-2秒：初始加速阶段
        "phase2": [],  # 2-6秒：稳定过渡阶段
        "phase3": []  # 6-10秒：减速阶段
    }

    for frame, actual, predicted in meter_data:
        if predicted is not None:  # 只处理有预测值的数据
            phase = get_time_phase(frame)
            phase_data[phase].append((frame, actual, predicted))

    # 计算各阶段的统计量
    phase_stats = {}

    for phase_name, data in phase_data.items():
        if len(data) == 0:
            print(f"  警告: {phase_name} 没有有效的预测数据")
            phase_stats[phase_name] = {
                'samples': 0,
                'total_se': 0,
                'rmse': 0,
                'mae': 0,
                'max_error': 0,
                'min_error': 0,
                'mean_error': 0
            }
            continue

        n = len(data)
        total_squared_error = 0
        errors = []

        for frame, actual, predicted in data:
            error = actual - predicted
            squared_error = error ** 2
            total_squared_error += squared_error
            errors.append(error)

        # 计算各阶段的统计量
        mse = total_squared_error / n
        rmse = math.sqrt(mse)
        mae = sum(abs(e) for e in errors) / n
        max_error = max(errors)
        min_error = min(errors)
        mean_error = sum(errors) / n

        phase_stats[phase_name] = {
            'samples': n,
            'total_se': total_squared_error,
            'rmse': rmse,
            'mae': mae,
            'max_error': max_error,
            'min_error': min_error,
            'mean_error': mean_error
        }

        print(f"  {phase_name}: {n}个样本, RMSE: {rmse:.6f}米")

    return phase_stats


def calculate_global_rmse(file_paths, camera_intrinsics, object_distance=1.0):
    """
    批量处理多个文件并计算各阶段的全局RMSE
    """
    # 初始化各阶段的全局统计量
    global_stats = {
        "phase1": {'total_se': 0.0, 'total_samples': 0},
        "phase2": {'total_se': 0.0, 'total_samples': 0},
        "phase3": {'total_se': 0.0, 'total_samples': 0}
    }

    # 存储每个文件的统计信息
    file_stats = []

    print("开始批量处理文件...")
    print("=" * 70)

    # 处理每个文件
    for i, file_path in enumerate(file_paths, 1):
        print(f"进度: {i}/{len(file_paths)}")

        # 处理单个文件
        phase_stats = process_single_file(file_path, camera_intrinsics, object_distance)

        if phase_stats:
            file_stats.append({
                'file_path': os.path.basename(file_path),
                'phase_stats': phase_stats
            })

            # 累积全局统计量
            for phase_name, stats in phase_stats.items():
                if stats['samples'] > 0:
                    global_stats[phase_name]['total_se'] += stats['total_se']
                    global_stats[phase_name]['total_samples'] += stats['samples']

    print("=" * 70)

    # 计算各阶段的全局RMSE
    results = {}

    for phase_name, stats in global_stats.items():
        if stats['total_samples'] > 0:
            global_mse = stats['total_se'] / stats['total_samples']
            global_rmse = math.sqrt(global_mse)

            results[phase_name] = {
                'rmse': global_rmse,
                'samples': stats['total_samples'],
                'total_se': stats['total_se'],
                'mse': global_mse
            }
        else:
            results[phase_name] = {
                'rmse': 0,
                'samples': 0,
                'total_se': 0,
                'mse': 0
            }

    return results, file_stats, global_stats


# 主程序
if __name__ == "__main__":
    # 定义要处理的文件路径列表
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

    # 定义阶段描述
    phase_descriptions = {
        "phase1": "0-2秒（初始加速阶段，风速从0到最高）",
        "phase2": "2-6秒（稳定过渡阶段，风速稳定）",
        "phase3": "6-10秒（减速阶段，风速从稳定到0）"
    }

    # 批量处理所有文件并计算各阶段的全局RMSE
    results, file_stats, global_stats = calculate_global_rmse(
        FILE_PATHS, camera, object_distance
    )

    # 输出各阶段的结果
    print(f"\n各阶段全局RMSE计算结果:")
    print("=" * 70)

    total_samples = 0
    for phase_name, desc in phase_descriptions.items():
        phase_result = results[phase_name]
        print(f"\n{desc}:")
        print(f"  样本数量: {phase_result['samples']}")
        print(f"  总平方误差: {phase_result['total_se']:.10f} 米²")
        print(f"  均方误差 (MSE): {phase_result['mse']:.10f} 米²")
        print(f"  均方根误差 (RMSE): {phase_result['rmse']:.6f} 米")
        total_samples += phase_result['samples']

    print(f"\n总样本数量: {total_samples}")

    # 计算加权平均RMSE（按样本数加权）
    weighted_rmse = 0
    for phase_name in phase_descriptions:
        if total_samples > 0:
            weight = results[phase_name]['samples'] / total_samples
            weighted_rmse += results[phase_name]['rmse'] * weight

    print(f"加权平均RMSE: {weighted_rmse:.6f} 米")

    # 可选：保存详细结果到文件
    save_results = input("\n是否将详细结果保存到文件? (y/n): ").lower().strip()
    if save_results == 'y':
        output_file = "phase_rmse_results.txt"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("各阶段RMSE计算结果\n")
                f.write("=" * 50 + "\n")
                f.write(f"处理文件数量: {len(FILE_PATHS)}\n")
                f.write(f"总样本数: {total_samples}\n\n")

                for phase_name, desc in phase_descriptions.items():
                    phase_result = results[phase_name]
                    f.write(f"{desc}:\n")
                    f.write(f"  样本数量: {phase_result['samples']}\n")
                    f.write(f"  RMSE: {phase_result['rmse']:.6f} 米\n")
                    f.write(f"  MSE: {phase_result['mse']:.10f} 米²\n\n")

                f.write(f"加权平均RMSE: {weighted_rmse:.6f} 米\n\n")

                f.write("各文件详细统计:\n")
                for file_stat in file_stats:
                    f.write(f"\n{file_stat['file_path']}:\n")
                    for phase_name, desc in phase_descriptions.items():
                        stats = file_stat['phase_stats'][phase_name]
                        f.write(f"  {desc}: {stats['samples']}样本, RMSE: {stats['rmse']:.6f}米\n")

            print(f"结果已保存到: {output_file}")
        except Exception as e:
            print(f"保存文件时出错: {e}")