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
    """将单个维度的像素坐标转换为米"""
    pixel_offset = pixel_value - principal_point
    meter_value = (pixel_offset * object_distance) / focal_length
    return meter_value


def read_3d_coords(filename):
    """读取3D坐标文件（真实值或预测值），每行格式为 (X, Y, Z)"""
    coords = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):  # 行号作为帧号
                line = line.strip()
                if not line:
                    continue
                # 正则匹配括号内的3个数值
                match = re.match(r'\(([\d.]+),\s*([\d.]+),\s*([\d.]+)\)', line)
                if not match:
                    print(f"警告: {filename} 第{line_num}行格式错误，已跳过：{line}")
                    continue
                try:
                    x = float(match.group(1))
                    y = float(match.group(2))
                    z = float(match.group(3))
                    coords.append((line_num, x, y, z))
                except ValueError as e:
                    print(f"警告: {filename} 第{line_num}行数据转换失败：{e}，已跳过")
                    continue
    except FileNotFoundError:
        print(f"错误: 文件未找到 - {filename}")
        return []
    except Exception as e:
        print(f"错误: 读取{filename}时异常：{e}")
        return []
    return coords


def convert_pixel_to_meter(data, camera_intrinsics, focal='fx', principal='cx', object_distance=1.0):
    """将单个维度的像素数据转换为米单位"""
    converted_data = []
    focal_length = getattr(camera_intrinsics, focal, camera_intrinsics.fx)
    principal_point = getattr(camera_intrinsics, principal, camera_intrinsics.cx)
    for frame, actual_pixel, predicted_pixel in data:
        actual_meter = pixel_to_meter(actual_pixel, focal_length, principal_point, object_distance)
        predicted_meter = pixel_to_meter(predicted_pixel, focal_length, principal_point,
                                         object_distance) if predicted_pixel is not None else None
        converted_data.append((frame, actual_meter, predicted_meter))
    return converted_data


def get_time_phase(frame, fps=15):
    """根据帧号划分时间阶段"""
    time_sec = frame / fps
    if time_sec <= 2:
        return "phase1"
    elif time_sec <= 6:
        return "phase2"
    else:
        return "phase3"


def calculate_euclidean_distance(x_meter, y_meter, z_meter):
    """计算3D欧式距离误差"""
    x_map = {f: (a, p) for f, a, p in x_meter}
    y_map = {f: (a, p) for f, a, p in y_meter}
    z_map = {f: (a, p) for f, a, p in z_meter}
    valid_frames = set(x_map.keys()) & set(y_map.keys()) & set(z_map.keys())
    euclidean_errors = []
    for frame in sorted(valid_frames):
        x_a, x_p = x_map[frame]
        y_a, y_p = y_map[frame]
        z_a, z_p = z_map[frame]
        if x_p is not None and y_p is not None and z_p is not None:
            distance = math.sqrt((x_a - x_p) ** 2 + (y_a - y_p) ** 2 + (z_a - z_p) ** 2)
            euclidean_errors.append((frame, distance))
    return euclidean_errors


def process_single_experiment(t_path, p_path, camera_intrinsics, object_distance=1.0):
    """处理单个实验（真实3D文件+预测3D文件）"""
    exp_name = os.path.splitext(os.path.basename(t_path))[0].replace("T", "Exp")
    print(f"\n处理实验: {exp_name}")

    # 读取真实和预测数据
    truth_3d = read_3d_coords(t_path)
    pred_3d = read_3d_coords(p_path)
    if not truth_3d:
        print(f"  警告: 真实3D文件无有效数据 - {t_path}")
        return None
    if not pred_3d:
        print(f"  警告: 预测3D文件无有效数据 - {p_path}")
        return None

    # 处理帧数不一致问题
    min_len = min(len(truth_3d), len(pred_3d))
    if len(truth_3d) != len(pred_3d):
        print(f"  警告: 真实帧数({len(truth_3d)})≠预测帧数({len(pred_3d)})，仅处理前{min_len}帧")
        truth_3d = truth_3d[:min_len]
        pred_3d = pred_3d[:min_len]

    # 构建X/Y/Z单维度数据
    x_data, y_data, z_data = [], [], []
    for (f_t, x_t, y_t, z_t), (f_p, x_p, y_p, z_p) in zip(truth_3d, pred_3d):
        if f_t != f_p:
            print(f"  警告: 帧号不匹配（真实{f_t} vs 预测{f_p}），已跳过")
            continue
        x_data.append((f_t, x_t, x_p))
        y_data.append((f_t, y_t, y_p))
        z_data.append((f_t, z_t, z_p))
    if not x_data or not y_data or not z_data:
        print(f"  警告: 无有效X/Y/Z维度数据")
        return None

    # 像素→米转换（X/Y/Z用对应内参）
    x_meter = convert_pixel_to_meter(x_data, camera_intrinsics, focal='fx', principal='cx',
                                     object_distance=object_distance)
    y_meter = convert_pixel_to_meter(y_data, camera_intrinsics, focal='fy', principal='cy',
                                     object_distance=object_distance)
    z_meter = convert_pixel_to_meter(z_data, camera_intrinsics, focal='fx', principal='cx',
                                     object_distance=object_distance)

    # 计算3D欧式距离
    euclidean_errors = calculate_euclidean_distance(x_meter, y_meter, z_meter)
    if not euclidean_errors:
        print(f"  警告: 无有效3D欧式距离数据")
        return None

    # 按阶段统计
    phase_data = {"phase1": [], "phase2": [], "phase3": []}
    for frame, error in euclidean_errors:
        phase = get_time_phase(frame)
        phase_data[phase].append((frame, error))

    phase_stats = {}
    for phase, data in phase_data.items():
        if not data:
            print(f"  警告: {phase}无有效数据")
            phase_stats[phase] = {'samples': 0, 'total_se': 0, 'rmse': 0, 'mae': 0, 'max_error': 0, 'min_error': 0,
                                  'mean_error': 0}
            continue
        n = len(data)
        errors = [err for _, err in data]
        total_se = sum(err ** 2 for err in errors)
        phase_stats[phase] = {
            'samples': n, 'total_se': total_se, 'rmse': math.sqrt(total_se / n),
            'mae': sum(abs(err) for err in errors) / n, 'max_error': max(errors),
            'min_error': min(errors), 'mean_error': sum(errors) / n
        }
        print(f"  {phase}: {n}个3D样本, RMSE: {phase_stats[phase]['rmse']:.6f}米")
    return phase_stats


# -------------------------- 关键修复：generate_experiment_paths函数 --------------------------
def generate_experiment_paths(truth_paths, pred_paths):
    """生成实验路径对（真实文件→预测文件一一对应）"""
    experiment_paths = []
    # 1. 先检查路径数量是否匹配
    if len(truth_paths) != len(pred_paths):
        print("警告: 真实文件数量≠预测文件数量，无法批量处理")
        return experiment_paths  # 返回空列表（此时确实无有效路径）

    # 2. 检查每个路径对应的文件是否存在，避免路径正确但文件缺失
    for idx, (t_path, p_path) in enumerate(zip(truth_paths, pred_paths), 1):
        # 检查真实文件
        if not os.path.exists(t_path):
            print(f"警告: 第{idx}个实验的真实文件不存在 - {t_path}")
            continue  # 跳过该实验，不影响其他实验
        # 检查预测文件
        if not os.path.exists(p_path):
            print(f"警告: 第{idx}个实验的预测文件不存在 - {p_path}")
            continue  # 跳过该实验

        # 3. 路径和文件都有效，添加到路径对列表
        experiment_paths.append((t_path, p_path))

    # 4. 返回有效路径对列表（而非空列表）
    return experiment_paths


def calculate_global_rmse(experiment_paths, camera_intrinsics, object_distance=1.0):
    """批量处理所有实验，计算全局各阶段RMSE"""
    global_stats = {"phase1": {'total_se': 0.0, 'total_samples': 0},
                    "phase2": {'total_se': 0.0, 'total_samples': 0},
                    "phase3": {'total_se': 0.0, 'total_samples': 0}}
    experiment_stats = []

    print("开始批量处理实验...")
    print("=" * 70)

    for i, (t_path, p_path) in enumerate(experiment_paths, 1):
        print(f"进度: {i}/{len(experiment_paths)}")
        phase_stats = process_single_experiment(t_path, p_path, camera_intrinsics, object_distance)
        if phase_stats:
            exp_name = os.path.splitext(os.path.basename(t_path))[0]
            experiment_stats.append({'experiment_name': exp_name, 'phase_stats': phase_stats})
            # 累积全局统计量
            for phase, stats in phase_stats.items():
                if stats['samples'] > 0:
                    global_stats[phase]['total_se'] += stats['total_se']
                    global_stats[phase]['total_samples'] += stats['samples']

    print("=" * 70)

    # 计算全局RMSE
    global_results = {}
    for phase, stats in global_stats.items():
        if stats['total_samples'] > 0:
            mse = stats['total_se'] / stats['total_samples']
            global_results[phase] = {'rmse': math.sqrt(mse), 'samples': stats['total_samples'],
                                     'total_se': stats['total_se'], 'mse': mse}
        else:
            global_results[phase] = {'rmse': 0, 'samples': 0, 'total_se': 0, 'mse': 0}
    return global_results, experiment_stats, global_stats


# 主程序入口
if __name__ == "__main__":
    # 配置真实和预测文件路径（用户提供的路径）
    T_FILE_PATHS = [
        r"D:\YOLO5\yolov5-master-new\depeth\txt\drawings\1\1-9\useful\T1-9-3D.txt",
        r"D:\YOLO5\yolov5-master-new\depeth\txt\drawings\2\2-9\useful\T2-9-1-3D.txt",
        r"D:\YOLO5\yolov5-master-new\depeth\txt\drawings\2\2-9\useful\T2-9-2-3D.txt",
        r"D:\YOLO5\yolov5-master-new\depeth\txt\drawings\3\3-9\useful\T3-9-1-3D.txt",
        r"D:\YOLO5\yolov5-master-new\depeth\txt\drawings\3\3-9\useful\T3-9-2-3D.txt",
        r"D:\YOLO5\yolov5-master-new\depeth\txt\drawings\3\3-9\useful\T3-9-3-3D.txt",
        r"D:\YOLO5\yolov5-master-new\depeth\txt\drawings\4\4-9\useful\T4-9-1-3D.txt",
        r"D:\YOLO5\yolov5-master-new\depeth\txt\drawings\4\4-9\useful\T4-9-2-3D.txt",
        r"D:\YOLO5\yolov5-master-new\depeth\txt\drawings\4\4-9\useful\T4-9-3-3D.txt",
        r"D:\YOLO5\yolov5-master-new\depeth\txt\drawings\4\4-9\useful\T4-9-4-3D.txt",
        r"D:\YOLO5\yolov5-master-new\depeth\txt\drawings\5\5-9\useful\T5-9-1-3D.txt",
        r"D:\YOLO5\yolov5-master-new\depeth\txt\drawings\5\5-9\useful\T5-9-2-3D.txt",
        r"D:\YOLO5\yolov5-master-new\depeth\txt\drawings\5\5-9\useful\T5-9-3-3D.txt",
        r"D:\YOLO5\yolov5-master-new\depeth\txt\drawings\5\5-9\useful\T5-9-4-3D.txt",
        r"D:\YOLO5\yolov5-master-new\depeth\txt\drawings\5\5-9\useful\T5-9-5-3D.txt",
    ]

    P_FILE_PATHS = [
        r"D:\YOLO5\yolov5-master-new\depeth\txt\drawings\1\1-9\useful\P1-9-3D.txt",
        r"D:\YOLO5\yolov5-master-new\depeth\txt\drawings\2\2-9\useful\P2-9-1-3D.txt",
        r"D:\YOLO5\yolov5-master-new\depeth\txt\drawings\2\2-9\useful\P2-9-2-3D.txt",
        r"D:\YOLO5\yolov5-master-new\depeth\txt\drawings\3\3-9\useful\P3-9-1-3D.txt",
        r"D:\YOLO5\yolov5-master-new\depeth\txt\drawings\3\3-9\useful\P3-9-2-3D.txt",
        r"D:\YOLO5\yolov5-master-new\depeth\txt\drawings\3\3-9\useful\P3-9-3-3D.txt",
        r"D:\YOLO5\yolov5-master-new\depeth\txt\drawings\4\4-9\useful\P4-9-1-3D.txt",
        r"D:\YOLO5\yolov5-master-new\depeth\txt\drawings\4\4-9\useful\P4-9-2-3D.txt",
        r"D:\YOLO5\yolov5-master-new\depeth\txt\drawings\4\4-9\useful\P4-9-3-3D.txt",
        r"D:\YOLO5\yolov5-master-new\depeth\txt\drawings\4\4-9\useful\P4-9-4-3D.txt",
        r"D:\YOLO5\yolov5-master-new\depeth\txt\drawings\5\5-9\useful\P5-9-1-3D.txt",
        r"D:\YOLO5\yolov5-master-new\depeth\txt\drawings\5\5-9\useful\P5-9-2-3D.txt",
        r"D:\YOLO5\yolov5-master-new\depeth\txt\drawings\5\5-9\useful\P5-9-3-3D.txt",
        r"D:\YOLO5\yolov5-master-new\depeth\txt\drawings\5\5-9\useful\P5-9-4-3D.txt",
        r"D:\YOLO5\yolov5-master-new\depeth\txt\drawings\5\5-9\useful\P5-9-5-3D.txt",
    ]

    # 生成有效实验路径对（关键：此时会返回正确的路径对，而非空列表）
    experiment_paths = generate_experiment_paths(T_FILE_PATHS, P_FILE_PATHS)
    if not experiment_paths:
        print("无有效实验路径，程序退出")
        exit()

    # 初始化相机内参和物体距离（需根据实际场景修改！）
    camera = CameraIntrinsics()
    object_distance = 0.5  # 物体到相机的实际距离（米），必须修改为真实值

    # 打印基础信息
    print("相机内参信息:")
    print(f"X轴：fx={camera.fx}像素, cx={camera.cx}像素")
    print(f"Y轴：fy={camera.fy}像素, cy={camera.cy}像素")
    print(f"物体到相机距离：{object_distance}米")
    print(f"待处理实验数量：{len(experiment_paths)}")
    print("-" * 70)

    # 阶段描述
    phase_descriptions = {
        "phase1": "0-2秒（初始加速阶段，风速从0到最高）",
        "phase2": "2-6秒（稳定过渡阶段，风速稳定）",
        "phase3": "6-10秒（减速阶段，风速从稳定到0）"
    }

    # 批量计算RMSE
    global_results, exp_stats, global_stats = calculate_global_rmse(
        experiment_paths, camera, object_distance
    )

    # 输出结果
    print(f"\n各阶段全局3D欧式距离RMSE结果:")
    print("=" * 70)
    total_samples = 0
    for phase, desc in phase_descriptions.items():
        res = global_results[phase]
        print(f"\n{desc}:")
        print(f"  总3D样本数: {res['samples']}")
        print(f"  总平方误差: {res['total_se']:.10f} 米²")
        print(f"  均方误差(MSE): {res['mse']:.10f} 米²")
        print(f"  均方根误差(RMSE): {res['rmse']:.6f} 米")
        total_samples += res['samples']

    # 加权平均RMSE
    weighted_rmse = 0.0
    if total_samples > 0:
        for phase in phase_descriptions:
            weight = global_results[phase]['samples'] / total_samples
            weighted_rmse += global_results[phase]['rmse'] * weight
    print(f"\n总3D样本数: {total_samples}")
    print(f"加权平均3D RMSE: {weighted_rmse:.6f} 米")

    # 保存结果
    save_choice = input("\n是否将结果保存到文件？(y/n): ").strip().lower()
    if save_choice == 'y':
        output_file = "3D_RMSE_Results.txt"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("3D欧式距离RMSE批量处理结果\n")
                f.write("=" * 60 + "\n")
                f.write(f"处理时间: {os.popen('date /t').read().strip()} {os.popen('time /t').read().strip()}\n")
                f.write(f"实验数量: {len(experiment_paths)}\n")
                f.write(f"物体到相机距离: {object_distance}米\n")
                f.write(f"相机内参: fx={camera.fx}, fy={camera.fy}, cx={camera.cx}, cy={camera.cy}\n\n")

                # 全局结果
                f.write("一、全局各阶段结果\n")
                f.write("-" * 40 + "\n")
                for phase, desc in phase_descriptions.items():
                    res = global_results[phase]
                    f.write(f"{desc}\n")
                    f.write(
                        f"  样本数: {res['samples']} | 总平方误差: {res['total_se']:.10f}米² | MSE: {res['mse']:.10f}米² | RMSE: {res['rmse']:.6f}米\n\n")
                f.write(f"加权平均3D RMSE: {weighted_rmse:.6f}米\n\n")

                # 单实验结果
                f.write("二、各实验详细结果\n")
                f.write("-" * 40 + "\n")
                for exp in exp_stats:
                    f.write(f"实验名称: {exp['experiment_name']}\n")
                    for phase, stats in exp['phase_stats'].items():
                        f.write(
                            f"  {phase_descriptions[phase]}: 样本数{stats['samples']} | RMSE{stats['rmse']:.6f}米 | MAE{stats['mae']:.6f}米\n")
                    f.write("\n")
            print(f"结果已保存到: {os.path.abspath(output_file)}")
        except Exception as e:
            print(f"保存文件失败: {e}")