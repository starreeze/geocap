import math

import cv2
import numpy as np


class LeafVeinDetector:
    """叶脉检测器类，用于从初始点开始追踪主叶脉和分支叶脉"""

    def __init__(
        self,
        line_length=50,
        angle_step=1,
        intensity_weight=0.4,
        direction_weight=0.6,
        max_iterations=100,
        branch_threshold=0.9,
        min_branch_points=3,
        min_intensity_threshold=0.0,
        n_branch_node=20,
        duplicate_distance_threshold=100,
    ):
        """
        初始化叶脉检测器

        参数:
            line_length: 采样线段长度
            angle_step: 角度采样步长（度数）
            intensity_weight: 像素强度权重
            direction_weight: 方向连续性权重
            max_iterations: 最大迭代次数
            branch_threshold: 开始分支搜索的平均像素阈值
            min_branch_points: 最小有效分支点数
            min_intensity_threshold: 线段扩展的最小平均像素阈值
            n_branch_node: 主脉相邻节点间的分割点数量（用于插入额外的分支搜索点）
            duplicate_distance_threshold: 判定重复叶脉的距离阈值（平均每节点距离）
        """
        self.line_length = line_length
        self.angle_step = angle_step
        self.intensity_weight = intensity_weight
        self.direction_weight = direction_weight
        self.max_iterations = max_iterations

        self.branch_threshold = branch_threshold
        self.min_branch_points = min_branch_points
        self.min_intensity_threshold = min_intensity_threshold
        self.n_branch_node = n_branch_node
        self.duplicate_distance_threshold = duplicate_distance_threshold

        # 运行时数据
        self.binary_mask = None
        self.image = None
        self.main_vein_path = None
        self.branches = None

    def get_line_average_intensity(self, x, y, angle_deg):
        """
        计算从点(x,y)出发，沿angle_deg方向的线段的平均像素值。

        参数:
        x, y: 起始点坐标
        angle_deg: 角度（度数，0度为向右，逆时针增加，90度为向上）

        返回:
        平均像素值（0-255）
        """
        if self.binary_mask is None:
            return 0
        h, w = self.binary_mask.shape
        angle_rad = math.radians(angle_deg)

        # 计算线段上的采样点
        num_samples = max(int(self.line_length), 1)
        total_intensity = 0
        valid_samples = 0

        for i in range(1, num_samples + 1):
            # 沿着角度方向采样
            # 注意：图像坐标系y轴向下，所以使用-sin来让90度向上
            sample_x = x + i * math.cos(angle_rad)
            sample_y = y - i * math.sin(angle_rad)
            px, py = int(round(sample_x)), int(round(sample_y))
            # 检查边界
            if 0 <= px < w and 0 <= py < h:
                total_intensity += int(self.binary_mask[py, px])
                valid_samples += 1

        if valid_samples == 0:
            return 0

        return total_intensity / valid_samples

    def find_best_direction_with_rotation(self, x, y, prev_angle, start_angle, end_angle):
        """
        使用旋转线段采样找到最佳搜索方向。

        参数:
        x, y: 当前点坐标
        prev_angle: 上一次搜索的方向角度（度数，None表示首次）
        start_angle: 起始角度（度数，0度为向右，逆时针增加，90度为向上）
        end_angle: 终止角度（度数）

        返回:
        (best_angle, best_score): 最佳角度和对应得分
        """
        best_score = -1
        best_angle = start_angle

        # 在指定角度范围内遍历
        angle = start_angle
        while angle < end_angle:
            # 像素得分
            avg_intensity = self.get_line_average_intensity(x, y, angle)
            intensity_score = avg_intensity / 255.0
            if intensity_score < self.min_intensity_threshold:
                angle += self.angle_step
                continue

            # 方向得分
            if prev_angle is not None:
                # 计算角度差（考虑360度循环）
                angle_diff = abs(angle - prev_angle)
                angle_diff = min(angle_diff, 360 - angle_diff)
                direction_score = 1.0 - (angle_diff / 180.0)
            else:
                direction_score = 0.5

            total_score = self.intensity_weight * intensity_score + self.direction_weight * direction_score
            if total_score > best_score:
                best_score = total_score
                best_angle = angle
            angle += self.angle_step

        return best_angle, best_score

    def find_farthest_point_in_direction(self, x, y, angle_deg):
        """
        沿着指定方向，找到最远有效点（白色）。

        参数:
        x, y: 起始点坐标
        angle_deg: 搜索方向（度数，0度为向右，逆时针增加，90度为向上）

        返回:
        (nx, ny): 找到的点坐标，如果没找到返回None
        """
        if self.binary_mask is None:
            return None
        h, w = self.binary_mask.shape
        angle_rad = math.radians(angle_deg)

        farthest_point = None

        # 从远到近尝试，找到第一个有效点
        for dist in range(int(self.line_length), 0, -1):
            # 注意：图像坐标系y轴向下，所以使用-sin来让90度向上
            nx = x + dist * math.cos(angle_rad)
            ny = y - dist * math.sin(angle_rad)
            px, py = int(round(nx)), int(round(ny))

            # 检查边界
            if not (0 <= px < w and 0 <= py < h):
                continue
            # 白色点
            if self.binary_mask[py, px] > 0:
                farthest_point = (px, py)
                break

        return farthest_point

    def line_search(
        self,
        start_point,
        initial_direction,
        start_angle,
        end_angle,
        post_start_angle=None,
        post_end_angle=None,
    ):
        """
        从初始点开始使用旋转线段搜索算法追踪叶脉。

        参数:
        start_point: 起始点坐标 (x, y)
        initial_direction: 初始搜索方向（度数）
        start_angle: 搜索角度范围的起始角度（用于前min_branch_points个点）
        end_angle: 搜索角度范围的终止角度（用于前min_branch_points个点）
        post_start_angle: 后续点的搜索角度范围起始角度（默认为None，使用start_angle）
        post_end_angle: 后续点的搜索角度范围终止角度（默认为None，使用end_angle）

        返回:
        path: 搜索到的路径点列表 [(x, y), ...]
        """
        path = [start_point]

        current_point = start_point
        prev_angle = initial_direction

        # 如果未指定post角度范围，则使用初始角度范围
        if post_start_angle is None:
            post_start_angle = start_angle
        if post_end_angle is None:
            post_end_angle = end_angle

        for i in range(self.max_iterations):
            x, y = current_point

            # 根据当前路径长度选择角度范围
            if len(path) < self.min_branch_points:
                # 前min_branch_points个点使用初始角度范围
                current_start_angle = start_angle
                current_end_angle = end_angle
            else:
                # 后续点使用post角度范围
                current_start_angle = post_start_angle
                current_end_angle = post_end_angle

            # 找到最佳搜索方向
            best_angle, best_score = self.find_best_direction_with_rotation(
                x, y, prev_angle, current_start_angle, current_end_angle
            )
            if best_score < self.branch_threshold and i <= 1:
                break

            # 沿最佳方向找到下一个点
            next_point = self.find_farthest_point_in_direction(x, y, best_angle)
            if next_point is None:
                break

            # 更新状态
            path.append(next_point)
            current_point = next_point
            prev_angle = best_angle

        return path

    def should_start_branch_search(self, start_point, angle_deg):
        """判断是否应该开始分支搜索"""
        avg_intensity = self.get_line_average_intensity(start_point[0], start_point[1], angle_deg)
        normalized_intensity = avg_intensity / 255.0
        return normalized_intensity > self.branch_threshold

    def search_branches(self):
        """
        从主脉的每个节点尝试向左右搜索分支。
        如果 n_branch_node > 0，会在相邻节点间插入均分点进行搜索。

        返回:
        branches: 字典，键为 'left' 和 'right'，值为分支路径列表
        """
        if self.main_vein_path is None:
            return {"left": [], "right": []}

        left_branches = []
        right_branches = []

        # 生成所有搜索点（包括原始节点和插入的分割点）
        search_points = []

        if self.n_branch_node > 0:
            # 在相邻节点间插入分割点
            for i in range(len(self.main_vein_path) - 1):
                node1 = self.main_vein_path[i]
                node2 = self.main_vein_path[i + 1]

                # 添加第一个节点
                search_points.append(node1)

                # 在两个节点之间插入 n_branch_node 个均分点
                for j in range(1, self.n_branch_node + 1):
                    t = j / (self.n_branch_node + 1)  # 均分比例
                    interpolated_x = int(node1[0] + t * (node2[0] - node1[0]))
                    interpolated_y = int(node1[1] + t * (node2[1] - node1[1]))
                    search_points.append((interpolated_x, interpolated_y))

            # 添加最后一个节点
            search_points.append(self.main_vein_path[-1])
        else:
            # 如果不插入分割点，直接使用主脉节点
            search_points = self.main_vein_path

        print("\n开始从主脉节点搜索分支...")
        print(f"主脉原始节点数: {len(self.main_vein_path)}")
        print(f"每段插入分割点数: {self.n_branch_node}")
        print(f"总搜索点数: {len(search_points)}")
        print(f"分支阈值: {self.branch_threshold * 100}%")

        # 遍历所有搜索点
        for i, node in enumerate(search_points):
            # 左分支搜索
            left_branch = self.line_search(
                node,
                initial_direction=135,
                start_angle=110,
                end_angle=135,
                post_start_angle=91,
                post_end_angle=135,
            )

            if len(left_branch) > self.min_branch_points:
                left_branches.append(left_branch)
                print(f"  搜索点 {i}: 找到左分支，长度 {len(left_branch)}")

            # 右分支搜索
            right_branch = self.line_search(node, initial_direction=45, start_angle=45, end_angle=70)

            if len(right_branch) > self.min_branch_points:
                right_branches.append(right_branch)
                print(f"  搜索点 {i}: 找到右分支，长度 {len(right_branch)}")

        branches = {"left": left_branches, "right": right_branches}

        # 后处理：移除重复叶脉
        branches = self.remove_duplicate_branches(branches)

        return branches

    def calculate_branch_distance(self, branch1, branch2):
        """
        计算两条分支之间各节点的平均最短距离（归一化）

        参数:
        branch1: 第一条分支的路径点列表
        branch2: 第二条分支的路径点列表

        返回:
        平均距离: float（归一化后的平均每节点距离）
        """
        if len(branch1) == 0 or len(branch2) == 0:
            return float("inf")

        total_distance = 0

        # 对每条分支的每个节点，找到另一条分支上最近的节点
        for pt1 in branch1:
            min_dist = float("inf")
            for pt2 in branch2:
                dist = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
                min_dist = min(min_dist, dist)
            total_distance += min_dist

        for pt2 in branch2:
            min_dist = float("inf")
            for pt1 in branch1:
                dist = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
                min_dist = min(min_dist, dist)
            total_distance += min_dist

        # 归一化：除以总节点数，得到平均每节点的距离
        total_nodes = len(branch1) + len(branch2)
        average_distance = total_distance / total_nodes

        return average_distance

    def remove_duplicate_branches(self, branches):
        """
        移除重复的叶脉分支

        参数:
        branches: 字典，键为 'left' 和 'right'，值为分支路径列表

        返回:
        filtered_branches: 移除重复后的分支字典
        """
        print("\n开始后处理：移除重复叶脉...")
        print(f"平均距离阈值: {self.duplicate_distance_threshold} 像素/节点")

        filtered_branches = {"left": [], "right": []}

        for side in ["left", "right"]:
            branch_list = branches[side]
            if len(branch_list) == 0:
                continue

            # 标记是否保留每条分支
            keep_flags = [True] * len(branch_list)
            removed_count = 0

            # 比较所有分支对
            for i in range(len(branch_list)):
                if not keep_flags[i]:
                    continue

                for j in range(i + 1, len(branch_list)):
                    if not keep_flags[j]:
                        continue

                    # 计算两条分支的平均距离（归一化）
                    avg_distance = self.calculate_branch_distance(branch_list[i], branch_list[j])

                    # 如果距离小于阈值，认为是重复叶脉
                    if avg_distance < self.duplicate_distance_threshold:
                        # 保留较长的分支
                        len_i = len(branch_list[i])
                        len_j = len(branch_list[j])
                        if len_i >= len_j:
                            keep_flags[j] = False
                            removed_count += 1
                            print(
                                f"  {side}侧：分支 {j}(长度{len_j}) 与分支 {i}(长度{len_i}) 重复（平均距离={avg_distance:.1f}像素），移除分支 {j}"
                            )
                        else:
                            keep_flags[i] = False
                            removed_count += 1
                            print(
                                f"  {side}侧：分支 {i}(长度{len_i}) 与分支 {j}(长度{len_j}) 重复（平均距离={avg_distance:.1f}像素），移除分支 {i}"
                            )
                            break  # i已被移除，不需要继续比较

            # 保留未被标记删除的分支
            filtered_branches[side] = [branch_list[i] for i in range(len(branch_list)) if keep_flags[i]]

            print(
                f"  {side}侧：原始分支数 {len(branch_list)}，移除 {removed_count} 条，保留 {len(filtered_branches[side])} 条"
            )

        return filtered_branches

    def preprocess_image(self, image_path, mask_path="mask.npy", resize_factor=2):
        """
        预处理图像，生成二值化mask

        参数:
        image_path: 输入图像路径
        mask_path: mask文件路径
        resize_factor: 图像缩放因子

        返回:
        binary_mask: 二值化后的mask
        image: 处理后的彩色图像
        """
        # 读取图像和mask
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"错误: 无法读取图像 {image_path}")
            return None, None

        mask = np.load(mask_path).astype(np.uint8)

        # resize图像
        image_h, image_w = image.shape[:2]
        image = cv2.resize(image, (image_w * resize_factor, image_h * resize_factor))

        # resize mask
        mask_resized = cv2.resize(
            mask, (image_w * resize_factor, image_h * resize_factor), interpolation=cv2.INTER_NEAREST
        )

        # 将mask向内腐蚀，让黑色背景盖住叶片轮廓
        erode_kernel_size = 16
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode_kernel_size * 2 + 1, erode_kernel_size * 2 + 1)
        )
        mask_resized = cv2.erode(mask_resized, kernel, iterations=1)

        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        cv2.imwrite("mask_gray.png", gray)

        # 增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        gray = clahe.apply(gray)
        cv2.imwrite("mask_enhanced.png", gray)

        # 局部自适应阈值二值化
        binary_mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blockSize=59, C=5
        )
        # 应用腐蚀后的mask，只保留叶片区域（排除轮廓）
        binary_mask = binary_mask * mask_resized
        cv2.imwrite("mask_binary_masked.png", binary_mask)

        return binary_mask, image

    def detect(self, image_path, start_point, initial_direction=90, output_path="vein_output.jpg"):
        """
        从初始点开始检测叶脉并可视化。

        参数:
        image_path: 输入图像路径
        start_point: 初始点坐标 (x, y)
        initial_direction: 初始搜索方向（度数，默认90度向上）
        angle_range: 主脉搜索角度范围 (start_angle, end_angle)
        output_path: 输出图像路径

        返回:
        output_image: 标注了叶脉的输出图像
        """
        # 预处理图像
        self.binary_mask, self.image = self.preprocess_image(image_path)
        if self.binary_mask is None or self.image is None:
            return None

        output_image = self.image.copy()

        print(f"开始从初始点 {start_point} 搜索主叶脉...")

        # 搜索主叶脉
        self.main_vein_path = self.line_search(start_point, initial_direction, start_angle=70, end_angle=110)
        print(f"主叶脉搜索完成: 总共 {len(self.main_vein_path)} 个点")

        # 从主脉节点搜索分支
        self.branches = self.search_branches()

        # 可视化搜索结果
        self.visualize_results(output_path)

        # 打印统计信息
        self.print_statistics()

        return output_image

    def visualize_results(self, output_path):
        """
        可视化检测结果

        参数:
        output_path: 输出图像路径
        """
        if self.binary_mask is None or self.main_vein_path is None or self.branches is None:
            print("错误: 检测结果不完整，无法可视化")
            return

        binary_vis = cv2.cvtColor(self.binary_mask, cv2.COLOR_GRAY2BGR)
        original_vis = self.image.copy()

        # 绘制主脉（蓝色线段，黄色节点）
        if len(self.main_vein_path) > 1:
            for i in range(len(self.main_vein_path) - 1):
                pt1 = self.main_vein_path[i]
                pt2 = self.main_vein_path[i + 1]
                cv2.line(binary_vis, pt1, pt2, (255, 0, 0), 2)
                cv2.line(original_vis, pt1, pt2, (255, 0, 0), 2)

            for pt in self.main_vein_path:
                cv2.circle(binary_vis, pt, 3, (0, 255, 255), -1)
                cv2.circle(original_vis, pt, 3, (0, 255, 255), -1)

        # 绘制左分支（蓝色线段，黄色节点）
        for branch in self.branches["left"]:
            if len(branch) > 1:
                for i in range(len(branch) - 1):
                    pt1 = branch[i]
                    pt2 = branch[i + 1]
                    cv2.line(binary_vis, pt1, pt2, (255, 0, 0), 2)
                    cv2.line(original_vis, pt1, pt2, (255, 0, 0), 2)
                for pt in branch:
                    cv2.circle(binary_vis, pt, 3, (0, 255, 255), -1)
                    cv2.circle(original_vis, pt, 3, (0, 255, 255), -1)

        # 绘制右分支（蓝色线段，黄色节点）
        for branch in self.branches["right"]:
            if len(branch) > 1:
                for i in range(len(branch) - 1):
                    pt1 = branch[i]
                    pt2 = branch[i + 1]
                    cv2.line(binary_vis, pt1, pt2, (255, 0, 0), 2)
                    cv2.line(original_vis, pt1, pt2, (255, 0, 0), 2)
                for pt in branch:
                    cv2.circle(binary_vis, pt, 3, (0, 255, 255), -1)
                    cv2.circle(original_vis, pt, 3, (0, 255, 255), -1)

        binary_output_path = output_path.replace(".jpg", "_binary.jpg")
        cv2.imwrite(binary_output_path, binary_vis)
        print(f"二值图像结果已保存到 {binary_output_path}")

        # 保存原图上的叶脉标注结果
        cv2.imwrite(output_path, original_vis)
        print(f"原图叶脉结果已保存到 {output_path}")

    def print_statistics(self):
        """打印检测统计信息"""
        if self.main_vein_path is None or self.branches is None:
            print("错误: 检测结果不完整，无法打印统计信息")
            return

        print("\n" + "=" * 60)
        print("叶脉检测统计:")
        print(f"  主脉节点数: {len(self.main_vein_path)}")
        print(f"  左分支数量: {len(self.branches['left'])}")
        print(f"  右分支数量: {len(self.branches['right'])}")
        print(f"  总分支数量: {len(self.branches['left']) + len(self.branches['right'])}")
        total_branch_points = sum(len(b) for b in self.branches["left"]) + sum(
            len(b) for b in self.branches["right"]
        )
        print(f"  分支节点总数: {total_branch_points}")
        print(f"  总节点数: {len(self.main_vein_path) + total_branch_points}")
        print("=" * 60)


def detect_leaf_veins_from_seed(image_path, start_point=(265 * 2, 1030 * 2), output_path="vein_output.jpg"):
    """
    从初始点开始检测叶脉并可视化（向后兼容的包装函数）。

    参数:
    image_path: 输入图像路径
    start_point: 初始点坐标 (x, y)
    output_path: 输出图像路径

    返回:
    output_image: 标注了叶脉的输出图像
    """
    detector = LeafVeinDetector()

    return detector.detect(image_path, start_point, output_path=output_path)


if __name__ == "__main__":
    # image_path = "Correa-Narvaez_2023_8_C.png"
    image_path = "image.png"
    start_point = (505, 1425)
    output_path = "vein_search_result.jpg"

    result = detect_leaf_veins_from_seed(image_path, start_point, output_path)
