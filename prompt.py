"""
Prompt templates for geometry point position detection task
"""

class PromptTemplate:
    """几何图形点坐标识别的提示词模板"""
    
    @staticmethod
    def get_point_detection_prompt(point_names):
        """
        生成点坐标检测的提示词
        
        Args:
            point_names: 需要检测的点名列表，如 ['A', 'B', 'C']
        
        Returns:
            str: 格式化的提示词
        """
        points_str = ', '.join(point_names)
        
        prompt = f"""This is a geometry diagram. Please identify the exact pixel coordinates of the following points: {points_str}.

For each point, provide the coordinates in the format:
Point_Name: [x_coordinate, y_coordinate]

The coordinates should be precise pixel positions where:
- x_coordinate is the horizontal position from the left edge
- y_coordinate is the vertical position from the top edge

Please analyze the diagram carefully and locate each labeled point."""
        
        return prompt
    
    @staticmethod
    def get_simple_prompt(point_names):
        """简化版提示词"""
        points_str = ', '.join(point_names)
        return f"Identify the pixel coordinates of points {points_str} in this geometry diagram."
    
    @staticmethod
    def format_output_prompt():
        """输出格式提示"""
        return """Output format: {"Point_Name": [x, y], ...}"""
    
    @staticmethod
    def get_instruction_prompt(point_names):
        """指令式提示词"""
        points_str = ', '.join(point_names)
        return f"""<image>
Task: Detect the coordinates of geometric points.
Points to detect: {points_str}
Output the coordinates for each point in JSON format."""

    @staticmethod
    def parse_model_output(output_text, expected_points):
        """
        解析模型输出，提取点坐标
        
        Args:
            output_text: 模型的文本输出
            expected_points: 期望的点名列表
            
        Returns:
            dict: {point_name: [x, y]}
        """
        import re
        import json
        
        coordinates = {}
        
        # 尝试JSON解析
        try:
            # 查找JSON格式的内容
            json_match = re.search(r'\{[^}]+\}', output_text)
            if json_match:
                parsed = json.loads(json_match.group())
                for point in expected_points:
                    if point in parsed and isinstance(parsed[point], list) and len(parsed[point]) == 2:
                        coordinates[point] = parsed[point]
                return coordinates
        except:
            pass
        
        # 尝试正则表达式匹配
        for point in expected_points:
            # 匹配多种格式: "A: [100, 200]", "A: (100, 200)", "A: 100, 200"
            patterns = [
                rf"{point}\s*:\s*\[(\d+\.?\d*),\s*(\d+\.?\d*)\]",
                rf"{point}\s*:\s*\((\d+\.?\d*),\s*(\d+\.?\d*)\)",
                rf"{point}\s*:\s*(\d+\.?\d*),\s*(\d+\.?\d*)",
                rf"{point}.*?(\d+\.?\d*).*?(\d+\.?\d*)"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, output_text)
                if match:
                    try:
                        x = float(match.group(1))
                        y = float(match.group(2))
                        coordinates[point] = [x, y]
                        break
                    except:
                        continue
        
        return coordinates


# 提示词变体，用于数据增强
PROMPT_VARIANTS = [
    lambda points: f"Locate points {', '.join(points)} in the geometry diagram and return their pixel coordinates.",
    lambda points: f"Find the exact positions of points {', '.join(points)}. Return coordinates as [x, y] for each point.",
    lambda points: f"Analyze this geometric figure and determine the coordinates of points: {', '.join(points)}",
    lambda points: f"In this diagram, identify where points {', '.join(points)} are located. Provide pixel coordinates.",
]


def get_random_prompt(point_names):
    """随机选择一个提示词变体"""
    import random
    prompt_func = random.choice(PROMPT_VARIANTS)
    return prompt_func(point_names)