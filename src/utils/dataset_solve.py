import pandas as pd

# 文件路径
input_file = '../../data/job.csv'  # 原始文件
output_file = '../../data/job_simplified.csv'  # 处理后的文件

def simplify_scholar():
    # 读取数据
    df = pd.read_csv(input_file)
    # 对第二行及之后的所有列（跳过第一列）进行缩写处理
    df.iloc[1:, 0] = df.iloc[1:, 0].apply(lambda x: x[0].upper() if isinstance(x, str) and x else '')
    # 保存简化后的数据到新文件
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f'处理完成，简化后的文件已保存到: {output_file}')

def convert_salary():
    # 读取数据
    df = pd.read_csv(output_file)
    # 定义函数，将范围转换为平均数
    def convert_salary_range(salary):
        if isinstance(salary, str) and '-' in salary:
            # 去掉符号并拆分范围
            salary = salary.replace('$', '').replace('K', '')
            low, high = map(int, salary.split('-'))
            # 返回范围的平均值
            return (low + high) * 500
        return None  # 如果数据格式不符合，返回空值
    # 处理第二列（假设是第1行以后的第2列，索引为 1）
    df.iloc[0:, 1] = df.iloc[0:, 1].apply(convert_salary_range)
    # 保存结果
    df.to_csv(output_file, index=False, encoding='utf-8')

def run():
    # simplify_scholar()
    # convert_salary()
    # 注意 还需要手动删除一些暂时不关心的数据列 这里没写函数处理
    # TODO: 后续如果来得及写一个脚本一起处理
    print('done!')

if __name__ == "__main__":
    run()