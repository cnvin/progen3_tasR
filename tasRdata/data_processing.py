import pandas as pd
import os

def extract_protein_sequences():
    """
    从TasR_Origin.xlsx文件的第二个sheet"ProteinInfo"中提取F列(aa_seq)的所有蛋白质序列
    并将其保存为txt文件，每行一个序列
    """
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Excel文件路径
    excel_file = os.path.join(current_dir, 'TasR_Origin.xlsx')
    
    # 输出txt文件路径
    output_file = os.path.join(current_dir, 'protein_sequences.txt')
    
    try:
        # 读取Excel文件的第二个sheet "ProteinInfo"
        print(f"正在读取Excel文件: {excel_file}")
        df = pd.read_excel(excel_file, sheet_name='ProteinInfo')
        
        # 检查是否存在F列(aa_seq)
        if 'aa_seq' not in df.columns:
            # 如果列名不是aa_seq，尝试使用F列的索引(第5列，从0开始计数)
            if df.shape[1] > 5:
                sequences = df.iloc[:, 5]  # F列是第6列，索引为5
                print("使用F列(索引5)提取序列")
            else:
                print("错误: 找不到F列或aa_seq列")
                return
        else:
            sequences = df['aa_seq']
            print("使用aa_seq列提取序列")
        
        # 过滤掉空值和NaN值
        valid_sequences = sequences.dropna()
        valid_sequences = valid_sequences[valid_sequences != '']
        
        print(f"找到 {len(valid_sequences)} 个有效的蛋白质序列")
        
        # 将序列写入txt文件，每行一个序列
        with open(output_file, 'w', encoding='utf-8') as f:
            for seq in valid_sequences:
                f.write(str(seq) + '\n')
        
        print(f"蛋白质序列已成功保存到: {output_file}")
        print(f"总共保存了 {len(valid_sequences)} 个序列")
        
    except FileNotFoundError:
        print(f"错误: 找不到Excel文件 {excel_file}")
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")

if __name__ == "__main__":
    extract_protein_sequences()