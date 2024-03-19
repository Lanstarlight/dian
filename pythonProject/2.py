import gzip

# 输入文件名和输出文件名
input_filename = 'train-labels-idx1-ubyte'
output_filename = 'train-labels-idx1-ubyte.gz'

# 打开输入文件以进行读取，并打开输出文件以进行写入（gzip格式）
with open(input_filename, 'rb') as f_in, gzip.open(output_filename, 'wb') as f_out:
    # 读取输入文件的内容，并写入输出文件（压缩）
    f_out.writelines(f_in)