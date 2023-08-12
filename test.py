# from pymilvus import connections, FieldSchema, DataType

# # Kết nối tới Milvus
# connections.connect(host='localhost', port='19530')

# # Chọn collection cần hiển thị dữ liệu
# collection_name = 'Image'

# # Lấy dữ liệu từ collection
# query = f"SELECT * FROM Image"
# results = connections.default_connection().search(collection_name, query)

# # Hiển thị dữ liệu
# for result in results:
#     print(result)

from datetime import datetime
import numpy as np

# Tạo đối tượng datetime
my_datetime = datetime.now()

# Chuyển đổi thành timestamp
timestamp = my_datetime.timestamp()

# Chuyển đổi thành int64
int64_value = np.int64(timestamp)

print(f"Giá trị datetime: {my_datetime}")
print(f"Giá trị timestamp: {timestamp}")
print(f"Giá trị int64: {int64_value}")