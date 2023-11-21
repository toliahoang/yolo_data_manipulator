import os
import re
path = "E:/COD/COD_total"



def get_directory(path):
    list_directory = []
    for dir, subdir, files in os.walk(path):
        for file in files:
            if file.endswith(".TXT"):
                x = os.path.join(dir,file)
                new_x = re.sub('\\\\', "/", x)
                list_directory.append(new_x)
    return list_directory

get_list_directory = get_directory(path)
print(get_list_directory)
# with open("E:/COD/train_1.txt", 'w') as f:
#     for line in get_list_directory:
#         f.write(f"{line}\n")




