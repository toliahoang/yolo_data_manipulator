import os
from yolo_rename_class.get_all_directory import get_directory


class_map_dict = {

"0":"0",
"7":"1",
"8":"2",
"9":"3",
"10":"4",
"11":"5",
"5":"6",
"13":"7",
"14":"8",
"33":"9",
"34":"10",
"35":"11",
"36":"12",
"17":"13",
"18":"14",
"19":"15",
"20":"16",
"30":"17",
"31":"18",
"4":"19",
"1":"20",
"2":"21",
"15":"22",
"16":"23",
"21":"24",
"22":"25",
"24":"26",
"25":"27",
"26":"28",
"27":"29",
"28":"30",
"29":"31",
"32":"32",
"3":"33",
"6":"34",
"23":"35",
"12":"36"

}


def write_new_line(text, file):
    with open(file, "r") as f:
        content = f.readlines()
        flag = False
        if content != []:
            pass

        for i in content:
            if i[:2] == 32 or i[:2] == "32":
                print(i[:2])
                flag = True

    with open(file, "a") as f:
        if not flag:
            if not content[-1].endswith('\n'):
                f.write('\n')
            print("class already existed",flag)
            f.writelines(text)


class SearchReplace:

    def get_id_coord_separate(self,text):
        if not text:
            raise IOError
        content = [i.split(" ", 1) for i in text]
        split_text_no_space = [i.strip() for i in content]
        return split_text_no_space

    def shift_id(self, file):
        with open(file, 'r') as f:
            content = f.readlines()
            content = [i.strip() for i in content]
            content = [i.split(" ", 1) for i in content]
            print(content)
            content = [[str(int(i[0]) + 100), i[1]] for i in content if int(i[0]) < 100]
            print(content)
        content = [" ".join(i) for i in content]
        with open(file, 'w') as f:
            for line in content:
                f.write(f"{line}\n")

    def replace_class_id(self, file):
        with open(file, 'r') as f:
            content = f.readlines()
            content = [i.strip() for i in content]
            content = [i.split(" ", 1) for i in content]
            for idx, i in enumerate(content):
                if class_map_dict.get(i[0]) is not None:
                    content[idx][0] = class_map_dict[i[0]]

        content = [" ".join(i) for i in content]
        with open(file, 'w') as f:
            for line in content:
                f.write(f"{line}\n")

    def replace_coord(self, file, id, new_coord):
        if not isinstance(id,str):
            id = str(id)
        if not isinstance(new_coord, str):
            raise TypeError("input of coordinate not string type")
        with open(file, 'r') as f:
            content = f.readlines()
            content = [i.strip() for i in content]
            content = [i.split(" ", 1) for i in content]
        list_idx = []
        for idx,i in enumerate(content):
            if i[0] == id:
                list_idx.append(idx)
        for idx in list_idx:
            content[idx][1] = new_coord
        content = [" ".join(i) for i in content]
        with open(file, 'w') as f:
            for line in content:
                f.write(f"{line}\n")


new_search = SearchReplace()

list_file = ["3B9QxsuDboI_win_match_9_11-04-2022_0013.txt", "3B9QxsuDboI_double_kill_2_11-04-2022_0012.txt"]
# list_file = ["3B9QxsuDboI_win_match_9_11-04-2022_0013.txt"]
path = "E:/COD"

# list_file = get_directory(path)
list_file = [path + "/" + i for i in list_file]



total_file = len(list_file)
for idx,i in enumerate(list_file):
    print(f"{idx}/{total_file}--{i}")
    full_path = i
    try:
        # new_search.shift_id(full_path)
        # new_search.replace_class_id(full_path)
        new_search.replace_coord(full_path, id=32, new_coord="2 2 2 2")
    except Exception as e:
        print(e)
        pass
