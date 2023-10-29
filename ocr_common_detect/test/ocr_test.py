import base64
import os
import time
import json
import json_tools
#from paas_req import send_req


def get_file_list(dir_path):
    ret_list = []
    for root, ds, fs in os.walk(dir_path):
        # only one level
        for f in fs:
            ret_list.append(os.path.join(root, f))
    return ret_list


def get_alg_std_res(dir_path):
    f_list = get_file_list(dir_path)
    res_dict = dict()
    for ele in f_list:
        with open(ele) as f:
            res_dict[os.path.basename(f.name)] = json.load(f)
    return res_dict


def func():
    img_dir = "./test_img"  # img dir
    res_json_dir = "test_res_json"  # res_dir
    alg_res_dir = "std_json_result"  # alg res dir
    alg_std_res_dict = get_alg_std_res(alg_res_dir)
    f_list = get_file_list(img_dir)
    with open("diff.txt", "w") as diff_f:
        for ele in f_list:
            with open(ele, "rb") as f:
                data = f.read()
                f.close()
                body = {
                    "image_base64": base64.b64encode(data).decode()
                }
                alg_res = {
                    "result": body["image_base64"]
                }
                base_name = os.path.splitext(os.path.basename(f.name))[0] + ".json"
                write_path = os.path.join(res_json_dir, base_name)
                with open(write_path, "wt") as w_f:
                    w_f.write(json.dumps(alg_res, ensure_ascii=False))
                    w_f.close()
                    dest_json = alg_std_res_dict[base_name]
                    src_json = alg_res
                    patch = json_tools.diff(src_json, dest_json)
                    if len(patch) > 0:
                        print(write_path + " diff len:", len(patch), " diff :", patch)
                        write_str = '{} diff len:{} diff {}:'.format(write_path, len(patch), patch)
                        diff_f.write(write_str)
                    else:
                        print(write_path)
                        diff_f.write(write_path+'\n')


if __name__ == "__main__":
    for _ in range(1):
        func()
    pass
