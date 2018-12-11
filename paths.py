import os

root_dir = os.path.dirname(__file__)
data_dir = os.path.join(root_dir, 'data')
raw_dir = os.path.join(root_dir, 'raw')
save_dir = os.path.join(root_dir, 'save')

sxhy_path = os.path.join(data_dir, 'sxhy_dict.txt')
char_dict_path = os.path.join(data_dir, 'char_dict.txt')
poems_path = os.path.join(data_dir, 'poem.txt')
char2vec_path = os.path.join(data_dir, 'char2vec.npy')
wordrank_path = os.path.join(data_dir, 'wordrank.txt')
plan_data_path = os.path.join(data_dir, 'plan_data.txt')
gen_data_path = os.path.join(data_dir, 'gen_data.txt')

# TODO: configure dependencies in another file.
_dependency_dict = {
    poems_path: [char_dict_path],
    char2vec_path: [char_dict_path, poems_path],
    wordrank_path: [sxhy_path, poems_path],
    gen_data_path: [char_dict_path, poems_path, sxhy_path, char2vec_path],
    plan_data_path: [char_dict_path, poems_path, sxhy_path, char2vec_path],
}


def check_uptodate(path):
    if not os.path.exists(path):
        return False
    # if not path in _dependency_dict:
    #     return False
    if path in _dependency_dict:
        dependencyes = _dependency_dict[path]
        timestamp = os.path.getmtime(path)
        for dependency_iter in dependencyes:
            if os.path.getmtime(dependency_iter) > timestamp:
                return False
    return True
