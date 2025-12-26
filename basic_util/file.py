import os 


def get_checkpoint_dir(
    root: str,
    index: int,
) -> str:
    def get_subdirs(
        dir_path: str,
        prefix: str = '',
    ) -> list[str]:
        subdir_path_list = []

        for subdir_path in os.listdir(dir_path):
            if prefix and not subdir_path.startswith(prefix):
                continue

            subdir_path = os.path.join(dir_path, subdir_path)

            if os.path.isdir(subdir_path):
                subdir_path_list.append(subdir_path) 

        return subdir_path_list 
    
    version_dir_path_list = get_subdirs(root, prefix='v')
    assert len(version_dir_path_list) == 1 
    version_dir_path = version_dir_path_list[0]
    assert os.path.basename(version_dir_path)[1].isdigit() 

    checkpoint_dir_path_list = get_subdirs(version_dir_path, prefix='checkpoint-')
    checkpoint_dir_path_list.sort(key=lambda path: int(os.path.basename(path).split('-')[-1]))

    return checkpoint_dir_path_list[index]



def get_subdir_paths(
    dir_path: str,
    prefix: str = '',
) -> list[str]:
    subdir_path_list = []

    for subdir_path in os.listdir(dir_path):
        if prefix and not subdir_path.startswith(prefix):
            continue

        subdir_path = os.path.join(dir_path, subdir_path)

        if os.path.isdir(subdir_path):
            subdir_path_list.append(subdir_path) 

    return subdir_path_list 


def get_relpath_to_dolphinfs_home(
    path: str,
) -> str:
    return os.path.relpath(
        os.path.realpath(path), 
        start = os.path.realpath('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/canyin/genghao07'),
    )
