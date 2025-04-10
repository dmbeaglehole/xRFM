
def get_param_tree(tree, is_root=False):
    if tree['type'] == 'leaf':
        leaf_model = tree['model']
        param_tree = {
            'type': 'leaf',
            'bandwidth': leaf_model.kernel_obj.bandwidth,
            'weights': leaf_model.weights,
            'M': leaf_model.M,
            'sqrtM': leaf_model.sqrtM,
            'train_indices': tree['train_indices'],
            'is_root': is_root
        }
        return param_tree
    else:
        return {
            'type': 'node',
            'split_direction': tree['split_direction'],
            'split_point': tree['split_point'],
            'left': get_param_tree(tree['left'], is_root=False),
            'right': get_param_tree(tree['right'], is_root=False),
            'is_root': is_root
        }