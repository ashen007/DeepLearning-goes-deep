def learner(inputs, groups):
    """ learner layers
    :param inputs: the input tensors (feature maps)
    :param groups: the block parameters for each group
    :return:
    """
    outputs = inputs

    # iterates through the dictionary values for each group attribute
    for group_paras in groups:
        outputs = group(outputs, **group_paras)

    return outputs


def group(inputs, **blocks):
    """ group layers
    :param inputs: the input tensors (feature maps)
    :param blocks: the block parameter for the block
    :return:
    """
    outputs = inputs

    # iterate through the dictionary values for each block attribute
    for block_params in blocks:
        outputs = block(**block_params)

    return outputs


def block(inputs, **params):
    """ block layers
    :param inputs: the input tensors (feature maps)
    :param params: the block parameters for the block
    :return:
    """
    outputs = []

    return outputs
