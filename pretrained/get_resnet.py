import sys
import pprint

supported_models = {
    'resnet_v1_50': 241
}


if sys.argv[1] not in supported_models:
    print("Model {} is not in supported models, which are:".format(
        sys.argv[1]
    ))
    pprint.pprint(list(supported_models.keys()))
