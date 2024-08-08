from argparse import ArgumentParser
from . import __version__

# $ myapp --help

def cli():
    # define
    parser = ArgumentParser(
        'myaiapp',
        description="Image classification with MobileNet"
        )
    
    # parse
    parser.add_argument(
        "-v", "--version", 
        action="stor_true", help="show version"
    )
    args = parser.parse_args()
    print(args)

    parser.add_argument(
        "-k", "--topk",
        type=int, default=5, help="number of highest prbabilities to be predicted"
    )

    parser.add_argument(
        "image_path",
        nargs="?",
        help="image path or url"
    )

    # execute
    if args.version:
        print(__version__)
        return

    if args.image_path is None:
        print("requires an image path")
        return 

    # load pytorch unless conducting inferece
    _inference(args)


def _inference(args):
    from .main import inference
    result = inference(args.image_path, topk=args.topk)

    for label, prob in result:
        print(label, prob)